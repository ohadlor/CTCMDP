from typing import Union, Optional
from tqdm import tqdm

import numpy as np
import torch as th
from gymnasium import Env
from torch.nn import functional as F

from src.buffers import HiddenReplayBuffer
from src.common.utils import polyak_update, safe_mean

from .base_algorithm import BaseAlgorithm
from src.common.noise import NormalActionNoise
from src.policies.m2td3_policy import M2TD3Policy


class TCM2TD3(BaseAlgorithm):
    """
    Time-Constrain Min-Max Twin-Delayed Deep Deterministic Policy Gradient (TC-M2TD3).
    Algorithmically equivalent to M2TD3, time constraint comes from environment restriction on hidden action space.
    """

    def __init__(
        self,
        env: Env,
        lr: float = 3e-4,
        buffer_size: int = 1_000_000,
        learning_starts: int = 10_000,
        batch_size: int = 256,
        gamma: float = 0.99,
        tau: float = 0.005,
        gradient_steps: int = 1,
        action_noise_std: float = 0.1,
        policy_delay: int = 2,
        target_policy_noise: float = 0.2,
        target_noise_clip: float = 0.5,
        oracle_actor: bool = False,
        policy_path: Optional[str] = None,
        device: Union[th.device, str] = "auto",
        tensorboard_log: str = "runs",
        seed: Optional[int] = None,
    ):
        # Action and state spaces are dicts that are split into hidden and observed for the policy.
        # For the env they are given and recieved as dicts
        self.hidden_observation_space = env.hidden_observation_space
        self.hidden_action_space = env.hidden_action_space

        super().__init__(env.observation_space, env.action_space, lr, device, seed, tensorboard_log)
        self.env = env
        self.replay_buffer = HiddenReplayBuffer(
            buffer_size,
            self.observation_space,
            self.action_space,
            self.hidden_observation_space,
            self.device,
            self.rng,
        )
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.policy_delay = policy_delay
        self.target_policy_noise = target_policy_noise
        self.target_noise_clip = target_noise_clip
        self.oracle_actor = oracle_actor
        self.gradient_steps = gradient_steps
        self.action_noise = NormalActionNoise(mean=0, std=action_noise_std, rng=self.rng)
        self.learning_starts = learning_starts

        if policy_path is not None:
            self.load(policy_path)

        self._setup_model()
        self._make_aliases()

        self._n_updates = 0

    def _setup_model(self):
        self.policy = M2TD3Policy(
            self.observation_space,
            self.action_space,
            self.hidden_observation_space,
            self.hidden_action_space,
            self.lr,
            oracle_actor=self.oracle_actor,
            device=self.device,
        )

    def _make_aliases(self):
        self.actor = self.policy.actor
        self.adversary = self.policy.adversary
        self.critic = self.policy.critic
        self.actor_target = self.policy.actor_target
        self.adversary_target = self.policy.adversary_target
        self.critic_target = self.policy.critic_target
        self.actor_optimizer = self.policy.actor_optimizer
        self.adversary_optimizer = self.policy.adversary_optimizer
        self.critic_optimizer = self.policy.critic_optimizer

    def train(self, gradient_steps: int, batch_size: int) -> None:
        self.policy.set_training_mode(True)
        actor_losses, critic_losses, adversary_losses = [], [], []
        for _ in range(gradient_steps):
            self._n_updates += 1
            replay_data = self.replay_buffer.sample(batch_size)

            # Train critic
            with th.no_grad():
                noise = replay_data.actions.clone().data.normal_(0, self.target_policy_noise)
                noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                hidden_noise = replay_data.hidden_states.clone().data.normal_(0, self.target_policy_noise)
                hidden_noise = hidden_noise.clamp(-self.target_noise_clip, self.target_noise_clip)

                next_actions = (self.actor_target(replay_data.next_observations) + noise).clamp(-1, 1)
                next_adversary_obs = self.policy.concat_obs_adversary(
                    replay_data.next_observations, replay_data.next_hidden_states, next_actions
                )
                next_hidden_actions = (self.adversary_target(next_adversary_obs) + hidden_noise).clamp(-1, 1)

                # Next_hidden_state is a sufficient statistic for (hidden_state, hidden_action)
                next_next_hidden_states = self.policy.predict_hidden_state(
                    next_hidden_actions, replay_data.next_hidden_states
                )

                critic_obs = self.policy.concat_obs_critic(replay_data.next_observations, next_next_hidden_states)
                next_q_values = th.cat(self.critic_target(critic_obs, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            critic_obs = self.policy.concat_obs_critic(replay_data.observations, replay_data.next_hidden_states)
            current_q_values = self.critic(critic_obs, replay_data.actions)
            critic_loss = sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            critic_losses.append(critic_loss.item())

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            if self._n_updates % self.policy_delay == 0:
                # Train agent
                agent_obs = self.policy.concat_obs_actor(replay_data.observations, replay_data.hidden_states)
                agent_actions = self.actor(agent_obs)
                critic_obs = self.policy.concat_obs_critic(replay_data.observations, replay_data.next_hidden_states)
                agent_loss = -self.critic.q1_forward(critic_obs, agent_actions).mean()
                actor_losses.append(agent_loss.item())

                self.actor_optimizer.zero_grad()
                agent_loss.backward()
                self.actor_optimizer.step()

                # Train adversary
                adversary_obs = self.policy.concat_obs_adversary(
                    replay_data.observations, replay_data.hidden_states, replay_data.actions
                )
                hidden_actions = self.adversary(adversary_obs)
                next_hidden_states = self.policy.predict_hidden_state(hidden_actions, replay_data.hidden_states)
                critic_obs = self.policy.concat_obs_critic(replay_data.observations, next_hidden_states)
                adversary_loss = self.critic.q1_forward(critic_obs, replay_data.actions).mean()
                adversary_losses.append(adversary_loss.item())

                self.adversary_optimizer.zero_grad()
                adversary_loss.backward()
                self.adversary_optimizer.step()

                polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.adversary.parameters(), self.adversary_target.parameters(), self.tau)

        return safe_mean(critic_losses), safe_mean(actor_losses), safe_mean(adversary_losses)

    def sample_action(self, actor_obs: np.ndarray, num_timesteps: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Sample an action from the policy, add noise, and scale it.
        :param obs: Observation from the environment
        :param num_timesteps: The current timestep
        :return: A tuple containing:
            - The unscaled action to be used in the environment.
            - The scaled action (between -1 and 1) to be stored in the buffer.
        """
        # Get scaled action from the actor network
        if num_timesteps < self.learning_starts:
            # Sample a random action from a uniform distribution between -1 and 1
            unscaled_action = self.action_space.sample()
        else:
            unscaled_action = self.predict(actor_obs)
            unscaled_action = unscaled_action

        scaled_action = self.policy.scale_action(unscaled_action)

        # Add noise to the action (improve exploration)
        if self.action_noise is not None:
            scaled_action = np.clip(scaled_action + self.action_noise(scaled_action.size), -1, 1)

        # We store the scaled action in the buffer
        buffer_action = scaled_action
        action = self.policy.unscale_action(scaled_action)

        return action, buffer_action

    def sample_hidden_space(self, adversary_obs: np.ndarray, num_timesteps: int) -> np.ndarray:
        """
        Sample an action from the policy, add noise, and scale it.
        :param obs: Observation from the environment
        :param num_timesteps: The current timestep
        :return: A tuple containing:
            - The unscaled action to be used in the environment.
            - The scaled action (between -1 and 1) to be stored in the buffer.
        """
        # Get scaled action from the actor network
        if num_timesteps < self.learning_starts:
            # Sample a random action from a uniform distribution between -1 and 1
            unscaled_hidden_action = self.hidden_action_space.sample()
        else:
            unscaled_hidden_action = self.policy.predict_hidden_action(adversary_obs)

        scaled_hidden_action = self.policy.scale_hidden_action(unscaled_hidden_action)

        # Add noise to the action (improve exploration)
        if self.action_noise is not None:
            scaled_hidden_action = np.clip(scaled_hidden_action + self.action_noise(scaled_hidden_action.size), -1, 1)
        hidden_action = self.policy.unscale_hidden_action(scaled_hidden_action)

        return hidden_action

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """
        Get the action from the policy.
        :param obs: Observation from the environment.
        :return: The action.
        """
        return self.policy.predict(obs)

    def collect_rollouts(
        self, obs: np.ndarray, hidden_state: np.ndarray, num_timesteps: int
    ) -> tuple[np.ndarray, np.ndarray, float, bool, bool, dict]:
        """
        Collect a single step from the environment and add it to the replay buffer.

        :param obs: The current observation.
        :param num_timesteps: The total number of timesteps collected so far.
        :return: A tuple containing the new observation, the reward, terminated, truncated, and infos.
        """
        agent_obs = self.policy.concat_obs_actor(obs, hidden_state)
        unscaled_action, scaled_action = self.sample_action(agent_obs, num_timesteps)

        adversary_obs = self.policy.concat_obs_adversary(obs, hidden_state, scaled_action)
        hidden_action = self.sample_hidden_space(adversary_obs, num_timesteps)

        # Step the environment
        new_obs, new_hidden_state, reward, terminated, truncated, infos = self.env.step(unscaled_action, hidden_action)

        # Add the transition to the replay buffer
        self.replay_buffer.add(obs, new_obs, scaled_action, reward, terminated, infos, hidden_state, new_hidden_state)

        return new_obs, new_hidden_state, reward, terminated, truncated, infos

    def learn(
        self,
        total_timesteps: int,
        log_interval: int = 4,
        tb_log_name: str = "TD3",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ):
        if self.tensorboard_log is not None and self.logger is None:
            from torch.utils.tensorboard import SummaryWriter

            self.logger = SummaryWriter(log_dir=self.tensorboard_log)
        num_timesteps = 0

        # Reset the environment
        obs, hidden_state, _ = self.env.reset(seed=self.seed)

        if progress_bar:
            pbar = tqdm(total=total_timesteps)

        ep_rewards = []
        ep_len = 0

        while num_timesteps < total_timesteps:
            new_obs, new_hidden_state, reward, terminated, truncated, infos = self.collect_rollouts(
                obs, hidden_state, num_timesteps
            )
            done = terminated or truncated
            obs, hidden_state = new_obs, new_hidden_state

            ep_rewards.append(reward)
            ep_len += 1

            if progress_bar:
                pbar.update(1)

            num_timesteps += 1

            # Train the agent
            if num_timesteps >= self.learning_starts:
                critic_loss, actor_loss, adversary_loss = self.train(self.gradient_steps, self.batch_size)
                if self.logger and self._n_updates % log_interval == 0:
                    self.logger.add_scalar("loss/critic_loss", critic_loss, self._n_updates)
                    if actor_loss:
                        self.logger.add_scalar("loss/actor_loss", actor_loss, self._n_updates)
                        self.logger.add_scalar("loss/adversary_loss", adversary_loss, self._n_updates)

            # Handle episode termination
            if done:
                obs, hidden_state, _ = self.env.reset(seed=self.seed)
                self.action_noise.reset()
                if self.logger:
                    self.logger.add_scalar("rollout/ep_rew_mean", sum(ep_rewards), num_timesteps)
                    self.logger.add_scalar("rollout/ep_len_mean", ep_len, num_timesteps)
                ep_rewards = []
                ep_len = 0

            # Log progress
            if not progress_bar and log_interval is not None and num_timesteps % log_interval == 0:
                print(f"Timestep: {num_timesteps}/{total_timesteps}")

        if progress_bar:
            pbar.close()

        return

    def save(self, path: str) -> None:
        th.save(
            {
                "actor": self.actor.state_dict(),
                "adversary": self.adversary.state_dict(),
                "critic": self.critic.state_dict(),
            },
            path,
        )
