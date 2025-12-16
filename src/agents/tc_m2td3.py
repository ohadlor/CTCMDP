from typing import Union, Optional

import numpy as np
import torch as th
from gymnasium import Env
from torch.nn import functional as F
from torch.amp import GradScaler, autocast

from src.buffers import HiddenReplayBuffer
from src.common.utils import polyak_update
from src.common.noise import NormalActionNoise
from src.policies.m2td3_policy import M2TD3Policy
from src.environments.env_utils import find_attribute_in_stack

from .base_algorithm import BaseAlgorithm


class TCM2TD3(BaseAlgorithm):
    """
    Time-Constrain Min-Max Twin-Delayed Deep Deterministic Policy Gradient (TC-M2TD3).
    Algorithmically equivalent to M2TD3, time constraint comes from environment restriction on hidden action space.

    :param env: The environment to learn from.
    :param lr: The learning rate for the optimizers.
    :param buffer_size: The size of the replay buffer.
    :param learning_starts: The number of steps before learning starts.
    :param batch_size: The size of the batches for training.
    :param gamma: The discount factor.
    :param tau: The soft update coefficient.
    :param gradient_steps: The number of gradient steps to take after each rollout.
    :param action_noise_std: The standard deviation of the action noise.
    :param policy_delay: The number of steps to wait before updating the policy.
    :param target_policy_noise: The standard deviation of the target policy noise.
    :param target_noise_clip: The clip value for the target policy noise.
    :param oracle_actor: Whether to use an oracle actor.
    :param device: The device to use for training.
    :param tensorboard_log: The path to the tensorboard log directory.
    :param seed: The seed for the random number generator.
    :param use_amp: Whether to use automatic mixed precision.
    """

    def __init__(
        self,
        env: Env,
        lr: float = 3e-4,
        adversary_lr: float = 3e-4,
        buffer_size: int = 1_000_000,
        learning_starts: int = 10_000,
        batch_size: int = 256,
        gamma: float = 0.99,
        tau: float = 0.005,
        gradient_steps: int = 1,
        action_noise_std: float = 0.1,
        adversary_noise_std: float = 0.1,
        policy_delay: int = 2,
        target_policy_noise: float = 0.2,
        target_noise_clip: float = 0.5,
        oracle_actor: bool = False,
        actor_path: Optional[str] = None,
        device: Union[th.device, str] = "auto",
        tensorboard_log: str = "runs",
        checkpoint_path: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        # Action and state spaces are split into hidden and not hidden
        self.hidden_observation_space = env.hidden_observation_space
        self.hidden_action_space = env.hidden_action_space

        super().__init__(env.observation_space, env.action_space, lr, device, seed, tensorboard_log, checkpoint_path)
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
        self.adversary_lr = adversary_lr
        self.adversary_noise_factor = adversary_noise_std / action_noise_std

        # Implementation currently only supports stack size of 2
        self.is_stacked = find_attribute_in_stack(self.env, "num_stack", 1) == 2
        self._setup_model()
        self._make_aliases()

        self.actor_path = actor_path
        if actor_path is not None:
            self._load_actor(actor_path)

        self._n_updates = 0
        self.scaler = GradScaler()

    def _setup_model(self):
        self.policy = M2TD3Policy(
            self.observation_space,
            self.action_space,
            self.hidden_observation_space,
            self.hidden_action_space,
            self.lr,
            self.adversary_lr,
            oracle_actor=self.oracle_actor,
            stacked_observation=self.is_stacked,
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

    def train(self, gradient_steps: int, batch_size: int) -> tuple[float, Optional[float], Optional[float]]:
        """
        Train the agent for a given number of gradient steps.

        :param gradient_steps: The number of gradient steps to perform.
        :param batch_size: The batch size to use for training.
        :return: A tuple containing the mean critic loss, the mean actor loss, and the mean adversary loss.
        """
        self.policy.set_training_mode(True)
        actor_losses, critic_losses, adversary_losses = [], [], []
        for _ in range(gradient_steps):
            self._n_updates += 1
            replay_data = self.replay_buffer.sample(batch_size)

            # Train critic
            with autocast(device_type=self.device):
                with th.no_grad():
                    noise = (th.rand_like(replay_data.actions) * self.target_policy_noise).clamp(
                        -self.target_noise_clip, self.target_noise_clip
                    )
                    hidden_noise = (th.rand_like(replay_data.hidden_states) * self.target_policy_noise).clamp(
                        -self.target_noise_clip, self.target_noise_clip
                    )

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
            self.scaler.scale(critic_loss).backward()
            self.scaler.step(self.critic_optimizer)

            agent_update = self._n_updates % self.policy_delay == 0
            if agent_update:
                # Train agent
                with autocast(device_type=self.device):
                    agent_obs = self.policy.concat_obs_actor(replay_data.observations, replay_data.hidden_states)
                    agent_actions = self.actor(agent_obs)
                    critic_obs = self.policy.concat_obs_critic(replay_data.observations, replay_data.next_hidden_states)
                    agent_loss = -self.critic.q1_forward(critic_obs, agent_actions).mean()
                    actor_losses.append(agent_loss.item())

                self.actor_optimizer.zero_grad()
                self.scaler.scale(agent_loss).backward()
                self.scaler.step(self.actor_optimizer)

                # Train adversary
                with autocast(device_type=self.device):
                    adversary_obs = self.policy.concat_obs_adversary(
                        replay_data.observations, replay_data.hidden_states, replay_data.actions
                    )
                    hidden_actions = self.adversary(adversary_obs)
                    next_hidden_states = self.policy.predict_hidden_state(hidden_actions, replay_data.hidden_states)
                    critic_obs = self.policy.concat_obs_critic(replay_data.observations, next_hidden_states)
                    adversary_loss = self.critic.q1_forward(critic_obs, replay_data.actions).mean()
                    adversary_losses.append(adversary_loss.item())

                self.adversary_optimizer.zero_grad()
                self.scaler.scale(adversary_loss).backward()
                self.scaler.step(self.adversary_optimizer)

            self.scaler.update()

            if agent_update:
                polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.adversary.parameters(), self.adversary_target.parameters(), self.tau)

        mean_critic_loss = np.mean(critic_losses) if critic_losses else 0.0
        mean_actor_loss = np.mean(actor_losses) if actor_losses else None
        mean_adversary_loss = np.mean(adversary_losses) if adversary_losses else None

        return mean_critic_loss, mean_actor_loss, mean_adversary_loss

    def sample_action(self, actor_obs: np.ndarray, num_timestep: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Sample an action from the policy, add noise, and scale it.
        :param actor_obs: Observation from the environment for the actor.
        :param num_timestep: The current timestep.
        :return: A tuple containing:
            - The unscaled action to be used in the environment.
            - The scaled action (between -1 and 1) to be stored in the buffer.
        """

        if num_timestep < self.learning_starts:
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

    def sample_hidden_space(self, adversary_obs: np.ndarray, num_timestep: int) -> np.ndarray:
        """
        Sample a hidden action from the policy, add noise, and scale it.
        :param adversary_obs: Observation from the environment for the adversary.
        :param num_timestep: The current timestep.
        :return: The unscaled hidden action to be used in the environment.
        """

        if num_timestep < self.learning_starts:
            unscaled_hidden_action = self.hidden_action_space.sample()
        else:
            unscaled_hidden_action = self.policy.predict_hidden_action(adversary_obs)

        scaled_hidden_action = self.policy.scale_hidden_action(unscaled_hidden_action)

        # Add noise to the action (improve exploration)
        if self.action_noise is not None:
            scaled_hidden_action = np.clip(
                scaled_hidden_action + self.adversary_noise_factor * self.action_noise(scaled_hidden_action.size),
                -1,
                1,
            )
        hidden_action = self.policy.unscale_hidden_action(scaled_hidden_action)

        return hidden_action

    def predict(self, obs: np.ndarray) -> np.ndarray:
        return self.policy.predict(obs)

    def collect_rollouts(
        self, obs: np.ndarray, hidden_state: np.ndarray, num_timestep: int
    ) -> tuple[np.ndarray, np.ndarray, float, bool, bool, dict]:
        """
        Collect a single step from the environment and add it to the replay buffer.

        :param obs: The current observation.
        :param hidden_state: The current hidden state.
        :param num_timestep: The current time step.
        :return: A tuple containing the new obs, the new hidden state, the reward, terminated, truncated, and infos.
        """
        agent_obs = self.policy.concat_obs_actor(obs, hidden_state)
        unscaled_action, scaled_action = self.sample_action(agent_obs, num_timestep)

        adversary_obs = self.policy.concat_obs_adversary(obs, hidden_state, scaled_action)
        hidden_action = self.sample_hidden_space(adversary_obs, num_timestep)

        new_obs, new_hidden_state, reward, terminated, truncated, infos = self.env.step(unscaled_action, hidden_action)

        self.replay_buffer.add(obs, new_obs, scaled_action, reward, terminated, hidden_state, new_hidden_state)

        return new_obs, new_hidden_state, reward, terminated, truncated, infos

    def learn(
        self,
        total_timesteps: int,
        log_interval: int = 1,
    ):
        """
        Train the agent for a given number of timesteps.

        :param total_timesteps: The total number of timesteps to train for.
        :param log_interval: The number of timesteps between each log.
        """
        if self.tensorboard_log is not None and self.logger is None:
            from torch.utils.tensorboard import SummaryWriter

            self.logger = SummaryWriter(log_dir=self.tensorboard_log)
        num_timesteps = 0

        obs, hidden_state, _ = self.env.reset(seed=self.seed)

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
            num_timesteps += 1

            # Train the agent
            if num_timesteps >= self.learning_starts:
                critic_loss, actor_loss, adversary_loss = self.train(self.gradient_steps, self.batch_size)
                if self.logger and self._n_updates % log_interval == 0:
                    self.logger.add_scalar("loss/critic_loss", critic_loss, self._n_updates)
                    if actor_loss is not None:
                        self.logger.add_scalar("loss/actor_loss", actor_loss, self._n_updates)
                        self.logger.add_scalar("loss/adversary_loss", adversary_loss, self._n_updates)

            # Handle episode termination
            if done:
                obs, hidden_state, _ = self.env.reset()
                self.action_noise.reset()
                if self.logger:
                    self.logger.add_scalar("rollout/ep_rew_mean", sum(ep_rewards), num_timesteps)
                    self.logger.add_scalar("rollout/ep_len_mean", ep_len, num_timesteps)
                ep_rewards = []
                ep_len = 0

            # Checkpoint agent every 5e5 steps
            if num_timesteps % 5e5 == 0:
                self.save(self.checkpoint_path + f"_{num_timesteps}.pth")

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

    def _load_actor(self, path: str) -> None:
        self.policy.load_actor(path)
