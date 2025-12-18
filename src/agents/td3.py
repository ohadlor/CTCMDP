from typing import Union, Optional

import numpy as np
import torch as th
from gymnasium import Env
from torch.nn import functional as F
from torch.amp import GradScaler, autocast

from src.buffers.replay_buffer import BaseBuffer, BaseReplayBufferSamples
from src.common.noise import NormalActionNoise
from src.policies import TD3Policy
from src.common.utils import polyak_update

from .base_algorithm import BaseAlgorithm


class TD3(BaseAlgorithm):
    """
    Twin Delayed Deep Deterministic Policy Gradient (TD3)
    Addressing Function Approximation Error in Actor-Critic Methods.

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
    :param device: The device to use for training.
    :param tensorboard_log: The path to the tensorboard log directory.
    :param actor_path: The path to a pretrained actor.
    :param seed: The seed for the random number generator.
    :param use_amp: Whether to use automatic mixed precision.
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
        device: Union[th.device, str] = "auto",
        tensorboard_log: str = "runs",
        checkpoint_path: Optional[str] = None,
        actor_path: Optional[str] = None,
        critic_path: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(env.observation_space, env.action_space, lr, device, seed, tensorboard_log, checkpoint_path)
        self.env = env
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.policy_delay = policy_delay
        self.target_policy_noise = target_policy_noise
        self.target_noise_clip = target_noise_clip
        self.gradient_steps = gradient_steps
        self.action_noise = NormalActionNoise(
            mean=0, std=action_noise_std, dim=self.action_space.shape[0], rng=self.rng
        )

        self.learning_starts = learning_starts
        self.replay_buffer = BaseBuffer(buffer_size, self.observation_space, self.action_space, self.device, self.rng)

        self.actor = None
        self.critic = None
        self.critic_target = None
        self.actor_optimizer = None
        self.critic_optimizer = None

        self._n_updates = 0
        self._setup_model()

        self.actor_path = actor_path
        self.critic_path = critic_path
        if actor_path is not None:
            self._load_actor(actor_path)
        if critic_path is not None:
            self._load_critic(critic_path)

        self._make_aliases()
        self.scaler = GradScaler(self.device)

    def _setup_model(self):
        self.policy = TD3Policy(
            self.observation_space,
            self.action_space,
            self.lr,
            device=self.device,
        )

    def _make_aliases(self):
        self.actor = self.policy.actor
        self.critic = self.policy.critic
        self.actor_target = self.policy.actor_target
        self.critic_target = self.policy.critic_target
        self.actor_optimizer = self.policy.actor_optimizer
        self.critic_optimizer = self.policy.critic_optimizer

    def update(self, replay_data: BaseReplayBufferSamples, gamma: float) -> tuple[float, Optional[float]]:
        actor_loss = None
        with autocast(device_type=self.device):
            with th.no_grad():
                # Target actions  with clipped noise
                noise = (th.rand_like(replay_data.actions) * self.target_policy_noise).clamp(
                    -self.target_noise_clip, self.target_noise_clip
                )
                next_actions = (self.actor_target(replay_data.next_observations) + noise).clamp(-1, 1)

                # Compute target
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * gamma * next_q_values

            # Calculate critic loss
            current_q_values = self.critic(replay_data.observations, replay_data.actions)
            critic_loss = sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)

        self.critic_optimizer.zero_grad()
        self.scaler.scale(critic_loss).backward()
        self.scaler.step(self.critic_optimizer)

        actor_update = self._n_updates % self.policy_delay == 0
        if actor_update:
            # Calculate actor loss
            with autocast(device_type=self.device):
                actor_loss = -self.critic.q1_forward(
                    replay_data.observations, self.actor(replay_data.observations)
                ).mean()
            self.actor_optimizer.zero_grad()
            self.scaler.scale(actor_loss).backward()
            self.scaler.step(self.actor_optimizer)

        self.scaler.update()

        if actor_update:
            polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)
            polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)

        return (critic_loss.item(), actor_loss.item()) if actor_loss is not None else (critic_loss.item(), actor_loss)

    def train(self, gradient_steps: int, batch_size: int) -> tuple[float, Optional[float]]:
        self.policy.set_training_mode(True)
        actor_losses, critic_losses = [], []
        for _ in range(gradient_steps):
            self._n_updates += 1
            replay_data = self.replay_buffer.sample(batch_size)
            critic_loss, actor_loss = self.update(replay_data, self.gamma)
            critic_losses.append(critic_loss)
            if actor_loss is not None:
                actor_losses.append(actor_loss)

        mean_critic_loss = np.mean(critic_losses) if critic_losses else 0.0
        mean_actor_loss = np.mean(actor_losses) if actor_losses else None

        return mean_critic_loss, mean_actor_loss

    def sample_action(self, obs: np.ndarray, num_timesteps: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Sample an action from the policy, add noise, and scale it.
        :param obs: Observation from the environment
        :param num_timesteps: The current timestep
        :return: A tuple containing:
            - The unscaled action to be used in the environment.
            - The scaled action (between -1 and 1) to be stored in the buffer.
        """
        if num_timesteps < self.learning_starts:
            unscaled_action = self.action_space.sample()
        else:
            unscaled_action = self.predict(obs)

        scaled_action = self.policy.scale_action(unscaled_action)

        # Add noise to the action (improve exploration)
        if self.action_noise is not None:
            scaled_action = np.clip(scaled_action + self.action_noise(), -1, 1)

        # We store the scaled action in the buffer
        buffer_action = scaled_action
        action = self.policy.unscale_action(scaled_action)

        return action, buffer_action

    def collect_rollouts(self, obs: np.ndarray, num_timesteps: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Collect a single step from the environment and add it to the replay buffer.

        :param obs: The current observation.
        :param num_timesteps: The total number of timesteps collected so far.
        :return: A tuple containing the new observation, the reward, terminated, truncated, and infos.
        """

        unscaled_action, scaled_action = self.sample_action(obs, num_timesteps)

        new_obs, reward, terminated, truncated, infos = self.env.step(unscaled_action)

        self.replay_buffer.add(obs, new_obs, scaled_action, reward, terminated)

        return new_obs, reward, terminated, truncated, infos

    def predict(self, obs: np.ndarray) -> np.ndarray:
        return self.policy.predict(obs)

    def learn(
        self,
        total_timesteps: int,
        log_interval: int = 100,
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

        obs, _ = self.env.reset(seed=self.seed)

        ep_rewards = []
        ep_len = 0

        while num_timesteps < total_timesteps:
            new_obs, reward, terminated, truncated, infos = self.collect_rollouts(obs, num_timesteps)
            done = terminated or truncated
            obs = new_obs

            ep_rewards.append(reward)
            ep_len += 1
            num_timesteps += 1

            # Train the agent
            if num_timesteps >= self.learning_starts:
                critic_loss, actor_loss = self.train(self.gradient_steps, self.batch_size)
                if self.logger and self._n_updates % log_interval == 0:
                    self.logger.add_scalar("loss/critic_loss", critic_loss, self._n_updates)
                    if actor_loss is not None:
                        self.logger.add_scalar("loss/actor_loss", actor_loss, self._n_updates)

            # Handle episode termination
            if done:
                obs, _ = self.env.reset()
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
                "critic": self.critic.state_dict(),
            },
            path,
        )

    def _load_actor(self, path: str) -> None:
        self.policy.load_actor(path)

    def _load_critic(self, path: str) -> None:
        self.policy.load_critic(path)
