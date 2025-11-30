from typing import Optional

import numpy as np
import torch as th
from torch.nn import functional as F

from src.agents.td3 import TD3
from src.agents.continual_algorithm import make_continual_learner
from src.buffers.replay_buffer import TimeIndexedReplayBuffer
from src.common.utils import polyak_update
from src.common.noise import NormalActionNoise


ContinualTD3 = make_continual_learner(TD3)


class DiscountModelContinualTD3(ContinualTD3):
    """
    A continual learning version of TD3 that uses a simulator model and an additional discount to generate
    additional data for training.

    Parameters
    ----------
    p_real : float, optional
        The probability of sampling a real transition from the replay buffer, by default 0.5.
    sim_gamma : float, optional
        The discount factor to use for the simulated environment, by default 0.9.
    sim_horizon : int, optional
        The horizon to use for the simulated environment, by default 1.
    sim_action_noise_std : float, optional
        The standard deviation of the action noise to use for the simulated environment, by default 0.1.
    sim_buffer_size : int, optional
        The size of the replay buffer for the simulated environment, by default 1_000_000.
    actor_path : Optional[str], optional
        The path to a pretrained actor, by default None.
    critic_path : Optional[str], optional
        The path to a pretrained critic, by default None.
    """

    def __init__(
        self,
        p_real: Optional[float] = None,
        sim_gamma: float = 0.9,
        sim_horizon: int = 1,
        sim_action_noise_std: float = 0.3,
        sim_buffer_size: int = 1_000_000,
        actor_path: Optional[str] = None,
        critic_path: Optional[str] = None,
        *args,
        **kwargs,
    ):

        super().__init__(*args, **kwargs)
        if self.gamma is None:
            self.gamma = sim_gamma
        self.replay_buffer = self._base_to_time_indexed_buffer(
            self.replay_buffer,
            gamma=self.gamma,
        )
        self.replay_buffers = [self.replay_buffer]

        if actor_path is not None:
            self._load_actor(actor_path)
        if critic_path is not None:
            self._load_critic(critic_path)

        self.sim_action_noise = None
        if p_real is None:
            self.p_real = 1 / (sim_horizon + 1)
        self._setup_sim(sim_gamma, sim_horizon, sim_action_noise_std, sim_buffer_size)

    def train(self, gradient_steps: int = 2, batch_size: int = 100) -> tuple[float, Optional[float]]:
        if self.stationary_env is not None and hasattr(self, "sim_replay_buffer"):
            self._add_to_sim_buffer()

        self.policy.set_training_mode(True)
        actor_losses, critic_losses = [], []
        for _ in range(gradient_steps):
            self._n_updates += 1
            # Select real or simulated data, use real or sim discont factor
            use_real = self.rng.binomial(1, self.p_real) == 1 if self.stationary_env is not None else True

            if use_real:
                replay_data = self.replay_buffer.sample(batch_size)
                gamma = self.gamma
            else:
                replay_data = self.sim_replay_buffer.sample(batch_size)
                gamma = self.sim_gamma

            with th.no_grad():
                # Target actions  with clipped noise
                noise = replay_data.actions.clone().data.normal_(0, self.target_policy_noise)
                noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                next_actions = (self.actor_target(replay_data.next_observations) + noise).clamp(-1, 1)

                # Compute target
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * gamma * next_q_values

            # Calculate critic loss
            current_q_values = self.critic(replay_data.observations, replay_data.actions)
            critic_loss = sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            critic_losses.append(critic_loss.item())

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            if self._n_updates % self.policy_delay == 0:
                # Calculate actor loss
                actor_loss = -self.critic.q1_forward(
                    replay_data.observations, self.actor(replay_data.observations)
                ).mean()
                actor_losses.append(actor_loss.item())
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)

        mean_critic_loss = np.mean(critic_losses) if critic_losses else 0.0
        mean_actor_loss = np.mean(actor_losses) if actor_losses else None

        return mean_critic_loss, mean_actor_loss

    def _add_to_sim_buffer(self) -> None:
        """
        Add transitions to the simulation replay buffer.
        """
        self.sim_replay_buffer.increment_time_indices()
        # obs, _ = self.stationy_env.reset(seed=self.seed)
        # Start rollout from the last true observation
        obs = self.last_obs

        for _ in range(self.sim_horizon):
            # Create simulated rollout for buffer
            action = self.predict(obs, learning=False)
            scaled_action = self.policy.scale_action(action)

            if self.sim_action_noise is not None:
                noise = self.sim_action_noise(scaled_action.size)
                scaled_action = np.clip(scaled_action + noise, -1, 1)
                action = self.policy.unscale_action(scaled_action)

            next_obs, reward, terminated, truncated, info = self.stationary_env.step(action)
            done = terminated or truncated

            self.sim_replay_buffer.add(obs, next_obs, scaled_action, reward, done)

            obs = next_obs
            if done:
                break

    def _setup_sim(self, gamma: float, horizon: int, action_noise_std: float, buffer_size: int):
        """
        Setup the simulation environment.

        Parameters
        ----------
        gamma : float
            The discount factor to use for the simulated environment.
        horizon : int
            The horizon to use for the simulated environment.
        action_noise_std : float
            The standard deviation of the action noise to use for the simulated environment.
        buffer_size : int
            The size of the replay buffer for the simulated environment.
        """
        if horizon >= 1:
            self.sim_horizon = horizon
            self.sim_gamma = gamma
            self.sim_action_noise = NormalActionNoise(mean=0, std=action_noise_std)
            self.sim_replay_buffer = TimeIndexedReplayBuffer(
                buffer_size * self.sim_horizon,
                self.observation_space,
                self.action_space,
                self.sim_gamma,
                device=self.device,
                rng=self.rng,
            )
            self.replay_buffers.append(self.sim_replay_buffer)

    def loss_logger(self, losses: tuple[np.ndarray, np.ndarray], log_interval: int = 1):
        critic_loss, actor_loss = losses
        if self.logger and self._n_updates % log_interval == 0:
            self.logger.add_scalar("loss/critic_loss", critic_loss, self._n_updates)
            if actor_loss is not None:
                self.logger.add_scalar("loss/actor_loss", actor_loss, self._n_updates)

    def _load_actor(self, path: str) -> None:

        self.policy.load_actor(path)

    def _load_critic(self, path: str) -> None:

        self.policy.load_critic(path)
