from typing import Optional
import time

import numpy as np
from gymnasium import Env

from src.agents.td3 import TD3
from src.agents.continual_algorithm import make_continual_learner
from src.buffers.replay_buffer import TimeIndexedReplayBuffer
from src.common.noise import NormalActionNoise


AbstractContinualTD3 = make_continual_learner(TD3)


class ContinualTD3(AbstractContinualTD3):
    def loss_logger(self, losses: tuple[np.ndarray, np.ndarray], log_interval: int = 1):
        critic_loss, actor_loss = losses
        if self.logger and self._n_updates % log_interval == 0:
            self.logger.add_scalar("loss/critic_loss", critic_loss, self._n_updates)
            if actor_loss is not None:
                self.logger.add_scalar("loss/actor_loss", actor_loss, self._n_updates)


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
        sim_action_noise_std: float = 0.5,
        sim_buffer_size: int = 50_000,
        current_episode_multiplier: float = 1,
        max_grad_norm: float = 10,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # For copy
        self.args = args
        self.kwargs = kwargs

        if self.gamma is None:
            self.gamma = sim_gamma
        if sim_horizon >= 1:
            self.replay_buffer = self._base_to_time_indexed_buffer(self.replay_buffer, current_episode_multiplier)

        self.max_grad_norm = max_grad_norm

        self.sim_action_noise = None
        if p_real is None:
            self.p_real = 1 / (sim_horizon + 1)
        else:
            self.p_real = p_real

        self.timer = 0

        self._setup_sim(sim_gamma, sim_horizon, sim_action_noise_std, sim_buffer_size, current_episode_multiplier)

    def train(self, gradient_steps: int = 1, batch_size: int = 256) -> tuple[float, Optional[float]]:
        tf = 0
        has_sim = self.stationary_env is not None and hasattr(self, "sim_replay_buffer")
        if has_sim:
            t1 = time.time()
            self._add_to_sim_buffer()
            tf += time.time() - t1

        self.policy.set_training_mode(True)
        actor_losses, critic_losses = [], []

        for _ in range(gradient_steps):
            self._n_updates += 1
            # Select real or simulated data, use real or sim discont factor
            use_real = self.rng.binomial(1, self.p_real) == 1 if has_sim else True

            if use_real:
                replay_data = self.replay_buffer.sample(batch_size)
                gamma = self.gamma
            else:
                replay_data = self.sim_replay_buffer.sample(batch_size)
                gamma = self.sim_gamma

            critic_loss, actor_loss = self.update(replay_data, gamma)

            critic_losses.append(critic_loss)
            if actor_loss is not None:
                actor_losses.append(actor_loss)

        mean_critic_loss = np.mean(critic_losses) if critic_losses else 0.0
        mean_actor_loss = np.mean(actor_losses) if actor_losses else None

        return mean_critic_loss, mean_actor_loss, tf

    def _add_to_sim_buffer(self) -> None:
        """
        Add transitions to the simulation replay buffer.
        """
        # Start rollout from the last true observation
        obs = self.last_obs

        for _ in range(self.sim_horizon):
            # Create simulated rollout for buffer
            action = self.predict(obs)
            scaled_action = self.policy.scale_action(action)

            if self.sim_action_noise is not None:
                noise = self.sim_action_noise(scaled_action.size)
                scaled_action = np.clip(scaled_action + noise, -1, 1)
                action = self.policy.unscale_action(scaled_action)

            next_obs, reward, terminated, truncated, info = self.stationary_env.step(action)
            done = terminated or truncated

            self.sim_replay_buffer.add(obs, next_obs, scaled_action, reward, terminated)

            obs = next_obs
            if done:
                continue
        self.sim_replay_buffer.step_times()

    def _setup_sim(
        self, gamma: float, horizon: int, action_noise_std: float, buffer_size: int, current_episode_multiplier: float
    ):
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
                current_episode_multiplier=current_episode_multiplier,
                device=self.device,
                rng=self.rng,
            )
            self.replay_buffers.append(self.sim_replay_buffer)
            self.stationary_env = None
            self.last_obs = None

    def set_stationary_env(self, env: Env):
        self.stationary_env = env

    def update_stationary_env(self, env: Env):
        self.stationary_env.copy_env(env)

    def loss_logger(self, losses: tuple[np.ndarray, np.ndarray], log_interval: int = 1000):
        critic_loss, actor_loss, tf = losses
        self.timer += tf
        if self.logger and self._n_updates % log_interval == 0:
            self.logger.add_scalar("loss/critic_loss", critic_loss, self._n_updates)
            if actor_loss is not None:
                self.logger.add_scalar("loss/actor_loss", actor_loss, self._n_updates)
            self.logger.add_scalar("time/train_step", self.timer / log_interval, self._n_updates)
            self.timer = 0

    def reset(self, seed: Optional[int] = None):
        self.set_seed(seed)
        for buffer in self.replay_buffers:
            buffer.reset(seed)
        self.policy.reset()
        if self.actor_path is not None:
            self._load_actor(self.actor_path)
        if self.critic_path is not None:
            self._load_critic(self.critic_path)
        # Redefine aliases to point to reset policy (critical)
        self._make_aliases()
