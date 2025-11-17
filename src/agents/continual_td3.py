from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
from torch.nn import functional as F

from src.agents.td3 import TD3
from src.agents.continual_algorithm import ContinualLearningAlgorithm
from src.buffers.replay_buffer import TimeIndexedReplayBuffer, BaseReplayBufferSamples


@dataclass
class SimParams:
    gamma: float = 0.99
    horizon: int = 1
    action_noise_std: float = 0.1
    p_real: float = 1.0
    replay_buffer_kwargs: Optional[dict[str, Any]] = None
    buffer_size: int = 1_000_000


class ContinualTD3(ContinualLearningAlgorithm, TD3):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


class DiscountModelContinualTD3(ContinualTD3):
    def __init__(
        self,
        sim_params: SimParams,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.rng = np.random.default_rng(self.seed)

        self.p_real = sim_params.p_real

        if self.p_real < 1.0:
            self._setup_sim(sim_params)

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Update learning rate according to schedule
        # self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

        if self.stationary_env is not None:
            self._add_to_sim_buffer()

        actor_losses, critic_losses = [], []

        for _ in range(gradient_steps):
            self._n_updates += 1
            use_real = self.rng.binomial(1, self.p_real) == 1 if self.stationary_env is not None else True

            if use_real:
                replay_data = self.replay_buffer.sample(batch_size)
            else:
                replay_data = self.sim_replay_buffer.sample(batch_size)

            self._update_policy(replay_data, actor_losses, critic_losses)

        if self.logger:
            self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
            if len(actor_losses) > 0:
                self.logger.record("train/actor_loss", np.mean(actor_losses))
            self.logger.record("train/critic_loss", np.mean(critic_losses))

    def _update_policy(
        self, replay_data: BaseReplayBufferSamples, actor_losses: list[float], critic_losses: list[float]
    ) -> None:

        with torch.no_grad():
            # Select action according to policy and add some noise
            noise = replay_data.actions.clone().data.normal_(0, self.target_policy_noise)
            noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
            next_actions = (self.actor_target(replay_data.next_observations) + noise).clamp(-1, 1)

            # Compute the target Q value
            target_q1, target_q2 = self.critic_target(replay_data.next_observations, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * target_q

        # Get current Q estimates
        current_q_values = self.critic(replay_data.observations, replay_data.actions)

        # Compute critic loss
        critic_loss = sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
        critic_losses.append(critic_loss.item())

        # Optimize the critic
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

        # Delayed policy updates
        if self._n_updates % self.policy_delay == 0:
            # Compute actor loss
            actor_loss = -self.critic.q1_forward(replay_data.observations, self.actor(replay_data.observations)).mean()
            actor_losses.append(actor_loss.item())

            # Optimize the actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            self._update_target_networks()

    def _update_target_networks(self) -> None:
        with torch.no_grad():
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.mul_(1 - self.tau)
                torch.add(target_param.data, param.data, alpha=self.tau, out=target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.mul_(1 - self.tau)
                torch.add(target_param.data, param.data, alpha=self.tau, out=target_param.data)

    def _add_to_sim_buffer(self) -> None:
        # TODO: check obs consistency (dict vs array)
        self.sim_replay_buffer.increment_time_indices()
        obs_dict = self.stationary_env.last_observation

        for _ in range(self.sim_horizon):
            obs = obs_dict["observed"]
            with torch.no_grad():
                # Convert to a batch of 1
                obs_tensor = torch.as_tensor(obs, device=self.device).unsqueeze(0)
                action_tensor = self.actor(obs_tensor)
                action = action_tensor.cpu().numpy()[0]

                if self.sim_action_noise is not None:
                    noise = self.sim_action_noise()
                    action = np.clip(action + noise, -1, 1)

            next_obs_dict, reward, terminated, truncated, info = self.stationary_env.step(action)
            done = terminated or truncated

            next_obs = next_obs_dict["observed"]

            # The sim_replay_buffer is for a single env, so we need to format arguments correctly.
            self.sim_replay_buffer.add(obs, next_obs, action, np.array([reward]), np.array([done]), [info])

            obs_dict = next_obs_dict
            if done:
                break

    def _setup_sim(self, sim_params: SimParams):
        if self.p_real < 1.0:
            self.sim_horizon = sim_params.horizon
            self.sim_action_noise = sim_params.action_noise
            self.sim_replay_buffer = TimeIndexedReplayBuffer(
                sim_params.buffer_size * self.sim_horizon,
                self.observation_space,
                self.action_space,
                sim_params.gamma,
                device=self.device,
                **sim_params.replay_buffer_kwargs,
            )
