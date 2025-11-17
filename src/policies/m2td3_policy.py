from typing import Optional, Union
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch as th
from torch import nn

from src.networks import Actor, Critic


class M2TD3Policy:
    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Box,
        hidden_state_space: spaces.Box,
        hidden_action_space: spaces.Box,
        lr: int,
        net_arch=[400, 300],
        activation_fn=nn.ReLU,
        n_critics=2,
        oracle_actor=False,
        device="auto",
        optimizer_class=th.optim.Adam,
        optimizer_kwargs=None,
    ):
        self.device = device
        self.oracle_actor = oracle_actor

        # Spaces
        self.observation_space = observation_space
        self.action_space = action_space
        self.hidden_state_space = hidden_state_space
        self.hidden_action_space = hidden_action_space

        self.hidden_state_space_tensor = th.as_tensor(
            np.array([self.hidden_state_space.low, self.hidden_state_space.high]), device=self.device
        )

        if self.oracle_actor:
            actor_obs_space = gym.spaces.Box(
                low=np.concatenate([self.observation_space.low, self.hidden_state_space.low]),
                high=np.concatenate([self.observation_space.high, self.hidden_state_space.high]),
                dtype=np.float64,
            )
        else:
            actor_obs_space = self.observation_space

        critic_obs_space = gym.spaces.Box(
            low=np.concatenate([self.observation_space.low, self.hidden_state_space.low]),
            high=np.concatenate([self.observation_space.high, self.hidden_state_space.high]),
            dtype=np.float64,
        )

        adversary_obs_space = gym.spaces.Box(
            low=np.concatenate([self.observation_space.low, self.hidden_state_space.low, self.action_space.low]),
            high=np.concatenate([self.observation_space.high, self.hidden_state_space.high, self.action_space.high]),
            dtype=np.float64,
        )

        self.actor_kwargs = {
            "observation_space": actor_obs_space,
            "action_space": self.action_space,
            "net_arch": net_arch,
            "activation_fn": activation_fn,
        }
        self.critic_kwargs = {
            "observation_space": critic_obs_space,
            "action_space": self.action_space,
            "net_arch": net_arch,
            "n_critics": n_critics,
        }
        self.adversary_kwargs = {
            "observation_space": adversary_obs_space,
            "action_space": self.hidden_action_space,
            "net_arch": net_arch,
            "activation_fn": activation_fn,
        }

        # Networks
        self.actor = self.make_actor()
        self.actor_target = self.make_actor()
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = self.make_critic()
        self.critic_target = self.make_critic()
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.adversary = self.make_adversary()
        self.adversary_target = self.make_adversary()
        self.adversary_target.load_state_dict(self.adversary.state_dict())

        # Optimizers
        optimizer_kwargs = optimizer_kwargs or {}
        self.actor.optimizer = optimizer_class(self.actor.parameters(), lr=lr, **optimizer_kwargs)
        self.critic.optimizer = optimizer_class(self.critic.parameters(), lr=lr, **optimizer_kwargs)
        self.adversary.optimizer = optimizer_class(self.adversary.parameters(), lr=lr, **optimizer_kwargs)

    def set_training_mode(self, mode: bool):
        self.actor.train(mode)
        self.critic.train(mode)
        self.adversary.train(mode)

    def make_critic(self):
        return Critic(**self.critic_kwargs).to(self.device)

    def make_actor(self):
        return Actor(**self.actor_kwargs).to(self.device)

    def make_adversary(self):
        return Actor(**self.adversary_kwargs).to(self.device)

    def predict(self, obs: np.ndarray) -> np.ndarray:
        self.set_training_mode(False)
        with th.no_grad():
            obs = th.as_tensor(obs, device=self.device).float()
            action = self.actor(obs).cpu().numpy().reshape((-1, *self.action_space.shape)).squeeze()
        return self.unscale_action(action)

    def predict_hidden_action(self, obs: np.ndarray) -> np.ndarray:
        self.set_training_mode(False)
        with th.no_grad():
            obs = th.as_tensor(obs, device=self.device).float()
            action = self.adversary(obs).cpu().numpy().reshape((-1, *self.hidden_action_space.shape)).squeeze()
        return self.unscale_hidden_action(action)

    def predict_hidden_state(
        self, hidden_action: Union[th.Tensor, np.ndarray], hidden_state: Union[th.Tensor, np.ndarray]
    ) -> Union[th.Tensor, np.ndarray]:
        hidden_state = hidden_state + hidden_action
        if isinstance(hidden_state, th.Tensor):
            return hidden_state.clamp(self.hidden_state_space_tensor[0], self.hidden_state_space_tensor[1])
        else:
            return np.clip(hidden_state, self.hidden_state_space.low, self.hidden_state_space.high)

    def scale_action(self, action: np.ndarray) -> np.ndarray:
        return self.actor.squash_action(action)

    def unscale_action(self, action: np.ndarray) -> np.ndarray:
        return self.actor.unsquash_action(action)

    def scale_hidden_action(self, action: np.ndarray) -> np.ndarray:
        return self.adversary.squash_action(action)

    def unscale_hidden_action(self, action: np.ndarray) -> np.ndarray:
        return self.adversary.unsquash_action(action)

    def concat_obs_actor(
        self, observation: Union[th.Tensor, np.ndarray], hidden_state: Optional[Union[th.Tensor, np.ndarray]] = None
    ) -> Union[th.Tensor, np.ndarray]:
        if self.oracle_actor:
            assert hidden_state is not None and type(observation) is type(hidden_state)
            if isinstance(observation, np.ndarray):
                return np.concatenate([observation, hidden_state], axis=-1)
            else:
                return th.cat([observation, hidden_state], dim=1)
        else:
            return observation

    def concat_obs_critic(
        self, observation: Union[th.Tensor, np.ndarray], hidden_state: Union[th.Tensor, np.ndarray]
    ) -> Union[th.Tensor, np.ndarray]:
        assert type(observation) is type(hidden_state)
        if isinstance(observation, np.ndarray):
            return np.concatenate([observation, hidden_state], axis=-1)
        else:
            return th.cat([observation, hidden_state], dim=1)

    def concat_obs_adversary(
        self,
        observation: Union[th.Tensor, np.ndarray],
        hidden_state: Union[th.Tensor, np.ndarray],
        action: Union[th.Tensor, np.ndarray],
    ) -> Union[th.Tensor, np.ndarray]:
        assert type(observation) is type(hidden_state) and type(observation) is type(action)
        if isinstance(observation, np.ndarray):
            return np.concatenate([observation, hidden_state, action], axis=-1)
        else:
            return th.cat([observation, hidden_state, action], dim=1)
