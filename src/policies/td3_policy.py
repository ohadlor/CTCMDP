from typing import Union

import torch as th
from torch import nn
import numpy as np
from gymnasium import spaces

from src.networks import Actor, Critic


class TD3Policy:
    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Box,
        lr: int,
        net_arch=[400, 300],
        activation_fn=nn.ReLU,
        n_critics=2,
        device="auto",
        optimizer_class=th.optim.Adam,
        optimizer_kwargs=None,
    ):
        self.device = device

        # Spaces
        self.observation_space = observation_space
        self.action_space = action_space

        # Networks
        self.actor_kwargs = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "net_arch": net_arch,
            "activation_fn": activation_fn,
        }
        self.critic_kwargs = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "net_arch": net_arch,
            "n_critics": n_critics,
        }
        self.actor = self.make_actor()
        self.actor_target = self.make_actor()
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = self.make_critic()
        self.critic_target = self.make_critic()
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        optimizer_kwargs = optimizer_kwargs or {}
        self.actor_optimizer = optimizer_class(self.actor.parameters(), lr=lr, **optimizer_kwargs)
        self.critic_optimizer = optimizer_class(self.critic.parameters(), lr=lr, **optimizer_kwargs)

    def set_training_mode(self, mode: bool):
        self.actor.train(mode)
        self.critic.train(mode)
        self.training = mode

    def make_critic(self):
        return Critic(**self.critic_kwargs).to(self.device)

    def make_actor(self):
        return Actor(**self.actor_kwargs).to(self.device)

    def predict(self, obs: np.ndarray) -> np.ndarray:
        self.set_training_mode(False)
        with th.no_grad():
            obs = th.as_tensor(obs, device=self.device).float()
            action = self.actor(obs).cpu().numpy().reshape((-1, *self.action_space.shape)).squeeze()
        return self.unscale_action(action)

    def unscale_action(self, squashed_action: Union[np.ndarray, th.Tensor]) -> Union[np.ndarray, th.Tensor]:
        if isinstance(squashed_action, th.Tensor):
            device = squashed_action.device
            action_space_low_th = th.from_numpy(self.action_space.low).to(device)
            action_space_high_th = th.from_numpy(self.action_space.high).to(device)
            return action_space_low_th + (squashed_action + 1) * 0.5 * (action_space_high_th - action_space_low_th)
        return self.action_space.low + (squashed_action + 1) * 0.5 * (self.action_space.high - self.action_space.low)

    def scale_action(self, action: np.ndarray) -> np.ndarray:
        if np.array_equal(self.action_space.high, self.action_space.low):
            return np.zeros_like(action)
        return (action - self.action_space.low) / (self.action_space.high - self.action_space.low) * 2 - 1

    def load_critic(self, path: str):
        weights = th.load(path, map_location=self.device, weights_only=True)
        self.critic.load_state_dict(weights["critic"])
        self.critic_target.load_state_dict(weights["critic"])

    def load_actor(self, path: str):
        weights = th.load(path, map_location=self.device, weights_only=True)
        self.actor.load_state_dict(weights["actor"])
        self.actor_target.load_state_dict(weights["actor"])

    def save(self, path: str):
        pass
