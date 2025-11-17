import torch as th
from torch import nn
import numpy as np
from gymnasium import spaces


# output is squashed, make sure to unsquash
class Actor(nn.Module):
    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Box,
        net_arch=[256, 256],
        activation_fn=nn.ReLU,
    ):
        super().__init__()
        obs_dim = observation_space.shape[0]
        action_dim = action_space.shape[0]

        self.action_space_low = action_space.low
        self.action_space_high = action_space.high

        layers = []
        last_dim = obs_dim
        for layer_dim in net_arch:
            layers.append(nn.Linear(last_dim, layer_dim))
            layers.append(activation_fn())
            last_dim = layer_dim
        layers.append(nn.Linear(last_dim, action_dim))
        layers.append(nn.Tanh())

        self.net = nn.Sequential(*layers)
        self.optimizer = None

    def forward(self, obs: th.Tensor) -> th.Tensor:
        return self.net(obs)

    def unsquash_action(self, squashed_action: np.ndarray) -> np.ndarray:
        return self.action_space_low + (squashed_action + 1) * 0.5 * (self.action_space_high - self.action_space_low)

    def squash_action(self, action: np.ndarray) -> np.ndarray:
        return (action - self.action_space_low) / (self.action_space_high - self.action_space_low) * 2 - 1

    def predict(self, obs: th.Tensor) -> np.ndarray:
        self.train(False)
        with th.no_grad():
            actions = self(obs)
        return self.unsquash_action(actions)


class Critic(nn.Module):
    def __init__(self, observation_space: spaces.Box, action_space: spaces.Box, net_arch=[256, 256], n_critics=2):
        super().__init__()
        obs_dim = observation_space.shape[0]
        action_dim = action_space.shape[0]

        self.q_networks = []
        for i in range(n_critics):
            layers = []
            last_dim = obs_dim + action_dim
            for layer_dim in net_arch:
                layers.append(nn.Linear(last_dim, layer_dim))
                layers.append(nn.ReLU())
                last_dim = layer_dim
            layers.append(nn.Linear(last_dim, 1))
            q_net = nn.Sequential(*layers)
            self.add_module(f"qf{i}", q_net)
            self.q_networks.append(q_net)

        self.optimizer = None

    def forward(self, obs: th.Tensor, action: th.Tensor) -> th.Tensor:
        q_in = th.cat([obs, action], dim=1)
        return [q_net(q_in) for q_net in self.q_networks]

    def q1_forward(self, obs: th.Tensor, action: th.Tensor) -> th.Tensor:
        q_in = th.cat([obs, action], dim=1)
        return self.q_networks[0](q_in)
