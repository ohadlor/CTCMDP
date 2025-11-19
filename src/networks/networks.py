import torch as th
from torch import nn
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


class Critic(nn.Module):
    def __init__(self, observation_space: spaces.Box, action_space: spaces.Box, net_arch=[256, 256], n_critics=2):
        super().__init__()
        obs_dim = observation_space.shape[0]
        action_dim = action_space.shape[0]

        self.q_networks = nn.ModuleList()
        for _ in range(n_critics):
            layers = []
            last_dim = obs_dim + action_dim
            for layer_dim in net_arch:
                layers.append(nn.Linear(last_dim, layer_dim))
                layers.append(nn.ReLU())
                last_dim = layer_dim
            layers.append(nn.Linear(last_dim, 1))
            q_net = nn.Sequential(*layers)
            self.q_networks.append(q_net)

        self.optimizer = None

    def forward(self, obs: th.Tensor, action: th.Tensor) -> list[th.Tensor]:
        q_in = th.cat([obs, action], dim=1)
        return [q_net(q_in) for q_net in self.q_networks]

    def q1_forward(self, obs: th.Tensor, action: th.Tensor) -> th.Tensor:
        q_in = th.cat([obs, action], dim=1)
        return self.q_networks[0](q_in)
