import torch as th
from torch import nn
from gymnasium import spaces
from torch.nn import functional as F


# output is squashed, make sure to unsquash
class Actor(nn.Module):
    """
    An actor network for a DDPG-style agent.

    Parameters
    ----------
    observation_space : spaces.Box
        The observation space of the environment.
    action_space : spaces.Box
        The action space of the environment.
    net_arch : list[int], optional
        The architecture of the network, by default [256, 256].
    activation_fn : nn.Module, optional
        The activation function to use, by default nn.ReLU.
    """

    # during test time this is set to determinstic. See if this helps continual learning
    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Box,
        net_arch=[256, 256],
        activation_fn=nn.ReLU,
        noisy_linear: bool = False,
    ):
        super().__init__()
        obs_dim = observation_space.shape[0]
        action_dim = action_space.shape[0]

        if noisy_linear:
            linear_layer = NoisyLinear
        else:
            linear_layer = nn.Linear

        layers = []
        last_dim = obs_dim
        for layer_dim in net_arch:
            layers.append(linear_layer(last_dim, layer_dim))
            layers.append(activation_fn())
            last_dim = layer_dim
        layers.append(linear_layer(last_dim, action_dim))
        layers.append(nn.Tanh())

        self.net = nn.Sequential(*layers)
        self.optimizer = None

    def forward(self, obs: th.Tensor) -> th.Tensor:
        return self.net(obs)


class Critic(nn.Module):
    """
    A critic network for a DDPG-style agent.

    Parameters
    ----------
    observation_space : spaces.Box
        The observation space of the environment.
    action_space : spaces.Box
        The action space of the environment.
    net_arch : list[int], optional
        The architecture of the network, by default [256, 256].
    n_critics : int, optional
        The number of critics to use, by default 2.
    """

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


class NoisyLinear(nn.Module):
    """
    Noisy linear layer with factorized Gaussian noise.

    Based on:
    "Noisy Networks for Exploration" https://arxiv.org/abs/1706.10295

    Parameters
    ----------
    in_features : int
        Number of input features.
    out_features : int
        Number of output features.
    std_init : float, optional
        Initial standard deviation of the noise, by default 0.5.
    """

    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(th.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(th.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(th.empty(out_features))
        self.bias_sigma = nn.Parameter(th.empty(out_features))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        mu_range = 1 / th.sqrt(th.tensor(self.in_features))
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / th.sqrt(th.tensor(self.in_features)))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / th.sqrt(th.tensor(self.out_features)))

    def _scale_noise(self, size: int) -> th.Tensor:
        x = th.randn(size, device=self.weight_mu.device)
        return x.sign().mul(x.abs().sqrt())

    def forward(self, x: th.Tensor) -> th.Tensor:
        if self.training:
            epsilon_in = self._scale_noise(self.in_features)
            epsilon_out = self._scale_noise(self.out_features)
            weight_epsilon = epsilon_out.outer(epsilon_in)
            bias_epsilon = epsilon_out
            return F.linear(
                x,
                self.weight_mu + self.weight_sigma * weight_epsilon,
                self.bias_mu + self.bias_sigma * bias_epsilon,
            )
        else:
            return F.linear(x, self.weight_mu, self.bias_mu)
