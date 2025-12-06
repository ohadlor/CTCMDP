from typing import Optional, Union
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch as th
from torch import nn

from src.networks import Actor, Critic


class M2TD3Policy:
    """
    Policy for the Min-Max Twin-Delayed Deep Deterministic Policy Gradient (M2TD3) algorithm.

    Parameters
    ----------
    observation_space : spaces.Box
        The observation space of the environment.
    action_space : spaces.Box
        The action space of the environment.
    hidden_state_space : spaces.Box
        The hidden state space of the environment.
    hidden_action_space : spaces.Box
        The hidden action space of the environment.
    lr : int
        The learning rate for the optimizers.
    net_arch : list[int], optional
        The architecture of the networks, by default [400, 300].
    activation_fn : nn.Module, optional
        The activation function to use, by default nn.ReLU.
    n_critics : int, optional
        The number of critics to use, by default 2.
    oracle_actor : bool, optional
        Whether to use an oracle actor, by default False.
    device : str, optional
        The device to use for training, by default "auto".
    optimizer_class : th.optim.Optimizer, optional
        The optimizer class to use, by default th.optim.Adam.
    optimizer_kwargs : dict, optional
        Keyword arguments for the optimizer, by default None.
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Box,
        hidden_state_space: spaces.Box,
        hidden_action_space: spaces.Box,
        lr: float,
        adversary_lr: float,
        net_arch=[400, 300],
        activation_fn: nn.Module = nn.ReLU,
        n_critics: int = 2,
        oracle_actor: bool = False,
        stacked_observation: bool = False,
        device="auto",
        optimizer_class=th.optim.Adam,
        optimizer_kwargs=None,
    ):
        self.device = device
        self.oracle_actor = oracle_actor
        self.is_stacked = stacked_observation

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

        if self.is_stacked:
            self.base_obs_size = int(self.observation_space.shape[-1] / 2)
            base_obs_space = gym.spaces.Box(
                low=self.observation_space.low[: self.base_obs_size],
                high=self.observation_space.high[: self.base_obs_size],
                dtype=np.float64,
            )
        else:
            base_obs_space = self.observation_space

        critic_obs_space = gym.spaces.Box(
            low=np.concatenate([base_obs_space.low, self.hidden_state_space.low]),
            high=np.concatenate([base_obs_space.high, self.hidden_state_space.high]),
            dtype=np.float64,
        )

        adversary_obs_space = gym.spaces.Box(
            low=np.concatenate([base_obs_space.low, self.hidden_state_space.low, self.action_space.low]),
            high=np.concatenate([base_obs_space.high, self.hidden_state_space.high, self.action_space.high]),
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
        self.critic = self.make_critic()
        self.adversary = self.make_adversary()

        self.create_targets()
        self.actor_optimizer = self.create_optimizers(self.actor.parameters(), lr, optimizer_class, optimizer_kwargs)
        self.critic_optimizer = self.create_optimizers(self.critic.parameters(), lr, optimizer_class, optimizer_kwargs)
        self.adversary_optimizer = self.create_optimizers(
            self.adversary.parameters(), adversary_lr, optimizer_class, optimizer_kwargs
        )

    def create_targets(self):
        self.actor_target = self.make_actor()
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target = self.make_critic()
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.adversary_target = self.make_adversary()
        self.adversary_target.load_state_dict(self.adversary.state_dict())

    def create_optimizers(self, parameters, lr, optimizer_class=th.optim.Adam, optimizer_kwargs=None):
        optimizer_kwargs = optimizer_kwargs or {}
        return optimizer_class(parameters, lr=lr, **optimizer_kwargs)

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
        """
        Predict a hidden action given an observation.

        Parameters
        ----------
        obs : np.ndarray
            The observation.

        Returns
        -------
        np.ndarray
            The predicted hidden action.
        """
        self.set_training_mode(False)
        with th.no_grad():
            obs = th.as_tensor(obs, device=self.device).float()
            action = self.adversary(obs).cpu().numpy().reshape((-1, *self.hidden_action_space.shape)).squeeze()
        return self.unscale_hidden_action(action)

    def predict_hidden_state(
        self, scaled_hidden_action: Union[th.Tensor, np.ndarray], hidden_state: Union[th.Tensor, np.ndarray]
    ) -> Union[th.Tensor, np.ndarray]:
        """
        Predict the next hidden state given a scaled hidden action and the current hidden state.

        Parameters
        ----------
        scaled_hidden_action : Union[th.Tensor, np.ndarray]
            The scaled hidden action.
        hidden_state : Union[th.Tensor, np.ndarray]
            The current hidden state.

        Returns
        -------
        Union[th.Tensor, np.ndarray]
            The predicted next hidden state.
        """
        unscaled_hidden_action = self.unscale_hidden_action(scaled_hidden_action)
        hidden_state = hidden_state + unscaled_hidden_action
        if isinstance(hidden_state, th.Tensor):
            return hidden_state.clamp(self.hidden_state_space_tensor[0], self.hidden_state_space_tensor[1])
        else:
            return np.clip(hidden_state, self.hidden_state_space.low, self.hidden_state_space.high)

    def unscale_action(self, squashed_action: Union[np.ndarray, th.Tensor]) -> Union[np.ndarray, th.Tensor]:
        """
        Unscale an action from [-1, 1] to the original action space.

        Parameters
        ----------
        squashed_action : Union[np.ndarray, th.Tensor]
            The action to unscale.

        Returns
        -------
        Union[np.ndarray, th.Tensor]
            The unscaled action.
        """
        if isinstance(squashed_action, th.Tensor):
            device = squashed_action.device
            action_space_low_th = th.from_numpy(self.action_space.low).to(device)
            action_space_high_th = th.from_numpy(self.action_space.high).to(device)
            return action_space_low_th + (squashed_action + 1) * 0.5 * (action_space_high_th - action_space_low_th)
        return self.action_space.low + (squashed_action + 1) * 0.5 * (self.action_space.high - self.action_space.low)

    def scale_action(self, action: np.ndarray) -> np.ndarray:
        """
        Scale an action to [-1, 1].

        Parameters
        ----------
        action : np.ndarray
            The action to scale.

        Returns
        -------
        np.ndarray
            The scaled action.
        """
        if np.array_equal(self.action_space.high, self.action_space.low):
            return np.zeros_like(action)
        scaled_action = (action - self.action_space.low) / (self.action_space.high - self.action_space.low) * 2 - 1
        self.assert_scaled(scaled_action)
        return scaled_action

    def unscale_hidden_action(
        self, squashed_hidden_action: Union[np.ndarray, th.Tensor]
    ) -> Union[np.ndarray, th.Tensor]:
        """
        Unscale a hidden action from [-1, 1] to the original hidden action space.

        Parameters
        ----------
        squashed_hidden_action : Union[np.ndarray, th.Tensor]
            The hidden action to unscale.

        Returns
        -------
        Union[np.ndarray, th.Tensor]
            The unscaled hidden action.
        """
        if isinstance(squashed_hidden_action, th.Tensor):
            device = squashed_hidden_action.device
            hidden_action_space_low_th = th.from_numpy(self.hidden_action_space.low).to(device)
            hidden_action_space_high_th = th.from_numpy(self.hidden_action_space.high).to(device)
            return hidden_action_space_low_th + (squashed_hidden_action + 1) * 0.5 * (
                hidden_action_space_high_th - hidden_action_space_low_th
            )
        return self.hidden_action_space.low + (squashed_hidden_action + 1) * 0.5 * (
            self.hidden_action_space.high - self.hidden_action_space.low
        )

    def scale_hidden_action(self, hidden_action: np.ndarray) -> np.ndarray:
        """
        Scale a hidden action to [-1, 1].

        Parameters
        ----------
        hidden_action : np.ndarray
            The hidden action to scale.

        Returns
        -------
        np.ndarray
            The scaled hidden action.
        """
        if np.array_equal(self.hidden_action_space.high, self.hidden_action_space.low):
            return np.zeros_like(hidden_action)
        scaled_hidden_action = (hidden_action - self.hidden_action_space.low) / (
            self.hidden_action_space.high - self.hidden_action_space.low
        ) * 2 - 1
        self.assert_scaled(scaled_hidden_action)
        return scaled_hidden_action

    def concat_obs_actor(
        self, observation: Union[th.Tensor, np.ndarray], hidden_state: Optional[Union[th.Tensor, np.ndarray]] = None
    ) -> Union[th.Tensor, np.ndarray]:
        """
        Concatenate the observation and hidden state for the actor if the actor is an oracle.

        Parameters
        ----------
        observation : Union[th.Tensor, np.ndarray]
            The observation.
        hidden_state : Optional[Union[th.Tensor, np.ndarray]], optional
            The hidden state, by default None.

        Returns
        -------
        Union[th.Tensor, np.ndarray]
            The concatenated observation.
        """
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
        """
        Concatenate the observation and hidden state for the critic.

        Parameters
        ----------
        observation : Union[th.Tensor, np.ndarray]
            The observation.
        hidden_state : Union[th.Tensor, np.ndarray]
            The hidden state.

        Returns
        -------
        Union[th.Tensor, np.ndarray]
            The concatenated observation.
        """
        if self.is_stacked:
            current_obs = observation[..., self.base_obs_size :]
        else:
            current_obs = observation

        if isinstance(observation, np.ndarray):
            return np.concatenate([current_obs, hidden_state], axis=-1)
        else:
            return th.cat([current_obs, hidden_state], dim=1)

    def concat_obs_adversary(
        self,
        observation: Union[th.Tensor, np.ndarray],
        hidden_state: Union[th.Tensor, np.ndarray],
        action: Union[th.Tensor, np.ndarray],
    ) -> Union[th.Tensor, np.ndarray]:
        """
        Concatenate the observation, hidden state, and action for the adversary.

        Parameters
        ----------
        observation : Union[th.Tensor, np.ndarray]
            The observation.
        hidden_state : Union[th.Tensor, np.ndarray]
            The hidden state.
        action : Union[th.Tensor, np.ndarray]
            The action.

        Returns
        -------
        Union[th.Tensor, np.ndarray]
            The concatenated observation.
        """
        if self.is_stacked:
            current_obs = observation[..., self.base_obs_size :]
        else:
            current_obs = observation

        if isinstance(observation, np.ndarray):
            return np.concatenate([current_obs, hidden_state, action], axis=-1)
        else:
            return th.cat([current_obs, hidden_state, action], dim=1)

    def load_actor(self, path: str):
        weights = th.load(path, map_location=self.device, weights_only=True)
        self.actor.load_state_dict(weights["actor"])
        self.actor_target.load_state_dict(weights["actor"])

    @staticmethod
    def assert_scaled(action: np.ndarray):
        assert np.all(action >= -1) and np.all(action <= 1), "Action must be scaled to [-1, 1]"
