from typing import Optional

import numpy as np
from gymnasium import spaces

from .base_schedule import BaseActionSchedule


class ReflectedBrownianMotionSchedule(BaseActionSchedule):
    """
    A stochastic schedule that generates hidden actions using a reflected Brownian motion.
    The motion is constrained within the specified bounds for each dimension of observation space.

    Parameters
    ----------
    action_space : spaces.Box
        The action space of the schedule.
    observation_space : spaces.Box
        The observation space of the schedule.
    volatility : float, optional
        The volatility of the Brownian motion, by default 0.1.
    seed : Optional[int], optional
        The seed for the random number generator, by default None.
    """

    def __init__(
        self,
        action_space: spaces.Box,
        observation_space: spaces.Box,
        volatility: float = 0.1,
        seed: Optional[int] = None,
    ):
        super().__init__(action_space, observation_space, seed)
        self.volatility = volatility

    def _action_selection(self, obs: np.ndarray) -> np.ndarray:
        """
        Generates a random step and correctly reflects it if it would push the observation out of bounds.

        Parameters
        ----------
        obs : np.ndarray
            The current observation.

        Returns
        -------
        np.ndarray
            The selected action.
        """
        scaled_action = self.rng.normal(loc=0, scale=self.volatility, size=self.action_space.shape[0]).clip(-1, 1)
        action = self.unscale_action(scaled_action)

        next_obs = obs + action

        # Implement reflection for each dimension
        next_obs = np.where(
            next_obs > self.observation_space.high, 2 * self.observation_space.high - next_obs, next_obs
        )
        next_obs = np.where(next_obs < self.observation_space.low, 2 * self.observation_space.low - next_obs, next_obs)

        effective_action = next_obs - obs

        return effective_action


class UniformRandomSchedule(BaseActionSchedule):
    """
    A stochastic schedule that generates hidden actions using a uniform random distribution.

    Parameters
    ----------
    action_space : spaces.Box
        The action space of the schedule.
    observation_space : spaces.Box
        The observation space of the schedule.
    seed : Optional[int], optional
        The seed for the random number generator, by default None.
    """

    def __init__(self, action_space: spaces.Box, observation_space: spaces.Box, seed: Optional[int] = None):
        super().__init__(action_space, observation_space, seed)

    def _action_selection(self, obs: np.ndarray) -> np.ndarray:
        """
        Generates a random action from the action space. Similar in nature to random walk, Brownian motion

        Parameters
        ----------
        obs : np.ndarray
            The current observation (unused).

        Returns
        -------
        np.ndarray
            The selected action.
        """
        action = self.action_space.sample()
        next_obs = obs + action
        next_obs = self.clip_to_observation_space(next_obs)
        effective_action = next_obs - obs
        return effective_action
