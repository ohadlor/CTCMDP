from typing import Optional

import numpy as np
from gymnasium import spaces

from .base_schedule import BaseActionSchedule


class ReflectedBrownianMotionSchedule(BaseActionSchedule):
    """
    A stochastic schedule that generates hidden actions using a reflected Brownian motion.
    The motion is constrained within the specified bounds for each dimension of psi.
    """

    def __init__(
        self,
        action_space: spaces.Box,
        observation_space: spaces.Box,
        volatility: float = 0.1,
        rng: Optional[np.random.Generator] = None,
    ):
        super().__init__(action_space, observation_space, rng)
        self.volatility = volatility

    def _action_selection(self, obs: np.ndarray) -> np.ndarray:
        """
        Generates a random step and correctly reflects it if it would push psi out of bounds.
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
    def __init__(self, action_space: spaces.Box, observation_space: spaces.Box, rng: np.random.Generator):
        super().__init__(action_space, observation_space, rng)

    def _action_selection(self, obs: np.ndarray) -> np.ndarray:
        return self.action_space.sample()
