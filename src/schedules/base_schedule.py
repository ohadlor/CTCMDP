from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from gymnasium import spaces


class BaseActionSchedule(ABC):
    def __init__(
        self, action_space: spaces.Box, observation_space: spaces.Box, rng: Optional[np.random.Generator] = None
    ):
        self.action_space = action_space
        self.observation_space = observation_space
        self.rng = np.random.default_rng() if rng is None else rng

        self.radius = action_space.high[0]
        # internal time-step
        self.t = 0

    def step(self, obs: np.ndarray) -> np.ndarray:
        action = self._action_selection(obs)
        self.t += 1
        assert self.observation_space.contains(obs + action)
        return action

    def clip_to_action_space(self, action: np.ndarray) -> np.ndarray:
        return self._clip_to_box(action, self.action_space)

    def clip_to_observation_space(self, observation: np.ndarray) -> np.ndarray:
        return self._clip_to_box(observation, self.observation_space)

    @staticmethod
    def _clip_to_box(sample: np.ndarray, target_space: spaces.Box) -> np.ndarray:
        clipped_obs = np.clip(sample, a_min=target_space.low, a_max=target_space.high)
        return clipped_obs.astype(target_space.dtype)

    @abstractmethod
    def _action_selection(self, obs: spaces.Box) -> np.ndarray:
        """Returns the next hidden action."""
        pass

    def reset(self) -> np.ndarray:
        """Reset internal state if the selector is stateful."""
        self.t = 0
