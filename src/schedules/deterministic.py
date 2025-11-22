from typing import Optional

import numpy as np
from gymnasium import spaces

from .base_schedule import BaseActionSchedule


class SinusoidalSchedule(BaseActionSchedule):
    def __init__(
        self,
        action_space: spaces.Box,
        observation_space: spaces.Box,
        period: int = 500,
        start: Optional[np.ndarray] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        super().__init__(action_space, observation_space, rng=rng)
        self.amplitude = (self.observation_space.high - self.observation_space.low) / 2
        self.constant = (self.observation_space.high + self.observation_space.low) / 2
        self.omega = 2 * np.pi / period

        if start:
            self.phase = np.arcsin((start - self.constant) / self.amplitude)
        else:
            self.phase = 0

    def _action_selection(self, obs: np.ndarray) -> np.ndarray:
        action = self.amplitude * self.omega * np.cos(self.omega * self.t + self.phase)
        return self.clip_to_action_space(action)


class StaticSchedule(BaseActionSchedule):
    def __init__(
        self,
        action_space: spaces.Box,
        observation_space: spaces.Box,
        rng: Optional[np.random.Generator] = None,
    ):
        super().__init__(action_space, observation_space, rng=rng)
        self.action = np.zeros_like(self.action_space)

    def _action_selection(self, obs: np.ndarray) -> np.ndarray:
        return self.action


class LinearSchedule(BaseActionSchedule):
    def __init__(
        self,
        action_space: spaces.Box,
        observation_space: spaces.Box,
        steps: int = 100,
        rng: Optional[np.random.Generator] = None,
    ):
        super().__init__(action_space, observation_space, rng=rng)
        self.steps = steps
        self.start = self.observation_space.low
        self.end = self.observation_space.high
        action = (self.end - self.start) * 1 / self.steps
        self.action = self.clip_to_action_space(action)

    def _action_selection(self, obs: np.ndarray) -> np.ndarray:
        return self.action
