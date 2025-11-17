from typing import Optional

import numpy as np
from numpy.random import Generator
from gymnasium import spaces


from .base_schedule import BaseSchedule


class SinusoidalSchedule(BaseSchedule):
    def __init__(self, hidden_dim: int, l2_radius: float, period: int = 500, rng: Optional[Generator] = None):
        super().__init__(hidden_dim, l2_radius, rng=rng)
        self.period = period

    def action_selection(self, obs: spaces.Dict) -> np.ndarray:
        angle = 2 * np.pi * self.t / self.period
        raw_action = self.l2_radius * np.array([np.sin(angle)] * self.hidden_dim)
        return raw_action


class StaticSchedule(BaseSchedule):
    def __init__(self, hidden_dim: int, l2_radius: float, rng: Optional[Generator] = None):
        super().__init__(hidden_dim, l2_radius, rng=rng)

    def step(self, obs: spaces.Dict) -> np.ndarray:
        return np.ones(self.hidden_dim)
