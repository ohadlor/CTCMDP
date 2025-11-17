from abc import ABC, abstractmethod

import numpy as np
from numpy.random import Generator
from gymnasium import spaces

from .hidden_action_selector import HiddenActionSelector


class BaseSchedule(HiddenActionSelector, ABC):
    def __init__(self, hidden_dim: int, l2_radius: float, rng: Generator):
        self.hidden_dim = hidden_dim
        self.l2_radius = l2_radius
        self.rng = rng

    def _clip_action(self, action: np.ndarray) -> np.ndarray:
        """Clips the action to an L2 ball of self.l2_radius."""
        norm = np.linalg.norm(action)
        if norm > self.l2_radius:
            action = action * (self.l2_radius / norm)
        return action

    @abstractmethod
    def _action_selection(self, obs: spaces.Dict) -> np.ndarray:
        """Returns the next hidden action."""
        pass
