from abc import ABC, abstractmethod

from gymnasium import spaces
import numpy as np
from numpy.random import Generator


class HiddenActionSelector(ABC):
    """
    Abstract base class for any module that selects a hidden action.
    This can be a predefined schedule or a learned policy.
    """

    def __init__(self, hidden_dim: int, l2_radius: float, rng: Generator):
        self.hidden_dim = hidden_dim
        self.l2_radius = l2_radius
        self.rng = rng

        # internal time-step
        self.t = 0

    def step(self, obs: spaces.Dict) -> np.ndarray:
        self._action_selection(obs)
        self.t += 1

    @abstractmethod
    def _action_selection(self, obs: spaces.Dict) -> np.ndarray:
        """Returns the next hidden action."""
        pass

    def reset(self) -> np.ndarray:
        """Reset internal state if the selector is stateful."""
        self.t = 0
