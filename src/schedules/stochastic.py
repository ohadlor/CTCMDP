import numpy as np
from numpy.random import Generator
from gymnasium import spaces


from .base_schedule import BaseSchedule


class ReflectedBrownianMotionSchedule(BaseSchedule):
    """
    A stochastic schedule that generates hidden actions using a reflected Brownian motion.
    The motion is constrained within the specified bounds for each dimension of psi.
    """

    def __init__(
        self,
        hidden_dim: int,
        l2_radius: float,
        rng: Generator,
        volatility: float = 0.1,
        bounds: tuple[float, float] = (-1.0, 1.0),
        to_clip: bool = True,
    ):
        super().__init__(hidden_dim, l2_radius, rng)
        self.volatility = volatility
        self.bounds = bounds
        self.to_clip = to_clip

    def _action_selection(self, obs: spaces.Dict) -> np.ndarray:
        """
        Generates a random step and correctly reflects it if it would push psi out of bounds.
        """
        psi = obs["hidden"]
        next_psi = self.rng.normal(loc=psi, scale=self.volatility, size=self.hidden_dim)

        # Implement reflection for each dimension
        for i in range(self.hidden_dim):
            min_bound, max_bound = self.bounds
            if next_psi[i] > max_bound:
                overshoot = next_psi[i] - max_bound
                next_psi[i] = max_bound - overshoot
            elif next_psi[i] < min_bound:
                overshoot = min_bound - next_psi[i]
                next_psi[i] = min_bound + overshoot

        effective_step = next_psi - psi

        if self.to_clip:
            effective_step = self._clip_action(effective_step)
        return effective_step

    def reset(self):
        """No internal state to reset for this stateless schedule."""
        pass


class UniformRandomSchedule(BaseSchedule):
    def __init__(self, hidden_dim: int, l2_radius: float, rng: Generator):
        super().__init__(hidden_dim, l2_radius, rng)

    def _action_selection(self, obs: spaces.Dict) -> np.ndarray:
        return self.rng.uniform(low=-1.0, high=1.0, size=self.hidden_dim)
