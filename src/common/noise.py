from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class BaseActionNoise(ABC):
    """
    The base for action noise
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def __call__(self) -> np.ndarray:
        raise NotImplementedError


class NormalActionNoise(BaseActionNoise):
    """
    A Gaussian action noise
    """

    def __init__(self, mean: np.ndarray = 0.0, std: np.ndarray = 0.1, rng: Optional[np.random.Generator] = None):
        super().__init__()
        self.mean = mean
        self.std = std
        self.rng = rng if rng is not None else np.random.default_rng()

    def __call__(self, dim: int) -> np.ndarray:
        return self.rng.normal(self.mean, self.std, size=dim).astype(np.float32)

    def reset(self) -> None:
        """
        Call end of episode reset for the noise
        """
        pass


class OrnsteinUhlenbeckActionNoise(BaseActionNoise):
    def __init__(
        self,
        mean: np.ndarray,
        sigma: np.ndarray,
        theta: float = 0.15,
        dt: float = 1e-2,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self._theta = theta
        self._mu = mean
        self._sigma = sigma
        self._dt = dt
        self.noise_prev = np.zeros_like(self._mu)
        self.reset()
        self.rng = rng if rng is not None else np.random.default_rng()

        super().__init__()

    def __call__(self) -> np.ndarray:
        noise = (
            self.noise_prev
            + self._theta * (self._mu - self.noise_prev) * self._dt
            + self._sigma * np.sqrt(self._dt) * self.rng.normal(size=self._mu.shape)
        )
        self.noise_prev = noise
        return noise.astype(np.float32)

    def reset(self) -> None:
        """
        reset the Ornstein Uhlenbeck noise, to the initial position
        """
        self.noise_prev = np.zeros_like(self._mu)
