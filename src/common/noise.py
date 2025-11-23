from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class BaseActionNoise(ABC):
    """
    The base for action noise.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def __call__(self) -> np.ndarray:
        """
        Generate a noise sample.
        """
        raise NotImplementedError


class NormalActionNoise(BaseActionNoise):
    """
    A Gaussian action noise.

    :param mean: The mean of the noise.
    :param std: The standard deviation of the noise.
    :param dim: The dimension of the noise.
    :param rng: The random number generator.
    """

    def __init__(
        self,
        mean: np.ndarray = 0.0,
        std: np.ndarray = 0.1,
        dim: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        super().__init__()
        self.mean = mean
        self.std = std
        self.dim = dim
        self.rng = rng if rng is not None else np.random.default_rng()

    def __call__(self, dim: Optional[int] = None) -> np.ndarray:
        """
        Generate a noise sample.

        :param dim: The dimension of the noise. If None, the dimension specified in the constructor is used.
        :return: The noise sample.
        """
        dim = self.dim if dim is None else dim
        return self.rng.normal(self.mean, self.std, size=dim).astype(np.float32)

    def reset(self) -> None:
        """
        Reset the noise.
        """
        pass


class OrnsteinUhlenbeckActionNoise(BaseActionNoise):
    """
    An Ornstein-Uhlenbeck action noise.

    :param mean: The mean of the noise.
    :param sigma: The standard deviation of the noise.
    :param theta: The rate of mean reversion.
    :param dt: The timestep.
    :param rng: The random number generator.
    """
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
        """
        Generate a noise sample.
        """
        noise = (
            self.noise_prev
            + self._theta * (self._mu - self.noise_prev) * self._dt
            + self._sigma * np.sqrt(self._dt) * self.rng.normal(size=self._mu.shape)
        )
        self.noise_prev = noise
        return noise.astype(np.float32)

    def reset(self) -> None:
        """
        Reset the noise to the initial position.
        """
        self.noise_prev = np.zeros_like(self._mu)
