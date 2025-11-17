from abc import ABC, abstractmethod
from typing import Optional, Union, Self

import numpy as np
import torch as th
from gymnasium import spaces
from torch.utils.tensorboard import SummaryWriter


class BaseAlgorithm(ABC):
    """
    The base class for all algorithms.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr: float = 1e-3,
        device: Union[th.device, str] = "auto",
        seed: Optional[int] = None,
        tensorboard_log: str = "runs",
    ):
        self.observation_space = observation_space
        self.action_space = action_space
        self.lr = lr
        self.tensorboard_log = tensorboard_log
        self.device = "cuda" if th.cuda.is_available() and device == "auto" else "cpu"
        self.logger: Optional[SummaryWriter] = None
        self.rng = np.random.default_rng(seed)
        th.manual_seed(seed)

        self.policy = None

    @abstractmethod
    def train(self, gradient_steps: int, batch_size: int) -> None:
        """
        Perform a single training step.
        """
        raise NotImplementedError

    @abstractmethod
    def _setup_model(self) -> None:
        """
        Setup the model (e.g., policies, critics).
        """
        raise NotImplementedError

    def _make_aliases(self):
        """
        Create aliases for the policy's components.
        """
        raise NotImplementedError

    def set_logger(self, logger: SummaryWriter) -> None:
        """
        Set the logger.
        """
        self.logger = logger

    def save(self, path: str) -> None:
        """
        Save the model.
        """
        raise NotImplementedError

    @classmethod
    def load(cls, path: str, device: Union[th.device, str] = "auto") -> Self:
        """
        Load the model.
        """
        raise NotImplementedError
