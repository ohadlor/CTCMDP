from abc import ABC, abstractmethod
from typing import Optional, Union, Self

import numpy as np
import torch as th
from gymnasium import spaces
from torch.utils.tensorboard import SummaryWriter


class BaseAlgorithm(ABC):
    """
    The base class for all algorithms.

    Parameters
    ----------
    observation_space : spaces.Space
        The observation space of the environment.
    action_space : spaces.Space
        The action space of the environment.
    lr : float, optional
        The learning rate for the optimizer, by default 1e-3.
    device : Union[th.device, str], optional
        The device to use for training, by default "auto".
    seed : Optional[int], optional
        The seed for the random number generator, by default None.
    tensorboard_log : str, optional
        The path to the tensorboard log directory, by default "runs".
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
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        # th.manual_seed(seed)

        self.policy = None
        self.update_logger_path(tensorboard_log)

    def update_logger_path(self, path: str) -> SummaryWriter:
        self.logger = SummaryWriter(log_dir=path)
        return self.logger

    @abstractmethod
    def train(self, gradient_steps: int, batch_size: int) -> None:
        """
        Perform a single training step.

        Parameters
        ----------
        gradient_steps : int
            The number of gradient steps to perform.
        batch_size : int
            The batch size to use for training.
        """
        raise NotImplementedError

    @abstractmethod
    def _setup_model(self) -> None:
        """
        Setup the model (e.g., policies, critics). This is called by the constructor.
        """
        raise NotImplementedError

    def _make_aliases(self):
        """
        Create aliases for the policy's components. This is called by the constructor.
        """
        raise NotImplementedError

    def save(self, path: str) -> None:
        """
        Save the model.

        Parameters
        ----------
        path : str
            The path to save the model to.
        """
        raise NotImplementedError

    @classmethod
    def load(cls, path: str, device: Union[th.device, str] = "auto") -> Self:
        """
        Load a model from a file.

        Parameters
        ----------
        path : str
            The path to load the model from.
        device : Union[th.device, str], optional
            The device to use for training, by default "auto".

        Returns
        -------
        Self
            The loaded model.
        """
        raise NotImplementedError

    def set_seed(self, seed: int):
        """
        Set the seed for the random number generator.

        Parameters
        ----------
        seed : int
            The seed to use.
        """
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        th.manual_seed(seed)
