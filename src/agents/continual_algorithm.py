from typing import Optional

import numpy as np

from src.environments.wrappers import FrozenHiddenObservation

from .base_algorithm import BaseAlgorithm


def make_continual_learner(
    base_algorithm: type[BaseAlgorithm], gradient_steps: int = 1, batch_size: int = 256
) -> type[BaseAlgorithm]:
    """
    Create a continual learning algorithm from a base algorithm.

    Parameters
    ----------
    base_algorithm : type[BaseAlgorithm]
        The base algorithm to use.
    gradient_steps : int, optional
        The number of gradient steps to perform at each training step, by default 1.
    batch_size : int, optional
        The batch size to use for training, by default 256.

    Returns
    -------
    type[BaseAlgorithm]
        A continual learning algorithm.
    """

    class ContinualLearningAlgorithm(base_algorithm):
        """
        A continual learning algorithm. This is a wrapper around a base algorithm that
        adds the ability to learn continually from a stream of data.

        Parameters
        ----------
        gradient_steps : int, optional
            The number of gradient steps to perform at each training step, by default 1.
        batch_size : int, optional
            The batch size to use for training, by default 256.
        """

        is_continual_learner = True

        def __init__(self, gradient_steps: int = gradient_steps, batch_size: int = batch_size, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.num_timesteps = 0
            self.gradient_steps = gradient_steps
            self.batch_size = batch_size

            self.stationary_env = None
            self.last_obs = None

        def predict(
            self,
            observation: np.ndarray,
            sample: Optional[dict] = None,
            stationary_env: Optional[FrozenHiddenObservation] = None,
            learning: bool = True,
            log_interval: int = 1,
        ) -> np.ndarray:
            """
            Get the policy action from an observation.
            Additionally, perform a training step if in continual learning mode and a sample is provided.

            Parameters
            ----------
            observation : np.ndarray
                The observation to get the action from.
            sample : Optional[dict], optional
                A dictionary containing the transition to add to the replay buffer, by default None.
            stationary_env : Optional[FrozenHiddenObservation], optional
                The stationary environment to use for training, by default None.
            learning : bool, optional
                Whether to perform a training step, by default True.

            Returns
            -------
            np.ndarray
                The policy action.
            """
            action = super().predict(observation)
            if learning:
                self.last_obs = observation
                self.num_timesteps += 1

                self.stationary_env = stationary_env

                if sample is not None:
                    # A transition has just occurred, add it to the replay buffer.
                    self.replay_buffer.add(**sample)

                # Perform a training step
                if self.num_timesteps >= self.learning_starts and self.replay_buffer.size > 0:
                    losses = self.train(gradient_steps=self.gradient_steps, batch_size=self.batch_size)
                    self.loss_logger(losses, log_interval)

            return action

    return ContinualLearningAlgorithm
