from typing import Optional
from abc import ABC

import numpy as np

from src.environments.wrappers import FrozenHiddenObservation
from src.buffers.replay_buffer import TimeIndexedReplayBuffer, BaseBuffer

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

    class ContinualLearningAlgorithm(base_algorithm, ABC):
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
            self.args = args
            self.kwargs = kwargs

            self.num_timesteps = 0
            self.gradient_steps = gradient_steps
            self.batch_size = batch_size

            self.stationary_env = None
            # Used for simulation
            self.last_obs = None
            self.replay_buffers: list[BaseBuffer] = [self.replay_buffer]

        @staticmethod
        def _base_to_time_indexed_buffer(buffer: BaseBuffer) -> TimeIndexedReplayBuffer:
            return TimeIndexedReplayBuffer(
                buffer.buffer_size,
                buffer.observation_space,
                buffer.action_space,
                beta=0.5,
                device=buffer.device,
                rng=buffer.rng,
            )

        def predict(
            self,
            observation: np.ndarray,
            log_interval: int = 1,
        ) -> np.ndarray:
            """
            Get the policy action from an observation.
            Additionally, perform a training step if in continual learning mode and a sample is provided.

            Parameters
            ----------
            observation : np.ndarray
                The observation to get the action from.

            Returns
            -------
            np.ndarray
                The policy action.
            """
            action, _ = self.sample_action(observation, self.num_timesteps)

            self.misc_logger(log_interval)
            return action

        def learn(
            self,
            observation: np.ndarray,
            stationary_env: Optional[FrozenHiddenObservation] = None,
            log_interval: int = 1,
        ):
            """Learning function for use during evaluation for continual learning

            Parameters
            ----------
            observation : np.ndarray
                _description_
            stationary_env : Optional[FrozenHiddenObservation], optional
                The stationary environment to use for training, by default None.
            log_interval : int, optional
                _description_, by default 1
            """

            self.num_timesteps += 1
            self.last_obs = observation
            self.stationary_env = stationary_env

            # Perform a training step
            if self.num_timesteps >= self.learning_starts and self.replay_buffer.size > 0:
                losses = self.train(gradient_steps=self.gradient_steps, batch_size=self.batch_size)
                self.loss_logger(losses, log_interval)

        def add(self, sample: Optional[dict] = None, truncated: bool = False):
            if sample is not None:
                # Actions are given from real experience, need to be scaled
                sample["action"] = self.policy.scale_action(sample["action"])
                self.replay_buffer.add(**sample)
                done = sample["done"] or truncated
                if done:
                    for buffer in self.replay_buffers:
                        if hasattr(buffer, "reset_delay"):
                            buffer.reset_from_env()

        def reset(self, seed: Optional[int] = None) -> None:
            self.set_seed(seed)
            for buffer in self.replay_buffers:
                buffer.reset(seed)
            self.policy.reset()
            # Redefine aliases to point to reset policy (critical)
            self._make_aliases()

        def sample_action(self, obs: np.ndarray, num_timesteps: int) -> tuple[np.ndarray, np.ndarray]:
            """
            Sample an action from the policy, add noise, and scale it.
            :param obs: Observation from the environment
            :param num_timesteps: The current timestep
            :return: A tuple containing:
                - The unscaled action to be used in the environment.
                - The scaled action (between -1 and 1) to be stored in the buffer.
            """
            if num_timesteps < self.learning_starts:
                unscaled_action = self.action_space.sample()
            else:
                unscaled_action = super().predict(obs)

            scaled_action = self.policy.scale_action(unscaled_action)

            # Add noise to the action (improve exploration)
            if self.action_noise is not None:
                scaled_action = np.clip(scaled_action + self.action_noise(), -1, 1)

            # We store the scaled action in the buffer
            buffer_action = scaled_action
            action = self.policy.unscale_action(scaled_action)

            return action, buffer_action

        def misc_logger(self, log_interval: int = 1):
            pass

        def loss_logger(self, losses: list[np.ndarray], log_interval: int = 1):
            pass

    return ContinualLearningAlgorithm
