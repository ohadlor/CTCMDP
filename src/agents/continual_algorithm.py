from typing import Optional

import numpy as np

from src.environments.wrappers import FrozenHiddenObservation

from .base_algorithm import BaseAlgorithm


def make_continual_learner(
    base_algorithm: type[BaseAlgorithm], gradient_steps: int = 1, batch_size: int = 256
) -> type[BaseAlgorithm]:

    class ContinualLearningAlgorithm(base_algorithm):
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
            learning: bool = False,
        ) -> np.ndarray:
            """
            Get the policy action from an observation.
            Additionally, perform a training step if in continual learning mode and a sample is provided.
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
                if self.num_timesteps >= self.learning_starts and self.replay_buffer.size() > 0:
                    self.train(gradient_steps=self.gradient_steps, batch_size=self.batch_size)

            return action

    return ContinualLearningAlgorithm
