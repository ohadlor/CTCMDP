from typing import Optional

import numpy as np


from src.environments.wrappers import FrozenHiddenObservation
from .base_algorithm import BaseAlgorithm


class ContinualLearningAlgorithm(BaseAlgorithm):
    def predict(
        self,
        observation: np.ndarray,
        state: Optional[tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
        continual_gradient_steps: int = 1,
        continual_batch_size: int = 256,
        sample: Optional[dict] = None,
        stationary_env: Optional[FrozenHiddenObservation] = None,
    ) -> tuple[np.ndarray, Optional[tuple[np.ndarray, ...]]]:
        """
        Get the policy action from an observation.
        Additionally, perform a training step if in continual learning mode and a sample is provided.
        """
        action, state = super().predict(observation, state, episode_start, deterministic)

        self.stationary_env = stationary_env

        if sample is not None:
            # A transition has just occurred, add it to the replay buffer.
            self.replay_buffer.add(
                obs=sample["obs"],
                next_obs=sample["next_obs"],
                action=sample["action"],
                reward=sample["reward"],
                done=sample["done"],
            )

        # Perform a training step
        if self.num_timesteps >= self.learning_starts and self.replay_buffer.size() > 0:
            self.train(gradient_steps=continual_gradient_steps, batch_size=continual_batch_size)

        return action, state
