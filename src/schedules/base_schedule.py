from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from gymnasium import spaces


class BaseActionSchedule(ABC):
    """
    Base class for action schedules.

    Parameters
    ----------
    action_space : spaces.Box
        The action space of the schedule.
    observation_space : spaces.Box
        The observation space of the schedule.
    seed : Optional[int], optional
        The seed for the random number generator, by default None.
    """

    def __init__(self, action_space: spaces.Box, observation_space: spaces.Box, seed: Optional[int] = None):
        self.action_space = action_space
        self.observation_space = observation_space
        self.rng = np.random.default_rng(seed)

        self.radius = action_space.high[0]
        # internal time-step
        self.t = 0

    def step(self, obs: np.ndarray) -> np.ndarray:
        """
        Take a step in the schedule.

        Parameters
        ----------
        obs : np.ndarray
            The current observation.

        Returns
        -------
        np.ndarray
            The selected action.
        """
        action = self._action_selection(obs)
        self.t += 1
        assert self.observation_space.contains(
            (obs + action).astype(self.observation_space.dtype)
        ), "New observation not in observation space"
        return action

    def clip_to_action_space(self, action: np.ndarray) -> np.ndarray:
        """
        Clip an action to the action space.

        Parameters
        ----------
        action : np.ndarray
            The action to clip.

        Returns
        -------
        np.ndarray
            The clipped action.
        """
        return self._clip_to_box(action, self.action_space)

    def clip_to_observation_space(self, observation: np.ndarray) -> np.ndarray:
        """
        Clip an observation to the observation space.

        Parameters
        ----------
        observation : np.ndarray
            The observation to clip.

        Returns
        -------
        np.ndarray
            The clipped observation.
        """
        return self._clip_to_box(observation, self.observation_space)

    @staticmethod
    def _clip_to_box(sample: np.ndarray, target_space: spaces.Box) -> np.ndarray:
        """
        Clip a sample to a given box space.

        Parameters
        ----------
        sample : np.ndarray
            The sample to clip.
        target_space : spaces.Box
            The space to clip to.

        Returns
        -------
        np.ndarray
            The clipped sample.
        """
        clipped_obs = np.clip(sample, a_min=target_space.low, a_max=target_space.high)
        return clipped_obs.astype(target_space.dtype)

    @abstractmethod
    def _action_selection(self, obs: spaces.Box) -> np.ndarray:
        """
        Returns the next hidden action.

        Parameters
        ----------
        obs : spaces.Box
            The current observation.

        Returns
        -------
        np.ndarray
            The selected action.
        """
        pass

    def reset(self, start_state: np.ndarray) -> None:
        """
        Reset internal state if the selector is stateful.
        """
        self.t = 0
        self._set_start(start_state)

    def _set_start(self, start_state: np.ndarray):
        pass

    def scale_action(self, unscaled_action: np.ndarray) -> np.ndarray:
        """
        Scale an action to [-1, 1].

        Parameters
        ----------
        unscaled_action : np.ndarray
            The action to scale.

        Returns
        -------
        np.ndarray
            The scaled action.
        """
        return (unscaled_action - self.action_space.low) / (self.action_space.high - self.action_space.low) * 2 - 1

    def unscale_action(self, scaled_action: np.ndarray) -> np.ndarray:
        """
        Unscale an action from [-1, 1] to the original action space.

        Parameters
        ----------
        scaled_action : np.ndarray
            The action to unscale.

        Returns
        -------
        np.ndarray
            The unscaled action.
        """
        return self.action_space.low + (scaled_action + 1) * 0.5 * (self.action_space.high - self.action_space.low)

    def set_seed(self, seed: int):
        """
        Set the seed for the random number generator.

        Parameters
        ----------
        seed : int
            The seed to use.
        """
        self.rng = np.random.default_rng(seed)
