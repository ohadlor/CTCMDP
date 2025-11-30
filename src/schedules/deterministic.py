from typing import Optional

import numpy as np
from gymnasium import spaces

from .base_schedule import BaseActionSchedule


class SinusoidalSchedule(BaseActionSchedule):
    """
    A sinusoidal observation schedule.

    Parameters
    ----------
    action_space : spaces.Box
        The action space of the schedule.
    observation_space : spaces.Box
        The observation space of the schedule.
    period : int, optional
        The period of the sinusoid, by default 500.
    start : Optional[np.ndarray], optional
        The starting observation, by default None.
    """

    def __init__(
        self,
        action_space: spaces.Box,
        observation_space: spaces.Box,
        omega: Optional[int] = None,
        amplitude: Optional[int] = None,
    ):
        super().__init__(action_space, observation_space)
        max_amplitude = (self.observation_space.high - self.observation_space.low) / 2
        if omega is None and amplitude is None:
            # If unspecified set amplitude to half the observation space size
            amplitude = max_amplitude
        if omega is None:
            # Maximize the frequency such that radius>= 2 * amplitude* sin(omega), this comes from
            # maximizing the potential of the action space
            omega = 2 * np.arcsin(self.radius / (2 * amplitude))
        if amplitude is None:
            # Maxmimze the amplitude from the same constraint on the radius, maximizing the potential
            # of the action space
            amplitude = self.radius / (2 * np.sin(omega / 2))
            amplitude = np.minimum(amplitude, max_amplitude)

        self.constant = (self.observation_space.low + self.observation_space.high) / 2

        # Make sure obsrevations don't go out of observation space
        self.amplitude = amplitude
        self.omega = omega
        self.period = 2 * np.pi / self.omega

    def _action_selection(self, obs: np.ndarray) -> np.ndarray:
        """
        Select an action based on the current time step.

        Parameters
        ----------
        obs : np.ndarray
            The current observation.

        Returns
        -------
        np.ndarray
            The selected action.
        """
        # current_obs = self.amplitude * np.sin(self.omega * self.t + self.phase) + self.constant
        # next_obs = self.amplitude * np.sin(self.omega * (self.t + 1) + self.phase) + self.constant

        action = 2 * self.amplitude * np.sin(self.omega / 2) * np.cos(self.omega * self.t + self.phase + self.omega / 2)
        return self.clip_to_action_space(action)

    def _set_start(self, start_state: np.ndarray):
        self.phase = np.arcsin((start_state - self.constant) / self.amplitude)


class StaticSchedule(BaseActionSchedule):
    """
    A static action schedule that always returns 0, constant observation.

    Parameters
    ----------
    action_space : spaces.Box
        The action space of the schedule.
    observation_space : spaces.Box
        The observation space of the schedule.
    """

    def __init__(
        self,
        action_space: spaces.Box,
        observation_space: spaces.Box,
    ):
        super().__init__(action_space, observation_space)
        self.action = np.zeros_like(self.action_space)

    def _action_selection(self, obs: np.ndarray) -> np.ndarray:
        """
        Select an action.

        Parameters
        ----------
        obs : np.ndarray
            The current observation.

        Returns
        -------
        np.ndarray
            The selected action.
        """
        return self.action


class LinearSchedule(BaseActionSchedule):
    """
    A linear observation schedule. Moves observation from min to max throughout the
    experiment steps.

    Parameters
    ----------
    action_space : spaces.Box
        The action space of the schedule.
    observation_space : spaces.Box
        The observation space of the schedule.
    steps : int, optional
        The number of steps to take, by default 100.
        Will be set to length of episode.
    """

    def __init__(
        self,
        action_space: spaces.Box,
        observation_space: spaces.Box,
        steps: int = 1000,
    ):
        super().__init__(action_space, observation_space)
        self.steps = steps

    def _action_selection(self, obs: np.ndarray) -> np.ndarray:
        """
        Constant action, based on observation space.

        Parameters
        ----------
        obs : np.ndarray
            The current observation.

        Returns
        -------
        np.ndarray
            The selected action.
        """
        return self.action

    def _set_start(self, start_state):
        self.start = start_state
        obs_space_np = np.array([self.observation_space.low, self.observation_space.high])
        dist = (obs_space_np - self.start) ** 2
        row_indx = np.argmax(dist, axis=0)
        column_indx = np.arange(obs_space_np.shape[1])
        self.end = obs_space_np[row_indx, column_indx]
        action = (self.end - self.start) * 1 / (self.steps + 1)
        self.action = self.clip_to_action_space(action)
