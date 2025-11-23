from typing import Optional
import copy

from gymnasium import Wrapper, Env
from gymnasium import spaces
import numpy as np
from rrls._interface import ModifiedParamsEnv

from .env_utils import bounds_to_space, remove_record_video_wrapper, find_attribute_in_stack, find_robust_env


class TCRMDP(Wrapper):
    """
    A wrapper for a Time-Constrained Reinforcement Learning environment.

    This wrapper extends a ModifiedParamsEnv to include a hidden state and action space,
    creating a Time-Constrained Markov Decision Process (TCRMDP).

    Parameters
    ----------
    env : ModifiedParamsEnv
        The environment to wrap.
    params_bound : dict[str, tuple[float, float]]
        A dictionary mapping parameter names to their bounds.
    radius : Optional[float], optional
        The radius of the hidden action space, by default None.
    """

    def __init__(
        self, env: ModifiedParamsEnv, params_bound: dict[str, tuple[float, float]], radius: Optional[float] = None
    ):
        super().__init__(env)
        hidden_obs_space, hidden_action_space, self.hidden_params = bounds_to_space(params_bound, radius=radius)

        original_obs_space = self.observation_space
        original_action_space = self.action_space
        self.observation_space = spaces.Dict(
            {
                "observed": original_obs_space,
                "hidden": hidden_obs_space,
            }
        )

        self.action_space = spaces.Dict(
            {
                "observed": original_action_space,
                "hidden": hidden_action_space,
            }
        )
        self.set_params = find_attribute_in_stack(self, "set_params")
        self.get_params = find_attribute_in_stack(self, "get_params")

    @property
    def state(self) -> spaces.Dict:
        """
        The state of the environment, including the observed and hidden states.
        """
        return {"observed": self.env.unwrapped.state, "hidden": self.hidden_state}

    @state.setter
    def state(self, full_state: spaces.Dict):
        self.env.unwrapped.state = full_state["observed"]
        self.hidden_state = full_state["hidden"]

    @property
    def hidden_state(self) -> np.ndarray:
        """
        The hidden state of the environment.
        """
        robust_env = find_robust_env(self)
        params = robust_env.get_params()
        return np.array([params[key] for key in self.hidden_params])

    @hidden_state.setter
    def hidden_state(self, hidden_state: np.ndarray):
        robust_env = find_robust_env(self)
        robust_env.set_params(**dict(zip(self.hidden_params, hidden_state)))

    def reset(self, *, seed=None, options=None):
        """
        Resets the environment and the state of the added observation dimensions.

        Parameters
        ----------
        seed : int, optional
            The seed to use for the environment's random number generator, by default None.
        options : dict, optional
            A dictionary of options for the environment, by default None.

        Returns
        -------
        tuple[spaces.Dict, dict]
            A tuple containing the initial observation and a dictionary of additional information.
        """
        obs, info = self.env.reset(seed=seed, options=options)
        if options is not None:
            self.set_params(**options)
        else:
            self.set_params()
        hidden_obs = self.hidden_state

        full_obs = {"observed": obs, "hidden": hidden_obs}
        info["hidden"] = hidden_obs
        self.last_observation = full_obs

        return full_obs, info

    def step(self, action: spaces.Dict) -> tuple[spaces.Dict, float, bool, bool, dict]:
        """
        Performs a step in the environment with the custom logic.

        Parameters
        ----------
        action : spaces.Dict
            The action to take in the environment.

        Returns
        -------
        tuple[spaces.Dict, float, bool, bool, dict]
            A tuple containing the new observation, the reward, whether the episode has terminated,
            whether the episode has been truncated, and a dictionary of additional information.
        """
        original_action = action["observed"]
        hidden_action = action["hidden"]
        previous_hidden = self.hidden_state

        # Hidden state changes before the original state
        self.hidden_state = self._hidden_step(hidden_action)
        obs, reward, terminated, truncated, info = self.env.step(original_action)

        full_obs = {"observed": obs, "hidden": self.hidden_state}
        info["previous_hidden"] = previous_hidden
        info["hidden"] = self.hidden_state
        self.last_observation = full_obs

        if terminated or truncated:
            info["terminal_hidden_state"] = self.hidden_state.copy()

        return full_obs, reward, terminated, truncated, info

    def _hidden_step(self, hidden_action: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Performs a step in the hidden state space.

        Parameters
        ----------
        hidden_action : Optional[np.ndarray], optional
            The action to take in the hidden state space, by default None.

        Returns
        -------
        np.ndarray
            The new hidden state.
        """
        if hidden_action is None:
            return self.hidden_state
        hidden_state = np.clip(
            self.hidden_state + hidden_action,
            self.observation_space["hidden"].low,
            self.observation_space["hidden"].high,
        )
        return hidden_state

    def copy_to_stationary_env(self) -> Env:
        """
        Creates a deep copy of the environment, sets a new hidden observation,
        and returns a wrapped environment with simplified observation and action spaces.

        Returns
        -------
        Env
            A stationary version of the environment.
        """
        new_env = copy.deepcopy(self)

        # Remove the recording wrapper if there is one
        new_env = remove_record_video_wrapper(new_env)
        stationary_env = FrozenHiddenObservation(new_env)
        # Designed for mujoco envs
        position = self.unwrapped.data.qpos.flatten()
        velocity = self.unwrapped.data.qvel.flatten()
        stationary_env._base_state = (position, velocity)
        return stationary_env


class SplitActionObservationSpace(Wrapper):
    """
    A wrapper that splits the action and observation spaces into observed and hidden components.

    Parameters
    ----------
    env : TCRMDP
        The TCRMDP environment to wrap.
    """

    def __init__(self, env: TCRMDP):
        super().__init__(env)
        self.observation_space = env.observation_space["observed"]
        self.action_space = env.action_space["observed"]
        self.hidden_observation_space = env.observation_space["hidden"]
        self.hidden_action_space = env.action_space["hidden"]

        self.copy_to_stationary_env = env.copy_to_stationary_env

    def step(self, action: np.ndarray, hidden_action: np.ndarray):
        """
        Performs a step in the environment.

        Parameters
        ----------
        action : np.ndarray
            The action to take in the observed action space.
        hidden_action : np.ndarray
            The action to take in the hidden action space.

        Returns
        -------
        tuple
            A tuple containing the new observation, the new hidden state, the reward,
            whether the episode has terminated, whether the episode has been truncated,
            and a dictionary of additional information.
        """
        dict_action = {"observed": action, "hidden": hidden_action}
        obs, reward, terminated, truncated, info = self.env.step(dict_action)
        info["hidden"] = obs["hidden"]
        return *obs.values(), reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        """
        Resets the environment.

        Parameters
        ----------
        seed : int, optional
            The seed to use for the environment's random number generator, by default None.
        options : dict, optional
            A dictionary of options for the environment, by default None.

        Returns
        -------
        tuple
            A tuple containing the initial observation, the initial hidden state,
            and a dictionary of additional information.
        """
        obs, info = self.env.reset(seed=seed, options=options)
        info["hidden"] = obs["hidden"]
        return *obs.values(), info

    @property
    def hidden_state(self) -> np.ndarray:
        """
        The hidden state of the environment.
        """
        return find_attribute_in_stack(self, "hidden_state")


class FrozenHiddenObservation(Wrapper):
    """
    A wrapper that freezes the hidden state of the environment.

    Parameters
    ----------
    env : TCRMDP
        The TCRMDP environment to wrap.
    """

    def __init__(self, env: TCRMDP):
        super().__init__(env)
        self.observation_space = self.env.observation_space["observed"]
        self.action_space = self.env.action_space["observed"]

        self.hidden_state = self.env.hidden_state
        self._base_state = None

    def step(self, action: np.ndarray):
        """
        Performs a step in the environment.

        Parameters
        ----------
        action : np.ndarray
            The action to take in the observed action space.

        Returns
        -------
        tuple
            A tuple containing the new observation, the reward, whether the episode has terminated,
            whether the episode has been truncated, and a dictionary of additional information.
        """
        dict_action = {"observed": action, "hidden": None}
        obs, reward, terminated, truncated, info = self.env.step(dict_action)
        return obs["observed"], reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        """
        Resets the environment.

        Parameters
        ----------
        seed : int, optional
            The seed to use for the environment's random number generator, by default None.
        options : dict, optional
            A dictionary of options for the environment, by default None.

        Returns
        -------
        tuple
            A tuple containing the initial observation and a dictionary of additional information.
        """
        obs, info = self.env.reset(seed=seed, options=options)

        return obs["observed"], info
