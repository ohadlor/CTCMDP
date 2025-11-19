from typing import Optional
import copy

from gymnasium import Wrapper, Env
from gymnasium import spaces
import numpy as np
from rrls._interface import ModifiedParamsEnv

from .env_utils import bounds_to_space, remove_record_video_wrapper, find_attribute_in_stack


class TCRMDP(Wrapper):

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
        return {"observed": self.env.unwrapped.state, "hidden": self.hidden_state}

    @state.setter
    def state(self, full_state: spaces.Dict):
        self.env.unwrapped.state = full_state["observed"]
        self.hidden_state = full_state["hidden"]

    @property
    def hidden_state(self) -> np.ndarray:
        params = self.get_params()
        return np.array([params[key] for key in self.hidden_params])

    @hidden_state.setter
    def hidden_state(self, hidden_state: np.ndarray):
        self.set_params(**dict(zip(self.hidden_params, hidden_state)))

    def reset(self, *, seed=None, options=None):
        """
        Resets the environment and the state of the added observation dimensions.
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
        """
        new_env = copy.deepcopy(self)

        # Remove the recording wrapper if there is one
        new_env = remove_record_video_wrapper(new_env)

        return FrozenHiddenObservation(new_env)


class SplitActionObservationSpace(Wrapper):
    def __init__(self, env: TCRMDP):
        super().__init__(env)
        self.observation_space = env.observation_space["observed"]
        self.action_space = env.action_space["observed"]
        self.hidden_observation_space = env.observation_space["hidden"]
        self.hidden_action_space = env.action_space["hidden"]

        self.copy_to_stationary_env = env.copy_to_stationary_env

    def step(self, action: np.ndarray, hidden_action: np.ndarray):
        dict_action = {"observed": action, "hidden": hidden_action}
        obs, reward, terminated, truncated, info = self.env.step(dict_action)
        info["hidden"] = obs["hidden"]
        return *obs.values(), reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        info["hidden"] = obs["hidden"]
        return *obs.values(), info

    @property
    def hidden_state(self) -> np.ndarray:
        return find_attribute_in_stack(self, "hidden_state")


class FrozenHiddenObservation(Wrapper):
    def __init__(self, env: TCRMDP):
        super().__init__(env)
        self.observation_space = self.env.observation_space["observed"]
        self.action_space = self.env.action_space["observed"]

        self.hidden_state = self.env.hidden_state

    def step(self, action: np.ndarray):
        dict_action = {"observed": action, "hidden": None}
        obs, reward, terminated, truncated, info = self.env.step(dict_action)
        return obs["observed"], reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        params = self.env.get_params()
        obs, info = self.env.reset(seed=seed, options=params)
        params_after = self.env.get_params()
        assert params == params_after
        return obs["observed"], info
