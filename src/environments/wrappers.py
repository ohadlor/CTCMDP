from typing import Optional, Any
import copy

from gymnasium import Wrapper, Env
from gymnasium import spaces
import numpy as np

from .env_utils import (
    bounds_to_space,
    get_param_defaults,
    find_attribute_in_stack,
    PARAMETER_SPACE,
)


TARGET_ATTRIBUTES = {
    "_mass": "body_mass",
    "_friction": "geom_friction",
}


class RobustWrapper(Wrapper):
    """
    A Gymnasium wrapper that implements an interface for modifying environment physics parameters.

    It allows accessing and modifying parameters like body mass, friction, and joint damping.
    It supports the `env.reset(options=params)` interface to set parameters for an episode.
    """

    def __init__(self, env: Env, param_space: Optional[dict[str, Any]] = None, seed: Optional[int] = None):
        super().__init__(env)

        # Ensure we can access the MuJoCo model
        if not hasattr(env.unwrapped, "model") or not hasattr(env.unwrapped, "data"):
            raise TypeError("RobustWrapper only supports Gymnasium MuJoCo environments.")

        self.model = env.unwrapped.model
        self.data = env.unwrapped.data

        # Cache initial default parameters to allow resetting to baseline
        self.env_name = self.env.unwrapped.__class__.__name__.replace("Env", "")
        self.parameter_mapping = self._get_dynamic_parameter_mapping(self.env_name)
        # Set every reset for stability
        self._defaults = self.get_params(for_defaults=True)
        # Actual defaults used
        self.param_defaults = get_param_defaults(self.env_name)

        # For domain randomization
        self.param_space = param_space
        if self.param_space:
            self.np_random = np.random.default_rng(seed)

    def _get_name_to_index_map(self, mujoco_attribute: str) -> list[str]:
        """
        Gets a dictionary mapping names to indices for a given MuJoCo attribute.

        Parameters
        ----------
        mujoco_attribute : str
            The MuJoCo attribute to get the names from (e.g., "body", "geom").

        Returns
        -------
        dict[str, int]
            A dictionary mapping names to indices.
        """
        if mujoco_attribute == "body":
            model_att = self.model.body
            n_obj = self.model.nbody
        elif mujoco_attribute == "geom":
            model_att = self.model.geom
            n_obj = self.model.ngeom
        else:
            raise ValueError(f"Unknown MuJoCo attribute: {mujoco_attribute}")

        names = []
        for i in range(n_obj):
            name = model_att(i).name
            names.append(name)
        return names

    def _get_dynamic_parameter_mapping(self, env_name: str) -> dict[str, tuple[int, str]]:
        """
        Dynamically creates a parameter mapping for the given environment.

        Parameters
        ----------
        env_name : str
            The name of the environment.

        Returns
        -------
        dict[str, tuple[int, str]]
            A dictionary mapping parameter names to a tuple of (index, attribute_name).
        """
        mapping = {}
        param_names = PARAMETER_SPACE.get(env_name, {}).keys()

        body_name_map = self._get_name_to_index_map("body")
        geom_name_map = self._get_name_to_index_map("geom")

        for param_name in param_names:
            for suffix, attr_name in TARGET_ATTRIBUTES.items():
                if param_name.endswith(suffix):
                    name = param_name.removesuffix(suffix)
                    if attr_name == "body_mass":
                        if name in body_name_map:
                            mapping[param_name] = (body_name_map.index(name), attr_name)
                    elif attr_name == "geom_friction":
                        if name in geom_name_map:
                            # Set all frictions of a part to value
                            mapping[param_name] = ((geom_name_map.index(name), slice(None)), attr_name)
                    break
            if param_name == "world_friction":
                # Change the sliding friction for all parts
                mapping[param_name] = ((slice(None), 0), "geom_friction")
            if param_name not in mapping:
                raise ValueError(f"Unknown parameter name: {param_name}")

        return mapping

    def get_params(self, for_defaults: bool = False) -> dict[str, float]:
        """
        Returns a dictionary of the current modifiable parameters.
        Returns flat keys like 'torso_mass', 'ankle_damping'.
        """
        params = {}
        for param_name, (param_index, param_attr) in self.parameter_mapping.items():
            params[param_name] = getattr(self.model, param_attr)[param_index]
        if "world_friction" in params and not for_defaults:
            param = set(params["world_friction"])
            assert len(param) == 1
            params["world_friction"] = param.pop()
        return params

    def set_params(self, params: dict[str, Any]):
        """
        Modifies the environment parameters based on the provided dictionary.

        Args:
            params: Dict where keys are parameter names (e.g., 'torso_mass')
                    and values are the new values (scalars or arrays).
        """
        # params = params.copy()
        # if "world_friction" in params:
        #     value = params.pop("world_friction")
        #     self.model.geom_friction[:, 0] = value

        for param_name, param_value in params.items():
            if param_name in self.parameter_mapping:
                param_index, param_attr = self.parameter_mapping[param_name]
                if param_attr in ["geom_friction", "body_mass"]:
                    getattr(self.model, param_attr)[param_index] = param_value
                else:
                    raise ValueError(f"Unknown parameter attribute: {param_attr}")

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None):
        """
        Gymnasium reset.

        Args:
            options: If provided, can contain a 'params' key or be the params dict itself
                     to set specific physics parameters for this episode.
        """
        # 1. Reset the underlying environment for stability
        self.set_params(self._defaults)
        super().reset(seed=seed, options=options)

        # Domain randomization
        if self.param_space:
            if seed is not None:
                self.np_random = np.random.default_rng(seed)

            new_params = {}
            for param, bounds in self.param_space.items():
                if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
                    low, high = bounds
                    val = self.np_random.uniform(low, high)
                    new_params[param] = val

            if options is None:
                options = {}
            options["params"] = new_params

        # 3. Apply new parameters if provided in options
        if options:
            # Check if options is the dict or contains 'params'
            # RRLS typically passed the params dict directly or inside a wrapper logic
            target_params = options.get("params", options)
            if isinstance(target_params, dict):
                # Filter options to only apply from default params
                valid_updates = {k: v for k, v in target_params.items() if k in self._defaults}
                if valid_updates:
                    self.set_params(valid_updates)
        else:
            self.set_params(self.param_defaults)

        obs, info = self.env.reset(seed=seed, options=options)
        info.update(self.get_params())
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info.update(self.get_params())
        return obs, reward, terminated, truncated, info


class TCRMDP(Wrapper):
    """
    A wrapper for a Time-Constrained Reinforcement Learning environment.

    This wrapper extends a ModifiedParamsEnv to include a hidden state and action space,
    creating a Time-Constrained Markov Decision Process (TCRMDP).

    Parameters
    ----------
    env : Env
        The environment to wrap.
    params_bound : dict[str, tuple[float, float]]
        A dictionary mapping parameter names to their bounds.
    radius : Optional[float], optional
        The radius of the hidden action space, by default None.
    """

    def __init__(
        self,
        env: Env,
        params_bound: dict[str, tuple[float, float]],
        radius: Optional[float] = None,
        params_bound_shrink_factor: float = 0.0,
    ):
        super().__init__(env)
        hidden_obs_space, hidden_action_space, self.hidden_params = bounds_to_space(
            params_bound, radius=radius, shrink_factor=params_bound_shrink_factor
        )

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
        params = self.get_params()
        return np.array([params[key] for key in self.hidden_params])

    @hidden_state.setter
    def hidden_state(self, hidden_state: np.ndarray):
        self.set_params(dict(zip(self.hidden_params, hidden_state)))

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
        self.hidden_state = self._hidden_step(self.hidden_state, hidden_action)
        obs, reward, terminated, truncated, info = self.env.step(original_action)

        full_obs = {"observed": obs, "hidden": self.hidden_state}
        info["previous_hidden"] = previous_hidden
        info["hidden"] = self.hidden_state
        self.last_observation = full_obs

        if terminated or truncated:
            info["terminal_hidden_state"] = self.hidden_state.copy()

        return full_obs, reward, terminated, truncated, info

    def _hidden_step(self, hidden_state: np.ndarray, hidden_action: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Performs a step in the hidden state space.

        Parameters
        ----------
        hidden_state : np.ndarray
            The current hidden state.
        hidden_action : Optional[np.ndarray], optional
            The action to take in the hidden state space, by default None.

        Returns
        -------
        np.ndarray
            The new hidden state.
        """
        if hidden_action is None:
            return hidden_state
        hidden_state = np.clip(
            hidden_state + hidden_action,
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


class BernoulliTruncation(Wrapper):
    def __init__(self, env, p: float = 1e-3, seed: Optional[int] = None):
        super().__init__(env)
        self.rng = np.random.default_rng(seed)
        self.p = p

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        truncated = bool(self.rng.binomial(1, self.p))
        return obs, reward, terminated, truncated, info
