from typing import Optional, Any

import gymnasium as gym
from gymnasium import Wrapper, Env
from gymnasium import spaces
import numpy as np

from .env_utils import (
    bounds_to_space,
    get_param_defaults,
    find_attribute_in_stack,
    find_wrapper_in_stack,
    PARAMETER_SPACE,
)
from .rewards import get_parameterized_reward_fn, ENV_REWARDS

TARGET_ATTRIBUTES = {
    "_mass": "body_mass",
    "_friction": "geom_friction",
}


class RobustWrapper(Wrapper):
    """
    A Gymnasium wrapper that implements an interface for modifying environment physics parameters
    and optionally managing a reward target state for reward augmentation.
    """

    def __init__(
        self,
        env: Env,
        domain_space: Optional[dict[str, Any]] = None,
        seed: Optional[int] = None,
        augment_reward: bool = False,
    ):
        super().__init__(env)

        if not hasattr(env.unwrapped, "model") or not hasattr(env.unwrapped, "data"):
            raise TypeError("RobustWrapper only supports Gymnasium MuJoCo environments.")
        self.augment_reward = augment_reward

        self.model = env.unwrapped.model
        self.data = env.unwrapped.data

        self.env_name = self.env.unwrapped.__class__.__name__.replace("Env", "")
        self.parameter_mapping = self._get_dynamic_parameter_mapping(self.env_name)
        self._defaults = self.get_params(for_defaults=True)
        self.param_defaults = get_param_defaults(self.env_name, self.augment_reward)

        self.domain_space = domain_space
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # --- Reward-related attributes ---

        if self.augment_reward:
            reward_info = ENV_REWARDS.get(self.env_name)
            if not reward_info:
                raise ValueError(f"No reward info found for {self.env_name} to augment reward.")

            self.reward_fn = get_parameterized_reward_fn(self.env_name)
            self.obs_index = reward_info["obs_index"]
            self.target_name = "reward_target"
            self.reward_target = self.param_defaults[self.target_name]
        else:
            self.reward_fn = None
            self.obs_index = None
            self.target_name = None
            self.reward_target = None

    def _get_name_to_index_map(self, mujoco_attribute: str) -> list[str]:
        if mujoco_attribute == "body":
            model_att, n_obj = self.model.body, self.model.nbody
        elif mujoco_attribute == "geom":
            model_att, n_obj = self.model.geom, self.model.ngeom
        else:
            raise ValueError(f"Unknown MuJoCo attribute: {mujoco_attribute}")
        return [model_att(i).name for i in range(n_obj)]

    def _get_dynamic_parameter_mapping(self, env_name: str) -> dict[str, tuple[int, str]]:
        mapping = {}
        param_names = PARAMETER_SPACE.get(env_name, {}).keys()
        body_name_map = self._get_name_to_index_map("body")
        geom_name_map = self._get_name_to_index_map("geom")
        for param_name in param_names:
            found = False
            for suffix, attr_name in TARGET_ATTRIBUTES.items():
                if param_name.endswith(suffix):
                    name = param_name.removesuffix(suffix)
                    if attr_name == "body_mass" and name in body_name_map:
                        mapping[param_name] = (body_name_map.index(name), attr_name)
                        found = True
                    elif attr_name == "geom_friction" and name in geom_name_map:
                        mapping[param_name] = ((geom_name_map.index(name), slice(None)), attr_name)
                        found = True
                    break
            if not found:
                if param_name == "world_friction":
                    mapping[param_name] = ((slice(None), 0), "geom_friction")
                else:
                    raise ValueError(f"Unknown parameter name: {param_name}")
        return mapping

    def get_params(self, for_defaults: bool = False) -> dict[str, float]:
        params = {name: getattr(self.model, attr)[idx] for name, (idx, attr) in self.parameter_mapping.items()}
        if "world_friction" in params and not for_defaults:
            param_set = set(params["world_friction"])
            if len(param_set) == 1:
                params["world_friction"] = param_set.pop()

        if self.augment_reward and not for_defaults:
            params[self.target_name] = self.reward_target
        return params

    def set_params(self, params: dict[str, Any]):
        if self.augment_reward and self.target_name in params:
            self.reward_target = params[self.target_name]

        for param_name, param_value in params.items():
            if param_name in self.parameter_mapping:
                param_index, param_attr = self.parameter_mapping[param_name]
                getattr(self.model, param_attr)[param_index] = param_value

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None):
        self.set_params(self._defaults)
        super().reset(seed=seed, options=options)

        if self.domain_space:
            new_params = {
                p: self.rng.uniform(low, high)
                for p, (low, high) in self.domain_space.items()
                if isinstance(self.domain_space.get(p), (list, tuple)) and len(self.domain_space.get(p)) == 2
            }
            if new_params:
                options = options or {}
                options["params"] = new_params

        if options and isinstance(options.get("params", options), dict):
            valid_updates = {k: v for k, v in options.get("params", options).items()}
            if valid_updates:
                self.set_params(valid_updates)
        else:
            self.set_params(self.param_defaults)

        obs, info = self.env.reset(seed=seed, options=options)
        info.update(self.get_params())
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.augment_reward:
            current_state = obs[self.obs_index]
            new_reward = self.reward_fn(current_state, self.reward_target)
            reward += new_reward
        info.update(self.get_params())
        return obs, reward, terminated, truncated, info

    def get_mujoco_state(self) -> dict:
        mj_data = self.unwrapped.data
        return {"qpos": mj_data.qpos.copy(), "qvel": mj_data.qvel.copy()}

    def set_mujoco_state(self, mujoco_state: dict) -> None:
        mj_env = self.unwrapped
        mj_env.set_state(mujoco_state["qpos"], mujoco_state["qvel"])

    def copy_env_state(self, env: Env) -> None:
        full_state = env.get_wrapper_attr("get_mujoco_state")()
        self.set_mujoco_state(full_state)

    def copy_env_hidden_state(self, env: Env) -> None:
        env_params = env.get_wrapper_attr("get_params")()
        self.set_params(env_params)

    def copy_env(self, env: Env) -> None:
        self.reset()
        self.copy_env_state(env)
        self.copy_env_hidden_state(env)

    def make_env(self) -> Env:
        env_id = self.unwrapped.spec.id
        env = gym.make(env_id, max_episode_steps=-1)
        env = RobustWrapper(env)
        env.reset(seed=self.seed)
        return env


class TCRMDP(Wrapper):
    """
    A wrapper for a Time-Constrained Reinforcement Learning environment.
    """

    def __init__(
        self,
        env: Env,
        params_bound: dict[str, tuple[float, float]],
        radius: Optional[float] = None,
        params_bound_shrink_factor: float = 0.0,
    ):
        super().__init__(env)
        self._robust_wrapper = find_wrapper_in_stack(self.env, RobustWrapper)
        if not self._robust_wrapper:
            raise ValueError("TCRMDP requires a RobustWrapper in the stack.")

        hidden_obs_space, hidden_action_space, self.hidden_params = bounds_to_space(
            params_bound, radius=radius, shrink_factor=params_bound_shrink_factor
        )
        self.observation_space = spaces.Dict({"observed": self.observation_space, "hidden": hidden_obs_space})
        self.action_space = spaces.Dict({"observed": self.action_space, "hidden": hidden_action_space})
        self._get_state = find_attribute_in_stack(self, "get_mujoco_state")
        self._set_state = find_attribute_in_stack(self, "set_mujoco_state")

    @property
    def state(self) -> dict:
        mujoco_state = self._get_state()
        return {"observed": mujoco_state, "hidden": self.hidden_state}

    @state.setter
    def state(self, full_state: dict):
        self._set_state(full_state["observed"])
        self.hidden_state = full_state["hidden"]

    @property
    def hidden_state(self) -> np.ndarray:
        params = self._robust_wrapper.get_params()
        return np.array([params[key] for key in self.hidden_params], dtype=np.float32)

    @hidden_state.setter
    def hidden_state(self, hidden_state: np.ndarray):
        self._robust_wrapper.set_params(dict(zip(self.hidden_params, hidden_state)))

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        hidden_obs = self.hidden_state
        full_obs = {"observed": obs, "hidden": hidden_obs}
        info["hidden"] = hidden_obs
        self.last_observation = full_obs
        return full_obs, info

    def step(self, action: spaces.Dict) -> tuple[spaces.Dict, float, bool, bool, dict]:
        original_action = action["observed"]
        hidden_action = action["hidden"]
        previous_hidden = self.hidden_state
        self.hidden_state = self._hidden_step(self.hidden_state, hidden_action)
        obs, reward, terminated, truncated, info = self.env.step(original_action)
        full_obs = {"observed": obs, "hidden": self.hidden_state}
        info.update({"previous_hidden": previous_hidden, "hidden": self.hidden_state})
        if terminated or truncated:
            info["terminal_hidden_state"] = self.hidden_state.copy()
        return full_obs, reward, terminated, truncated, info

    def _hidden_step(self, hidden_state: np.ndarray, hidden_action: Optional[np.ndarray] = None) -> np.ndarray:
        if hidden_action is None:
            return hidden_state
        return np.clip(
            hidden_state + hidden_action, self.observation_space["hidden"].low, self.observation_space["hidden"].high
        )


class SplitActionObservationSpace(Wrapper):
    """
    A wrapper that splits the action and observation spaces into observed and hidden components.
    """

    def __init__(self, env: TCRMDP):
        super().__init__(env)
        self.observation_space = env.observation_space["observed"]
        self.action_space = env.action_space["observed"]
        self.hidden_observation_space = env.observation_space["hidden"]
        self.hidden_action_space = env.action_space["hidden"]

    def step(self, action: np.ndarray, hidden_action: Optional[np.ndarray] = None):
        hidden_action = hidden_action if hidden_action is not None else np.zeros(self.hidden_action_space.shape)
        dict_action = {"observed": action, "hidden": hidden_action}
        obs, reward, terminated, truncated, info = self.env.step(dict_action)
        info["hidden"] = obs["hidden"]
        return obs["observed"], obs["hidden"], reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        info["hidden"] = obs["hidden"]
        return obs["observed"], obs["hidden"], info

    @property
    def hidden_state(self) -> np.ndarray:
        return find_attribute_in_stack(self, "hidden_state")


class BernoulliTruncation(Wrapper):
    def __init__(self, env, p: float = 1e-3, seed: Optional[int] = None):
        super().__init__(env)
        self.rng = np.random.default_rng(seed)
        self.p = p

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.rng.binomial(1, self.p):
            truncated = True
        return obs, reward, terminated, truncated, info
