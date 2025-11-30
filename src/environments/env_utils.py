import importlib
import numpy as np

import gymnasium as gym
from gymnasium import spaces

# Mapping of rrls registered env name to (rrls.env_module, param_class_name)
env_mapping = {
    "ant": ("ant", "Ant"),
    "halfcheetah": ("half_cheetah", "HalfCheetah"),
    "hopper": ("hopper", "Hopper"),
    "walker2d": ("walker", "Walker2d"),
    "humanoidstandup": ("humanoid", "HumanoidStandup"),
}


def get_param_bounds(env_name: str, bound_dim_name: str = "THREE_DIM"):
    """
    Gets the parameter bounds dictionary from the gym id string and a dimension name.
    For example, for "rrls/robust-ant-v0" and "THREE_DIM", it returns the corresponding bounds dictionary.

    Parameters
    ----------
    env_name : str
        The name of the environment.
    bound_dim_name : str, optional
        The name of the bound dimension, by default "THREE_DIM".

    Returns
    -------
    dict
        The parameter bounds dictionary.
    """

    module_name, class_name_base = env_mapping[env_name]
    class_name = f"{class_name_base}ParamsBound"

    # Dynamically import the class
    try:
        module_name = f"rrls.envs.{module_name}"
        module = importlib.import_module(module_name)
        param_class = getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(
            f"Could not import '{class_name}' from '{module_name}'. "
            "Please check if the rrls library is installed correctly and the naming convention is as expected."
            f"Error: {e}"
        )

    # Now, get the specific bounds from the enum class
    try:
        bounds_enum_member = getattr(param_class, bound_dim_name)
        return bounds_enum_member.value
    except AttributeError:
        raise AttributeError(
            f"'{bound_dim_name}' is not a valid dimension in {class_name}. "
            f"Available dimensions are: {[member.name for member in param_class]}"
        )


def bounds_to_space(
    param_bounds: dict[str, tuple[float, float]], radius: float, shrink_factor: float = 0.0
) -> tuple[spaces.Box, spaces.Box, list]:
    """
    Convert parameter bounds to observation and action spaces.

    Parameters
    ----------
    param_bounds : dict[str, tuple[float, float]]
        A dictionary mapping parameter names to their bounds.
    radius : float
        The radius of the action space.
    shrink_factor : float
        The amount to shrink the hidden observation space, between 0 and 1 (0 is no shrinking)

    Returns
    -------
    tuple[spaces.Box, spaces.Box, list]
        A tuple containing the observation space, the action space, and the list of parameter names.
    """
    assert shrink_factor >= 0 and shrink_factor <= 1, "Shrink factor is out of range"
    assert radius >= 0, "Radius is negative"
    params = list(param_bounds.keys())

    # Observation space
    low_obs = np.array([param_bounds[param][0] for param in params], dtype=np.float32)
    high_obs = np.array([param_bounds[param][1] for param in params], dtype=np.float32)
    middle_obs = (high_obs + low_obs) / 2
    range_obs = (high_obs - low_obs) * (1 - shrink_factor) / 2
    low_obs = middle_obs - range_obs
    high_obs = middle_obs + range_obs

    obs_space = spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)

    # Action space
    num_params = len(params)
    low_act = np.full((num_params,), -radius, dtype=np.float32)
    high_act = np.full((num_params,), radius, dtype=np.float32)
    action_space = spaces.Box(low=low_act, high=high_act, dtype=np.float32)

    return obs_space, action_space, params


def find_attribute_in_stack(start_env, attribute_name: str):
    """
    Searches down the wrapper stack starting from start_env for the attribute_name.

    Parameters
    ----------
    start_env : gym.Env
        The environment to start the search from.
    attribute_name : str
        The name of the attribute to search for.

    Returns
    -------
    any
        The value of the attribute if found.

    Raises
    ------
    AttributeError
        If the attribute is not found in the wrapper stack.
    """
    current_env = start_env

    # Loop as long as the current object is a Gymnasium object
    while isinstance(current_env, gym.Wrapper) or isinstance(current_env, gym.Env):

        if hasattr(current_env, attribute_name):
            return getattr(current_env, attribute_name)

        # Move one level deeper
        if hasattr(current_env, "env"):
            current_env = current_env.env
        else:
            break

    raise AttributeError(f"Attribute or method '{attribute_name}' not found anywhere below the starting environment.")


def check_for_wrapper(env, wrapper_class):
    """
    Checks if a specific wrapper class is applied to a Gymnasium environment.

    Parameters
    ----------
    env : gymnasium.Env
        The Gymnasium environment instance.
    wrapper_class : type
        The class of the wrapper to check for (e.g., gym.wrappers.ClipAction).

    Returns
    -------
    bool
        True if the wrapper is found, False otherwise.
    """
    current_env = env
    while True:
        if isinstance(current_env, wrapper_class):
            return True
        # Check if the current environment has a 'env' attribute (meaning it's a Wrapper)
        if hasattr(current_env, "env"):
            current_env = current_env.env
        else:
            # Reached the base environment, wrapper not found
            return False


def name_to_env_id(name: str, is_rrls: bool = True) -> str:
    """
    Convert a name to a Gymnasium environment ID.

    Parameters
    ----------
    name : str
        The name of the environment.
    is_rrls : bool, optional
        Whether the environment is an rrls environment, by default True.

    Returns
    -------
    str
        The Gymnasium environment ID.
    """
    vanilla_map = {
        "ant": "Ant-v5",
        "halfcheetah": "HalfCheetah-v5",
        "hopper": "Hopper-v5",
        "walker2d": "Walker2d-v5",
        "humanoidstandup": "HumanoidStandup-v5",
    }
    robust_map = {
        "ant": "rrls/robust-ant-v0",
        "halfcheetah": "rrls/robust-halfcheetah-v0",
        "hopper": "rrls/robust-hopper-v0",
        "walker2d": "rrls/robust-walker-v0",
        "humanoidstandup": "rrls/robust-humanoidstandup-v0",
    }
    if is_rrls:
        return robust_map[name]
    else:
        return vanilla_map[name]


def find_robust_env(env: gym.Env) -> gym.Env:
    current_env = env
    while hasattr(current_env, "env"):
        if "Robust" in current_env.__class__.__name__:
            return current_env
        current_env = current_env.env
