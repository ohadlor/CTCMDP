import numpy as np
import gymnasium as gym
from gymnasium import spaces

# --- Robust Uncertainty Sets ---
# Standard uncertainty sets for perturbations (Mass, Friction, Damping)
# Values represent absolute ranges [min, max] based on standard MuJoCo XML defaults.
# These are aligned with common Robust RL benchmarks (e.g., RARL, M2TD3).

PARAMETER_SPACE = {
    "Ant": {
        "torso_mass": ([0.1, 3.0], 0.33),
        "front_left_leg_mass": ([0.01, 3.0], 0.04),
        "front_right_leg_mass": ([0.01, 3.0], 0.06),
    },
    "HalfCheetah": {
        "world_friction": ([0.1, 4.0], 0.4),
        "torso_mass": ([0.1, 7.0], 6.36),
        "bthigh_mass": ([0.1, 3.0], 1.53),
    },
    "Hopper": {
        "world_friction": ([0.1, 3.0], 1.00),
        "torso_mass": ([0.1, 3.0], 3.53),
        "thigh_mass": ([0.1, 4.0], 3.93),
    },
    "HumanoidStandup": {
        "torso_mass": ([0.1, 16.0], 8.32),
        "right_foot_mass": ([0.1, 5.0], 1.77),
        "left_thigh_mass": ([0.1, 8.0], 4.53),
    },
    "Walker2d": {
        "world_friction": ([0.1, 4.0], 0.7),
        "torso_mass": ([0.1, 5.0], 3.53),
        "thigh_mass": ([0.1, 6.0], 3.93),
    },
}


def get_param_bounds(env_id: str) -> dict[str, tuple[float, float]]:
    env_name = env_id.split("-")[0]
    full_set = PARAMETER_SPACE.get(env_name, {})
    uncertainty_set = {param: bounds[0] for param, bounds in full_set.items()}
    return uncertainty_set


def get_param_defaults(env_name: str) -> dict[str, float]:
    """
    Gets the parameter bounds dictionary from the gym
    id string and a dimension name.
    """
    full_set = PARAMETER_SPACE.get(env_name, {})
    defaults = {param: bounds[1] for param, bounds in full_set.items()}
    return defaults


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


def find_attribute_in_stack(start_env, attribute_name: str, default_value=None):
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

    return default_value


def check_for_wrapper(env: gym.Env, wrapper_class: type) -> bool:
    """
    Checks if a wrapper of a specific class exists in the environment stack.

    Parameters
    ----------
    env : gym.Env
        The environment to check.
    wrapper_class : type
        The wrapper class to look for.

    Returns
    -------
    bool
        True if the wrapper is found, False otherwise.
    """
    current_env = env
    while isinstance(current_env, gym.Wrapper):
        if isinstance(current_env, wrapper_class):
            return True
        current_env = current_env.env
    return False
