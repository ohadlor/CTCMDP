import importlib
import re
import numpy as np

from gymnasium import spaces
from gymnasium.wrappers import RecordVideo


def get_param_bounds(gym_id: str, bound_dim_name: str = "THREE_DIM"):
    """
    Gets the parameter bounds dictionary from the gym id string and a dimension name.
    For example, for "rrls/robust-ant-v0" and "THREE_DIM", it returns the corresponding bounds dictionary.
    """
    # Extract the core environment name from the gym_id
    # e.g., "rrls/robust-ant-v0" -> "ant"
    # e.g., "rrls/robust-walker2d-v0" -> "walker2d"
    match = re.search(r"rrls/robust-(.*?)-v\d+", gym_id)
    if not match:
        raise ValueError(f"Invalid gym_id format: {gym_id}. Expected format 'rrls/robust-<env_name>-v0'.")

    core_env_name_kebab = match.group(1)

    # Module name is likely snake_case
    core_env_name_snake = core_env_name_kebab.replace("-", "_")
    module_name = f"rrls.envs.{core_env_name_snake}"

    # Convert core_env_name to CamelCase for the class name.
    # e.g., "walker2d" -> "Walker2d"
    # e.g., "half-cheetah" -> "HalfCheetah"
    class_name_base = "".join(x.capitalize() for x in core_env_name_kebab.split("-"))
    class_name = f"{class_name_base}ParamsBound"

    # Dynamically import the class
    try:
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


def bounds_to_space(param_bounds: dict[str, tuple[float, float]], radius: float) -> tuple[spaces.Box, spaces.Box, list]:
    params = list(param_bounds.keys())

    # Observation space
    low_obs = np.array([param_bounds[param][0] for param in params], dtype=np.float32)
    high_obs = np.array([param_bounds[param][1] for param in params], dtype=np.float32)
    obs_space = spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)

    # Action space
    num_params = len(params)
    low_act = np.full((num_params,), -radius, dtype=np.float32)
    high_act = np.full((num_params,), radius, dtype=np.float32)
    action_space = spaces.Box(low=low_act, high=high_act, dtype=np.float32)

    return obs_space, action_space, params


def remove_record_video_wrapper(env):
    """
    Recursively removes all instances of the RecordVideo wrapper from the environment chain.
    """
    if isinstance(env, RecordVideo):
        # If the current env is RecordVideo, unwrap it and continue searching
        return remove_record_video_wrapper(env.env)
    elif hasattr(env, "env"):
        # Otherwise, continue searching down the wrapper stack
        env.env = remove_record_video_wrapper(env.env)
    return env
