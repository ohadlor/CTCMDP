import numpy as np


def tracking_reward(current_state: float, target_state: float, k: float = 1.0, alpha: float = 10.0) -> float:
    """Generic velocity tracking for Ant, HalfCheetah, Walker, Hopper."""
    error = (current_state - target_state) ** 2
    return alpha * np.exp(-k * error)


ENV_REWARDS = {
    "Ant": {
        "range": (0.0, 5.0),
        "fn_params": ([0.5, 10.0], 2.5),
        "obs_index": 13,
    },
    "HalfCheetah": {
        "range": ([0.0, 12.0], 6.0),
        "fn_params": (1.0, 10.0),
        "obs_index": 8,
    },
    "Walker2d": {
        "range": ([0.0, 6.0], 3.0),
        "fn_params": (1.0, 10.0),
        "obs_index": 8,
    },
    "Hopper": {
        "range": ([0.8, 1.5], 1.25),
        "fn_params": (4.0, 15.0),
        "obs_index": 0,
    },
    "HumanoidStandup": {
        "range": ([0.6, 1.4], 1.3),
        "fn_params": (5.0, 20.0),
        "obs_index": 0,
    },
}


def get_parameterized_reward_fn(env_name: str):
    reward_info = ENV_REWARDS.get(env_name)
    if reward_info is None:
        raise ValueError(f"No reward function found for environment: {env_name}")

    k, alpha = reward_info["fn_params"]

    def parameterized_reward_fn(current_state, target_state):
        return tracking_reward(current_state, target_state, k=k, alpha=alpha)

    return parameterized_reward_fn
