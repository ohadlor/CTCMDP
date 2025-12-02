import os
from typing import Any

import numpy as np


def load_numpy_reward(path: str) -> np.ndarray:
    """Loads a numpy array reward from a given path.

    Args:
        path (str): The path to the npy file.

    Returns:
        np.ndarray: The reward data.
    """
    return np.load(path)


def parse_dir_name(dir_name: str) -> dict[str, Any]:
    """Parses a directory name and extracts the parameters.

    This function is the reverse of the `custom_dir_resolver` in
    `hydra_plugins/my_resolvers_plugin.py`. It takes a directory name
    string and returns a dictionary of the parameters.

    Args:
        dir_name (str): The directory name, e.g.,
            "20231027_exciting-newt_ant_continual_TD3_ContinualSchedule_radius-0.5_simdiscount-0.99_shrink".

    Returns:
        A dictionary of the parameters, e.g.,
        {
            'timestamp': '20231027',
            'cool_name': 'exciting-newt',
            'env_name': 'ant',
            'variant': 'continual',
            'model': 'TD3',
            'schedule': 'ContinualSchedule',
            'radius': 0.5,
            'sim_discount': 0.99,
            'shrink': True
        }
    """
    params: dict[str, Any] = {}
    parts = dir_name.split("_")

    if len(parts) < 3:
        return {}

    params["timestamp"] = parts.pop(0)
    params["cool_name"] = parts.pop(0)
    params["env_name"] = parts.pop(0)

    # Extract key-value parameters and flags first
    remaining_parts = []
    for part in parts:
        if part.startswith("radius-"):
            try:
                params["radius"] = float(part.split("-", 1)[1])
            except (ValueError, IndexError):
                # Handle cases where the value is not a valid float
                params["radius"] = part.split("-", 1)[1]
        elif part.startswith("simdiscount-"):
            try:
                params["sim_discount"] = float(part.split("-", 1)[1])
            except (ValueError, IndexError):
                params["sim_discount"] = part.split("-", 1)[1]
        elif part == "shrink":
            params["shrink"] = True
        else:
            remaining_parts.append(part)

    # The remaining parts should be variant, model, and schedule, in that order.
    # The model is always present. Variant and schedule are optional.
    if remaining_parts:
        if remaining_parts[-1].endswith("Schedule"):
            params["schedule"] = remaining_parts.pop(-1)

    # What's left can be [variant, model] or [model]
    if len(remaining_parts) == 2:
        params["variant"] = remaining_parts[0]
        params["model"] = remaining_parts[1]
    elif len(remaining_parts) == 1:
        # It could be a variant or a model. But the resolver logic suggests model is always there.
        # Let's assume it's the model.
        params["model"] = remaining_parts[0]

    return params


def process_experiment_dir(
    experiment_dir: str, reward_file_name: str = "reward.npy"
) -> list[tuple[dict[str, Any], np.ndarray]]:
    """
    Walks through an experiment directory, parses run directory names, and loads rewards.

    Args:
        experiment_dir (str): The path to the experiment directory.
        reward_file_name (str): The name of the numpy file containing the reward data.
                                Defaults to "reward.npy".

    Returns:
        A list of tuples, where each tuple contains the parameters dictionary
        from the directory name and the loaded reward data.
    """
    results = []
    for root, dirs, files in os.walk(os.path.join("outputs", experiment_dir)):
        if reward_file_name in files:
            dir_name = os.path.basename(root)
            params = parse_dir_name(dir_name)

            # Add the full path to the params for reference
            params["full_path"] = root

            reward_path = os.path.join(root, reward_file_name)
            try:
                reward_data = load_numpy_reward(reward_path)
                results.append((params, reward_data))
            except FileNotFoundError:
                # This shouldn't happen due to the `if reward_file_name in files` check, but good practice.
                print(f"Warning: Could not find reward file {reward_path}")
            except Exception as e:
                print(f"Warning: Error loading reward from {reward_path}: {e}")

    return results
