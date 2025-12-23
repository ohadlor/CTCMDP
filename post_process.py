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
            'sim_gamma': 0.99,
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
                params["sim_gamma"] = float(part.split("-", 1)[1])
            except (ValueError, IndexError):
                params["sim_gamma"] = part.split("-", 1)[1]
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


def process_experiment_dir(experiment_dir: str, reward_file_name: str = "reward.npy") -> Any:
    """
    Walks through an experiment directory, parses run directory names, and loads rewards.
    Groups runs with the same parameters (except for seed-specific ones) and stacks their rewards.

    Args:
        experiment_dir (str): The path to the experiment directory.
        reward_file_name (str): The name of the numpy file containing the reward data.
                                Defaults to "reward.npy".

    Returns:
        A list of tuples, where each tuple contains the parameters dictionary
        from the directory name and the loaded reward data. The reward data will be a numpy
        array with an extra dimension for different seeds if multiple runs are found.
    """
    grouped_results: dict[tuple, list[Any]] = {}
    param_map: dict[tuple, dict[str, Any]] = {}

    for root, dirs, files in os.walk(os.path.join("outputs", experiment_dir)):
        if reward_file_name in files or "evaluation.txt" in files:
            dir_name = os.path.basename(root)
            params = parse_dir_name(dir_name)

            if not params:
                continue

            # Create a unique key for the experiment based on its parameters,
            # ignoring seed-specific variations like timestamp and cool_name.
            key_params = params.copy()
            key_params.pop("timestamp", None)
            key_params.pop("cool_name", None)
            key_params.pop("full_path", None)

            experiment_key = tuple(sorted(key_params.items()))

            reward_data = None
            if reward_file_name in files:
                reward_path = os.path.join(root, reward_file_name)
                try:
                    reward_data = load_numpy_reward(reward_path)
                except Exception as e:
                    print(f"Warning: Error loading reward from {reward_path}: {e}")
                    continue
            elif "evaluation.txt" in files:
                evaluation_path = os.path.join(root, "evaluation.txt")
                try:
                    with open(evaluation_path, "r") as f:
                        first_line = f.readline().strip()
                        if "Time average reward:" in first_line:
                            reward_data = float(first_line.split(":")[1].strip())
                except Exception as e:
                    print(f"Warning: Error reading evaluation.txt from {evaluation_path}: {e}")
                    continue

            if reward_data is not None:
                if experiment_key not in grouped_results:
                    grouped_results[experiment_key] = []
                    param_map[experiment_key] = key_params
                grouped_results[experiment_key].append(reward_data)

    results = []
    for experiment_key, rewards_list in grouped_results.items():
        if not rewards_list:
            continue

        params = param_map[experiment_key]

        if all(isinstance(r, (int, float)) for r in rewards_list):
            stacked_rewards = np.array(rewards_list)
        else:
            try:
                stacked_rewards = np.stack(rewards_list, axis=0)
            except ValueError:
                print(f"Warning: Could not stack rewards for experiment {params}. Storing as list.")
                stacked_rewards = rewards_list

        results.append((params, stacked_rewards))

    return results


def process_experiment(experiment_dir: str):
    results = process_experiment_dir(experiment_dir)

    match experiment_dir:
        case "discount_radius":
            processed_results = {}
            # intermediate dict: {env: {radius: {sim_gamma: reward_array}}}
            for params, reward_data in results:
                env = params.get("env_name")
                radius = params.get("radius")
                sim_gamma = params.get("sim_gamma")
                if env and radius is not None and sim_gamma is not None:
                    if env not in processed_results:
                        processed_results[env] = {}
                    if radius not in processed_results[env]:
                        processed_results[env][radius] = {}
                    processed_results[env][radius][sim_gamma] = reward_data

            # final dict: {env: {radius: [discount_factors_list, avg_returns_list, std_returns_list]}}
            final_results = {}
            for env, radii in processed_results.items():
                final_results[env] = {}
                for radius, sim_gammas in radii.items():
                    # Sort by sim_gamma to have a consistent order for discount factors
                    sorted_gammas = sorted(sim_gammas.items())
                    discount_factors = [gamma for gamma, reward in sorted_gammas]
                    avg_returns = [np.mean(reward) for gamma, reward in sorted_gammas]
                    std_returns = [np.std(reward) for gamma, reward in sorted_gammas]
                    final_results[env][radius] = [discount_factors, avg_returns, std_returns]
            plot_and_save_results(final_results)

        case "schedules":
            pass
        case "shrink_factor":
            pass
        case _:
            raise ValueError(f"Unknown experiment directory: {experiment_dir}")


def plot_and_save_results(results: dict[str, dict[float, list[list[float]]]]):
    """
    Plots the best discount factor for each radius and saves the data.

    Args:
        results (dict): A dictionary with the processed results, with the format
                        {env: {radius: [discount_factors, avg_returns, std_returns]}}.
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    plot_dir = "plots"
    os.makedirs(plot_dir, exist_ok=True)

    plt.figure()
    for env, radii in results.items():
        radii_sorted = sorted(radii.keys())
        best_discounts = []
        for radius in radii_sorted:
            discounts, returns, _ = radii[radius]
            best_discount_idx = np.argmax(returns)
            best_discounts.append(discounts[best_discount_idx])
        plt.plot(radii_sorted, best_discounts, label=env, marker="o")

    plt.xlabel("Radius")
    plt.ylabel("Best Discount Factor")
    plt.legend()
    plt.title("Best Discount Factor vs. Radius for each Environment")
    plt.savefig(f"{plot_dir}/discount_radius_plot.png")
    plt.close()

    # Plot max return vs radius with error bars
    plt.figure()
    for env, radii in results.items():
        radii_sorted = sorted(radii.keys())
        max_returns = []
        max_returns_std = []
        for radius in radii_sorted:
            _, returns, stds = radii[radius]
            best_idx = np.argmax(returns)
            max_returns.append(returns[best_idx])
            max_returns_std.append(stds[best_idx])

        max_returns = np.array(max_returns)
        max_returns_std = np.array(max_returns_std)

        plt.plot(radii_sorted, max_returns, label=env, marker="o")
        plt.fill_between(radii_sorted, max_returns - max_returns_std, max_returns + max_returns_std, alpha=0.2)

    plt.xlabel("Radius")
    plt.ylabel("Average Return")
    plt.legend()
    plt.savefig(f"{plot_dir}/max_return_vs_radius_plot.png")
    plt.close()

    # Save the data
    output_dir = "processed_data"
    os.makedirs(output_dir, exist_ok=True)

    for env, radii in results.items():
        for radius, data in radii.items():
            df = pd.DataFrame({"discount_factor": data[0], "avg_return": data[1], "std_return": data[2]})
            filename = os.path.join(output_dir, f"{env}_radius_{radius}.csv")
            df.to_csv(filename, index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "experiment_dir",
        type=str,
        help="The directory of the experiment to process, relative to the 'outputs' folder.",
    )
    args = parser.parse_args()

    process_experiment(args.experiment_dir)
