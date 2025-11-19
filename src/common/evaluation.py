from typing import Any, Callable, Optional

from gymnasium import Env
import numpy as np

from src.agents.continual_algorithm import ContinualLearningAlgorithm
from src.schedules.base_schedule import BaseSchedule
from src.environments import check_for_wrapper
from src.environments.wrappers import TCRMDP


def evaluate_policy(
    model,
    env: Env,
    n_eval_episodes: int = 10,
    render: bool = False,
    callback: Optional[Callable[[dict[str, Any], dict[str, Any]], None]] = None,
    return_episode_rewards: bool = False,
    average_reward: bool = False,
    seed: Optional[int] = None,
) -> np.ndarray:
    all_rewards = []

    if check_for_wrapper(env, TCRMDP):
        env.reset()
        env = env.copy_to_stationary_env()
        hidden_state = env.hidden_state

    for _ in range(n_eval_episodes):
        episode_rewards = []
        done = False
        observations, _ = env.reset(seed=seed)
        current_step = 0

        while not done:
            current_step += 1
            if getattr(model, "oracle_actor", False):
                observations = np.concatenate([observations, hidden_state])
            actions = model.predict(observations)
            next_observations, reward, terminated, truncated, _ = env.step(actions)
            episode_rewards.append(reward)
            observations = next_observations
            done = terminated or truncated

            if callback is not None:
                callback(locals(), globals())

            if render:
                env.render()
        all_rewards.append(episode_rewards)

    max_len = max(len(ep) for ep in all_rewards) if all_rewards else 0
    padded_rewards = np.nan * np.ones((len(all_rewards), max_len))
    for i, ep in enumerate(all_rewards):
        padded_rewards[i, : len(ep)] = ep

    if average_reward:
        avg_reward = np.nanmean(padded_rewards, axis=1)
        std_reward = np.nanstd(avg_reward)
        return avg_reward.mean(), std_reward
    elif return_episode_rewards:
        return padded_rewards

    total_reward = np.nansum(padded_rewards, axis=1)
    return total_reward.mean(), total_reward.std()


def evaluate_policy_hidden_state(
    env,
    model,
    n_eval_episodes: int = 10,
    continual_args: dict = {},
    max_steps: int = 100_000,
    adversary_policy: Optional[BaseSchedule] = None,
    render: bool = False,
    callback: Optional[Callable[[dict[str, Any], dict[str, Any]], None]] = None,
    return_episode_rewards: bool = False,
    average_reward: bool = False,
    seed: Optional[int] = None,
):
    all_rewards = []

    if n_eval_episodes == 0:
        return np.array([])

    for _ in range(n_eval_episodes):
        episode_rewards = []
        done = False
        observations, _ = env.reset(seed=seed)
        if isinstance(model, ContinualLearningAlgorithm):
            continual_args_dict = continual_args.asdict()
            continual_args_dict["stationary_env"] = env.copy_to_stationary_env()
        else:
            continual_args_dict = {}
        sample = None
        current_step = 0
        while not done:
            current_step += 1
            actions = model.predict(observations, sample=sample, **continual_args_dict)
            if adversary_policy is not None:
                obs, hidden_state = observations.values()
                hidden_actions = adversary_policy.select_action(obs, hidden_state, deterministic=True)
                actions = {"observed": actions, "hidden": hidden_actions}

            next_observations, reward, terminated, truncated, _ = env.step(actions)
            if isinstance(model, ContinualLearningAlgorithm):
                continual_args_dict["stationary_env"] = env.copy_to_stationary_env()
            episode_rewards.append(reward)
            done = terminated or truncated
            sample = {
                "obs": observations,
                "next_obs": next_observations,
                "action": actions,
                "reward": reward,
                "done": done,
                "hidden_state": hidden_state,
                "next_hidden_state": next_hidden_state,
            }
            observations = next_observations

            if callback is not None:
                callback(locals(), globals())

            if render:
                env.render()
        all_rewards.append(episode_rewards)

    max_len = max(len(ep) for ep in all_rewards) if all_rewards else 0
    padded_rewards = np.nan * np.ones((len(all_rewards), max_len))
    for i, ep in enumerate(all_rewards):
        padded_rewards[i, : len(ep)] = ep

    if average_reward:
        avg_reward = padded_rewards.mean(axis=1)
        std_reward = avg_reward.std()
        return avg_reward.mean(), std_reward
    elif return_episode_rewards:
        return padded_rewards

    total_reward = padded_rewards.sum(axis=1)
    return total_reward.mean(), total_reward.std()
