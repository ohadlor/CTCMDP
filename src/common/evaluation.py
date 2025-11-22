from typing import Any, Callable, Optional

import gymnasium as gym
import numpy as np

from src.agents.base_algorithm import BaseAlgorithm
from src.schedules import BaseActionSchedule
from src.environments import check_for_wrapper
from src.environments.wrappers import TCRMDP


def evaluate_policy(
    model,
    env: gym.Env,
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
    env: gym.Env,
    model: BaseAlgorithm,
    adversary_policy: Optional[BaseActionSchedule] = None,
    iterations: int = 10,
    render: bool = False,
    callback: Optional[Callable[[dict[str, Any], dict[str, Any]], None]] = None,
    return_episode_rewards: bool = False,
    seed: Optional[int] = None,
):
    all_rewards = []
    is_continual_learner = getattr(model, "is_continual_learner", False)
    continual_args = {}

    for _ in range(iterations):
        episode_rewards = []
        done = False
        observation, hidden_state, _ = env.reset(seed=seed)
        if is_continual_learner:
            continual_args["stationary_env"] = env.copy_to_stationary_env()
            continual_args["sample"] = None

        current_step = 0
        while not done:
            current_step += 1
            action = model.predict(observation, **continual_args)
            if adversary_policy is not None:
                hidden_action = adversary_policy.select_action(observation, hidden_state, deterministic=True)

            next_observation, next_hidden_state, reward, terminated, truncated, _ = env.step(action, hidden_action)
            episode_rewards.append(reward)
            done = terminated or truncated
            sample = {
                "obs": observation,
                "next_obs": next_observation,
                "action": action,
                "reward": reward,
                "done": done,
                "hidden_state": hidden_state,
                "next_hidden_state": next_hidden_state,
            }

            if is_continual_learner:
                continual_args["stationary_env"] = env.copy_to_stationary_env()
                continual_args["sample"] = sample

            observation = next_observation
            hidden_state = next_hidden_state

            if callback is not None:
                callback(locals(), globals())

            if render:
                env.render()
        all_rewards.append(episode_rewards)

    max_len = max(len(ep) for ep in all_rewards) if all_rewards else 0
    padded_rewards = np.nan * np.ones((len(all_rewards), max_len))
    for i, ep in enumerate(all_rewards):
        padded_rewards[i, : len(ep)] = ep

    if return_episode_rewards:
        return padded_rewards

    avg_reward = padded_rewards.mean(axis=1)
    std_reward = avg_reward.std()
    return avg_reward.mean(), std_reward

    total_reward = padded_rewards.sum(axis=1)
    return total_reward.mean(), total_reward.std()
