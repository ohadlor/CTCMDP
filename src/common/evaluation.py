from typing import Any, Callable, Optional
import time

import gymnasium as gym
import numpy as np

from src.agents.base_algorithm import BaseAlgorithm
from src.schedules import BaseActionSchedule
from src.environments.env_utils import check_for_wrapper
from src.environments.wrappers import TCRMDP


def evaluate_policy(
    model,
    env: gym.Env,
    n_eval_episodes: int = 10,
    render: bool = False,
    callback: Optional[Callable[[dict[str, Any], dict[str, Any]], None]] = None,
    average_reward: bool = False,
    seed: Optional[int] = None,
    logging_freq: int = 1000,
) -> np.ndarray:
    """
    Evaluates the policy of a model in a given Gymnasium environment.

    This function runs the model's policy for a specified number of episodes
    and collects the rewards. It can handle both standard gym environments and
    those wrapped with TCRMDP. For TCRMDP environments, it creates a stationary
    version of the environment for evaluation.

    Parameters
    ----------
    model : object
        The model to be evaluated. It should have a `predict` method that
        takes an observation and returns an action.
    env : gym.Env
        The Gymnasium environment to evaluate the policy on.
    n_eval_episodes : int, optional
        The number of episodes to run for evaluation, by default 10.
    render : bool, optional
        Whether to render the environment during evaluation, by default False.
    callback : Optional[Callable[[dict[str, Any], dict[str, Any]], None]], optional
        A callback function to be called at each step with locals and globals,
        by default None.
    return_episode_rewards : bool, optional
        If True, returns the rewards for each episode, by default False. The rewards
        are padded with NaNs to ensure a consistent shape.
    average_reward : bool, optional
        If True, returns the mean and standard deviation of the average reward
        per episode, by default False.
    seed : Optional[int], optional
        The seed for the environment's random number generator, by default None.

    Returns
    -------
    np.ndarray or tuple[float, float]
        If `return_episode_rewards` is True, returns a 2D numpy array of shape
        (n_eval_episodes, max_episode_length) with rewards for each step.
        If `average_reward` is True, returns the mean and standard deviation of
        the average rewards per episode.
        Otherwise, returns the mean and standard deviation of the total reward
        per episode.
    """
    all_rewards = []

    # If robust alg, evaluate on the stationary env
    if check_for_wrapper(env, TCRMDP):
        env.reset(seed=seed)
        env = env.get_wrapper_attr("make_env")()
        hidden_state = np.array(list(env.get_params().values()))

    for i in range(n_eval_episodes):
        episode_rewards = []
        done = False
        observations, _ = env.reset(seed=seed + i)
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
    np.save(f"{model.tensorboard_log}/rewards.npy", padded_rewards)

    if average_reward:
        avg_reward = np.nanmean(padded_rewards, axis=1)
        std_reward = np.nanstd(avg_reward)
        return avg_reward.mean(), std_reward

    total_reward = np.nansum(padded_rewards, axis=1)
    return total_reward.mean(), total_reward.std()


def evaluate_policy_hidden_state(
    seed: int,
    model: BaseAlgorithm,
    env: gym.Env,
    adversary_policy: Optional[BaseActionSchedule],
    total_timesteps: int,
    logging_freq: int = 1000,
):
    is_continual_learner = getattr(model, "is_continual_learner", False)

    logger = model.logger

    ep_reward = 0
    avg_reward = 0
    total_rewards = []
    observation, hidden_state, _ = env.reset(seed=seed)

    if adversary_policy is not None:
        adversary_policy.reset(start_state=hidden_state)

    if getattr(model, "oracle_actor", False):
        observation = np.concatenate([observation, hidden_state])

    if hasattr(model, "stationary_env"):
        stationary_env = env.get_wrapper_attr("make_env")()
        model.set_stationary_env(stationary_env)

    start_time = time.time()
    for current_step in range(total_timesteps):
        # Predict
        action = model.predict(observation)
        if adversary_policy is not None:
            hidden_action = adversary_policy.step(hidden_state)
        else:
            hidden_action = np.zeros_like(hidden_state)

        # Step env
        next_observation, next_hidden_state, reward, terminated, truncated, _ = env.step(action, hidden_action)

        if getattr(model, "oracle_actor", False):
            next_observation = np.concatenate([next_observation, next_hidden_state])
        if is_continual_learner:
            sample = {
                "obs": observation,
                "next_obs": next_observation,
                "action": action,
                "reward": reward,
                "done": terminated,
            }
            # Add to continual learner buffer
            model.add(sample, truncated)
            if hasattr(model, "stationary_env"):
                model.update_stationary_env(env)
            # Learn
            model.learn(next_observation)

        observation = next_observation
        hidden_state = next_hidden_state

        total_rewards.append(reward)
        ep_reward += reward
        avg_reward += (reward - avg_reward) / (current_step + 1)

        if current_step % logging_freq == 0:
            total_time = time.time() - start_time
            start_time = time.time()
            logger.add_scalar("rollout/avg_rew", avg_reward, current_step)
            logger.add_scalar(f"time/avg_per_last_{logging_freq}_step", total_time / logging_freq, current_step)

        # Handle episode termination
        done = terminated or truncated
        if done:
            observation, hidden_state, _ = env.reset()
            if adversary_policy is not None:
                adversary_policy.reset(start_state=hidden_state)

            logger.add_scalar("rollout/ep_rew_mean", ep_reward, current_step)
            ep_reward = 0

    returns = np.array(total_rewards)
    return returns
