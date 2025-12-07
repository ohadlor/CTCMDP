from typing import Any, Callable, Optional

import gymnasium as gym
import numpy as np
from joblib import Parallel, delayed

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

    if check_for_wrapper(env, TCRMDP):
        env.reset(seed=seed)
        env = env.copy_to_stationary_env()
        hidden_state = env.hidden_state

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


def _run_evaluation_iteration(
    i: int,
    seed: int,
    model: BaseAlgorithm,
    env: gym.Env,
    adversary_policy: Optional[BaseActionSchedule],
    total_timesteps: int,
    is_continual_learner: bool,
):
    """Helper function to run a single evaluation iteration in parallel."""
    if is_continual_learner:
        model.reset(seed)
    else:
        model.set_seed(seed)
    if adversary_policy:
        adversary_policy.set_seed(seed)
    logger = model.update_logger_path(model.tensorboard_log + f"/iter_{i}")

    ep_rewards = []
    iter_rewards = []
    observation, hidden_state, _ = env.reset(seed=seed)

    if adversary_policy is not None:
        adversary_policy.reset(start_state=hidden_state)

    continual_args = {}
    if is_continual_learner:
        continual_args["stationary_env"] = env.copy_to_stationary_env()

    for current_step in range(total_timesteps):
        # Get observation
        if getattr(model, "oracle_actor", False):
            observation = np.concatenate([observation, hidden_state])
        # Predict (and learn)
        action = model.predict(observation, **continual_args)
        if adversary_policy is not None:
            hidden_action = adversary_policy.step(hidden_state)
        else:
            hidden_action = np.zeros_like(hidden_state)

        # Step env
        next_observation, next_hidden_state, reward, terminated, truncated, _ = env.step(action, hidden_action)
        done = terminated or truncated
        sample = {
            "obs": observation,
            "next_obs": next_observation,
            "action": action,
            "reward": reward,
            "done": done,
        }
        # Add to continual learner buffer
        if is_continual_learner:
            model.add(sample)
            continual_args["stationary_env"] = env.copy_to_stationary_env()

        observation = next_observation
        hidden_state = next_hidden_state

        iter_rewards.append(reward)
        ep_rewards.append(reward)
        logger.add_scalar("rollout/avg_rew", np.mean(iter_rewards), current_step)
        logger.add_scalar("rollout/rew", reward, current_step)

        # Handle episode termination
        if done:
            observation, hidden_state, _ = env.reset(seed=seed)
            if adversary_policy is not None:
                adversary_policy.reset(start_state=hidden_state)

            logger.add_scalar("rollout/ep_rew", sum(ep_rewards), current_step)
            ep_rewards = []

    return iter_rewards


def evaluate_policy_hidden_state(
    env: gym.Env,
    model: BaseAlgorithm,
    adversary_policy: Optional[BaseActionSchedule] = None,
    total_timesteps: int = 100000,
    seeds: Optional[list[int]] = [],
    n_jobs: int = -1,
):
    """
    Evaluates a model's policy in a continual learning setup with a hidden state.

    This function is designed for environments where the agent's performance is
    evaluated over a series of iterations (episodes), each with a potentially
    different underlying system dynamic controlled by a hidden state. It supports
    continual learners and can incorporate an adversary policy that influences
    the hidden state.

    Parameters
    ----------
    env : gym.Env
        The Gymnasium environment, which should expose hidden states.
    model : BaseAlgorithm
        The algorithm to evaluate. It should be a subclass of `BaseAlgorithm`.
    adversary_policy : Optional[BaseActionSchedule], optional
        An optional policy for the adversary that acts on the hidden state,
        by default None.
    total_timesteps : int, optional
        The total number of timesteps for each evaluation, by default 100000.
    seeds : Optional[list[int]], optional
        A list of seeds for each evaluation iteration to ensure reproducibility,
        by default [].
    n_jobs : int, optional
        The number of jobs to run in parallel. -1 means using all available cores,
        by default -1.

    Returns
    -------
    tuple[float, float]
        The mean and standard deviation of the average reward across all episodes.
    """
    is_continual_learner = getattr(model, "is_continual_learner", False)

    with Parallel(n_jobs=n_jobs, verbose=10, backend="loky", mmap_mode="c") as parallel:
        all_rewards = parallel(
            delayed(_run_evaluation_iteration)(
                i,
                seed,
                model,
                env,
                adversary_policy,
                total_timesteps,
                is_continual_learner,
            )
            for i, seed in enumerate(seeds)
        )

    all_rewards = np.array(all_rewards)
    time_avg_reward = all_rewards.mean(axis=1)
    np.save(f"{model.tensorboard_log}/rewards.npy", all_rewards)
    return time_avg_reward.mean(), time_avg_reward.std()
