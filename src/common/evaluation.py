from typing import Any, Callable, Optional
from tqdm import tqdm

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
    total_timesteps: int = 1e5,
    # render: bool = False,
    # callback: Optional[Callable[[dict[str, Any], dict[str, Any]], None]] = None,
    return_episode_rewards: bool = False,
    seeds: Optional[list[int]] = [],
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
    render : bool, optional
        Whether to render the environment, by default False.
    callback : Optional[Callable[[dict[str, Any], dict[str, Any]], None]], optional
        A callback function to be called at each step, by default None.
    return_episode_rewards : bool, optional
        If True, returns rewards for each episode, by default False. The rewards
        are padded with NaNs to ensure a consistent shape.
    seeds : Optional[list[int]], optional
        A list of seeds for each evaluation iteration to ensure reproducibility,
        by default [].

    Returns
    -------
    np.ndarray or tuple[float, float]
        If `return_episode_rewards` is True, returns a 2D numpy array of rewards.
        Otherwise, returns the mean and standard deviation of the average reward
        across all episodes.
    """
    all_rewards = []
    is_continual_learner = getattr(model, "is_continual_learner", False)
    continual_args = {}
    temp_env = env
    while hasattr(temp_env, "env"):
        if isinstance(temp_env, gym.wrappers.TimeLimit):
            max_steps = temp_env._max_episode_steps
            break
        temp_env = temp_env.env

    for i, seed in tqdm(enumerate(seeds), total=len(seeds), desc="Iteration Loop"):
        model.set_seed(seed)
        adversary_policy.set_seed(seed)
        logger = model.update_logger_path(model.tensorboard_log + f"/iter_{i}")

        ep_rewards = []
        iter_rewards = []
        observation, hidden_state, _ = env.reset(seed=seed)

        if adversary_policy is not None:
            adversary_policy.reset(start_state=hidden_state)
        if is_continual_learner:
            continual_args["stationary_env"] = env.copy_to_stationary_env()

        pbar = tqdm(total=max_steps, desc="Continual Learning Evaluation", leave=False)
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

            pbar.update(1)
            # if callback is not None:
            #     callback(locals(), globals())

            # if render:
            #     env.render()
        all_rewards.append(iter_rewards)
    pbar.close()

    all_rewards = np.array(all_rewards)
    time_avg_reward = all_rewards.mean(axis=1)
    if return_episode_rewards:
        return all_rewards
    return time_avg_reward.mean(), time_avg_reward.std()

    total_reward = all_rewards.sum(axis=1)
    return total_reward.mean(), total_reward.std()
