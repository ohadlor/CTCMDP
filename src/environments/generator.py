from omegaconf import DictConfig
import gymnasium as gym
from gymnasium.wrappers import FrameStackObservation, TimeAwareObservation, FlattenObservation

from .env_utils import get_param_bounds
from .wrappers import TCRMDP, SplitActionObservationSpace, RobustWrapper  # , BernoulliTruncation


def create_env(cfg: DictConfig) -> gym.Env:
    """
    Creates the environment.

    Parameters
    ----------
    cfg : DictConfig
        The configuration object.
    run_dir : str
        The run directory.

    Returns
    -------
    gym.Env
        The created environment.
    """
    # Define gym env
    env = gym.make(cfg.env.id)

    train_radius = cfg.agent.get("radius", None)  # train_radius used for training robust algs (tc_m2td3)
    test_radius = cfg.env.get("radius", None)  # If test_radius is none, we are not in test
    augment_reward = cfg.env.get("augment_reward", False)

    # If training not robust algorithm (TD3), return basic env
    if test_radius is None and train_radius is None:
        env = RobustWrapper(env, augment_reward=augment_reward)
        if cfg.env.get("time_aware", False):
            env = TimeAwareObservation(env)
        return env

    param_bounds = get_param_bounds(cfg.env.id, augment_reward)
    if cfg.env.get("domain_randomization", False):
        env = RobustWrapper(env, augment_reward=augment_reward, domain_space=param_bounds, seed=cfg.seed)
    else:
        env = RobustWrapper(env, augment_reward=augment_reward, seed=cfg.seed)

    if test_radius is not None:
        # Truncation is sampled true with probability p
        pass
        # env = BernoulliTruncation(env, seed=cfg.seed)

    if cfg.env.get("time_aware", False):
        env = TimeAwareObservation(env)

    if cfg.agent.get("variant", None) == "stacked":
        env = FlattenObservation(FrameStackObservation(env, stack_size=2))

    shrink_factor = cfg.env.get("shrink_factor", 0)
    # If we are in test use the test_radius, else in training use train_radius
    radius = test_radius if test_radius is not None else train_radius
    env = SplitActionObservationSpace(TCRMDP(env, param_bounds, radius, shrink_factor))

    return env
