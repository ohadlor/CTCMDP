from omegaconf import DictConfig
import gymnasium as gym
from gymnasium.wrappers import FrameStackObservation, TimeAwareObservation, FlattenObservation

from .env_utils import get_param_bounds
from .wrappers import TCRMDP, SplitActionObservationSpace, BernoulliTruncation, RobustWrapper


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
    radius = cfg.agent.get("radius", None)
    is_robust = False if radius is None else True

    env = gym.make(cfg.env.id)
    env_name = cfg.env.id.split("-")[0]

    env = RobustWrapper(env)
    # If training not robust algorithm, return basic env
    if not cfg.get("test", False) and not is_robust:
        if cfg.env.get("time_aware", False):
            env = TimeAwareObservation(env)
        return env

    param_bounds = get_param_bounds(env_name)
    if cfg.get("test", False):
        # Truncation is sampled true with probability p
        env = BernoulliTruncation(env, seed=cfg.master_seed)

    if cfg.env.get("time_aware", False):
        env = TimeAwareObservation(env)

    if cfg.agent.get("variant", None) == "stacked":
        env = FlattenObservation(FrameStackObservation(env, stack_size=2))

    shrink_factor = cfg.env.get("shrink_factor", 0)
    env = SplitActionObservationSpace(TCRMDP(env, param_bounds, radius, shrink_factor))

    return env
