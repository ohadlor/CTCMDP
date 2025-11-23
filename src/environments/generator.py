from omegaconf import DictConfig
import gymnasium as gym
from gymnasium.wrappers import FrameStackObservation, TimeAwareObservation, FlattenObservation, TimeLimit
from rrls.wrappers import DomainRandomization

from .env_utils import get_param_bounds, name_to_env_id, remove_extra_time_wrapper
from .wrappers import TCRMDP, SplitActionObservationSpace


def create_env(cfg: DictConfig, run_dir: str) -> gym.Env:
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
    is_rrls = hasattr(cfg.agent, "radius")
    env_id = name_to_env_id(cfg.env.name, is_rrls)
    max_episode_steps = cfg.env.get("max_episode_steps", None)

    env = gym.make(env_id, max_episode_steps=-1)
    if is_rrls:
        param_bounds = get_param_bounds(cfg.env.name)
        # Initial hidden states are randomized upon reset
        env = DomainRandomization(env, params_bound=param_bounds)
    # Time limit must be added after domain randomization
    env = TimeLimit(env, max_episode_steps)

    # rrls seems to incorporate an extra timelimit wrapper
    env = remove_extra_time_wrapper(env)

    if cfg.env.get("time_aware", False):
        env = TimeAwareObservation(env)

    if cfg.agent.get("variant", "") == "stacked":
        env = FlattenObservation(FrameStackObservation(env, stack_size=2))
    if is_rrls:
        # Create the augmented environment with hidden variable
        env = TCRMDP(env, param_bounds, cfg.agent.radius)
        env = SplitActionObservationSpace(env)

    return env
