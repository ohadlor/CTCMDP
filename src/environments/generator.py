from omegaconf import DictConfig
import gymnasium as gym
from gymnasium.wrappers import FrameStackObservation, TimeAwareObservation, FlattenObservation
from rrls.wrappers import DomainRandomization

from .env_utils import get_param_bounds, name_to_env_id
from .wrappers import TCRMDP, SplitActionObservationSpace, BernoulliTruncation


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
    is_rrls = False if radius is None else True
    env_id = name_to_env_id(cfg.env.name, is_rrls)

    env = gym.make(env_id)
    # If test time add the following wrappers
    param_bounds = get_param_bounds(cfg.env.name)
    if cfg.get("test", False):
        # Initial hidden states are randomized upon reset
        env = DomainRandomization(env, params_bound=param_bounds)
        # truncation is sampled true with probability p
        env = BernoulliTruncation(env, seed=cfg.master_seed)

    if cfg.env.get("time_aware", False):
        env = TimeAwareObservation(env)

    if cfg.agent.get("variant", None) == "stacked":
        env = FlattenObservation(FrameStackObservation(env, stack_size=2))

    # If robust or test time make into TCRMDP
    if is_rrls or cfg.get("test", False):
        # Create the augmented environment with hidden variable
        shrink_factor = cfg.env.get("shrink_factor", 0)
        env = SplitActionObservationSpace(TCRMDP(env, param_bounds, radius, shrink_factor))

    return env
