from omegaconf import DictConfig
import gymnasium as gym
from gymnasium.wrappers import FrameStackObservation, TimeAwareObservation, RecordVideo, FlattenObservation

from .env_utils import get_param_bounds, name_to_env_id
from .wrappers import TCRMDP, SplitActionObservationSpace


def create_env(cfg: DictConfig, run_dir: str) -> tuple[gym.Env, gym.Env]:
    """Creates the environment, simulator, and hidden policy."""
    # Define gym env
    is_rrls = hasattr(cfg.env, "radius")
    env_id = name_to_env_id(cfg.env.name, is_rrls)
    max_episode_steps = cfg.env.get("max_episode_steps", None)
    if cfg.get("record", False):
        env = gym.make(env_id, max_episode_steps=max_episode_steps, render_mode="rgb_array")
        # Record video every n steps
        n = 1e4
        env = RecordVideo(
            env, video_folder=run_dir + "/videos", video_length=200, step_trigger=lambda t: t % n == n - 1
        )
    else:
        env = gym.make(env_id, max_episode_steps=max_episode_steps)

    if cfg.env.get("time_aware", False):
        env = TimeAwareObservation(env)

    simulator = None
    # Create the augmented environment with hidden variable
    if is_rrls:
        agent_variant = cfg.agent.get("variant", "")
        if agent_variant == "stacked":
            env = FlattenObservation(FrameStackObservation(env, stack_size=2))
        param_bounds = get_param_bounds(cfg.env.name)
        env = TCRMDP(env, param_bounds, cfg.env.radius)
        env = SplitActionObservationSpace(env)

    return env, simulator
