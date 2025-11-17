from omegaconf import DictConfig
import gymnasium
from gymnasium.wrappers import FrameStackObservation, TimeAwareObservation, RecordVideo

from .env_utils import get_param_bounds
from .wrappers import TCRMDP, SplitActionObservationSpace


def create_env(cfg: DictConfig, run_dir: str) -> tuple[gymnasium.Env, gymnasium.Env]:
    """Creates the environment, simulator, and hidden policy."""
    # Define gym env
    env_id = cfg.env.env_id
    max_episode_steps = cfg.env.get("max_episode_steps", None)
    if cfg.get("record", False):
        env = gymnasium.make(env_id, max_episode_steps=max_episode_steps, render_mode="rgb_array")
        # Record video every n steps
        n = 1e4
        env = RecordVideo(
            env, video_folder=run_dir + "/videos", video_length=200, step_trigger=lambda t: t % n == n - 1
        )
    else:
        env = gymnasium.make(env_id, max_episode_steps=max_episode_steps)

    if cfg.env.get("time_aware", False):
        env = TimeAwareObservation(env)

    simulator = None
    # Create the augmented environment with hidden variable
    if "rrls" in env_id:
        param_bounds = get_param_bounds(env_id)
        env = TCRMDP(env, param_bounds, cfg.env.radius)

        agent_variant = cfg.agent.get("variant", None)
        if agent_variant == "stacked":
            env = FrameStackObservation(env, num_frames=cfg.env.num_frames)
        env = SplitActionObservationSpace(env)

    return env, simulator
