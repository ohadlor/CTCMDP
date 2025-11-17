from dataclasses import dataclass, asdict

from stable_baselines3.td3 import TD3
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise, ActionNoise
import numpy as np


@dataclass
class TD3SB3Params:
    action_noise: ActionNoise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(1), sigma=0.5 * np.ones(1))
    stats_window_size: int = 1
    policy: str = "MlpPolicy"

    def asdict(self):
        return asdict(self)
