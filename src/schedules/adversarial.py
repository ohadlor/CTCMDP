import torch
import torch.nn as nn
import numpy as np
from numpy.random import Generator
from gymnasium import spaces

from .base_schedule import BaseActionSchedule


# TODO: clip actions, make into actor-critic agent
class AdversarialSchedule(BaseActionSchedule):
    def __init__(self, action_space: spaces.Box, observation_space: spaces.Box, model_path: str, rng: Generator):
        super().__init__(action_space, observation_space, rng=rng)
        obs_dim = observation_space["observation"].shape[0]
        psi_dim = observation_space["hidden"].shape[0]
        self.policy = nn.Sequential(
            nn.Linear(obs_dim + psi_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_space.shape[0]),
            nn.Tanh(),
        )
        # try: self.policy.load_state_dict(torch.load(model_path))
        # except: print("Could not load adversarial model, using random init.")

        self.policy.train()
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-4)

    @property
    def l2_radius(self):
        return self.action_space.high[0]

    def _action_selection(self, obs: spaces.Dict) -> np.ndarray:
        with torch.no_grad():
            state_input = torch.FloatTensor(np.concatenate([obs])).unsqueeze(0)
            action = self.policy(state_input).squeeze(0).numpy() * self.l2_radius
        return action

    def update(self, agent_critic, replay_buffer):
        print("Placeholder: Updating adversarial schedule...")
