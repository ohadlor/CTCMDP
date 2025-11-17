import torch
import torch.nn as nn
import numpy as np
from numpy.random import Generator
from gymnasium import spaces

from .hidden_action_selector import HiddenActionSelector


# TODO: clip actions, make into actor-critic agent
class AdversarialSchedule(HiddenActionSelector):
    def __init__(self, hidden_dim: int, l2_radius: float, obs_dim: int, psi_dim: int, model_path: str, rng: Generator):
        super().__init__(hidden_dim, l2_radius, rng=rng)
        self.policy = nn.Sequential(
            nn.Linear(obs_dim + psi_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, hidden_dim),
            nn.Tanh(),
        )
        # try: self.policy.load_state_dict(torch.load(model_path))
        # except: print("Could not load adversarial model, using random init.")

        self.policy.train()
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-4)

    def _action_selection(self, obs: spaces.Dict) -> np.ndarray:
        with torch.no_grad():
            state_input = torch.FloatTensor(np.concatenate([obs])).unsqueeze(0)
            action = self.policy(state_input).squeeze(0).numpy() * self.l2_radius
        return action

    def update(self, agent_critic, replay_buffer):
        print("Placeholder: Updating adversarial schedule...")
