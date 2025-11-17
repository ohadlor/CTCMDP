from typing import NamedTuple, Union, Optional

import numpy as np
import torch as th
from gymnasium import spaces

from .replay_buffer import BaseBuffer


class HiddenReplayBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor
    hidden_states: th.Tensor
    next_hidden_states: th.Tensor


class HiddenReplayBuffer(BaseBuffer):
    """
    A replay buffer that adds hidden_state and next_hidden_state to the samples.
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        hidden_state_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        rng: Optional[np.random.Generator] = None,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, rng)
        self.hidden_state_space = hidden_state_space
        self.hidden_states = np.zeros((self.buffer_size, *hidden_state_space.shape), dtype=hidden_state_space.dtype)
        self.next_hidden_states = np.zeros(
            (self.buffer_size, *hidden_state_space.shape), dtype=hidden_state_space.dtype
        )

    def add(self, obs, next_obs, action, reward, done, infos, hidden_state, next_hidden_state) -> None:
        super().add(obs, next_obs, action, reward, done, infos)
        pos = (self.pos - 1 + self.buffer_size) % self.buffer_size
        self.hidden_states[pos] = hidden_state
        self.next_hidden_states[pos] = next_hidden_state

    def _get_samples(self, batch_inds: np.ndarray):
        samples = super()._get_samples(batch_inds)
        hidden_states = self.hidden_states[batch_inds]
        next_hidden_states = self.next_hidden_states[batch_inds]

        return HiddenReplayBufferSamples(
            *samples,
            hidden_states=th.as_tensor(hidden_states).to(self.device),
            next_hidden_states=th.as_tensor(next_hidden_states).to(self.device),
        )
