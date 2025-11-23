from typing import NamedTuple, Union, Optional

import numpy as np
import torch as th
from gymnasium import spaces

from .replay_buffer import BaseBuffer


class HiddenReplayBufferSamples(NamedTuple):
    """
    A sample of transitions from the replay buffer.

    :param observations: The observations.
    :param actions: The actions.
    :param next_observations: The next observations.
    :param dones: The dones.
    :param rewards: The rewards.
    :param hidden_states: The hidden states.
    :param next_hidden_states: The next hidden states.
    """

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

    :param buffer_size: The size of the replay buffer.
    :param observation_space: The observation space of the environment.
    :param action_space: The action space of the environment.
    :param hidden_state_space: The hidden state space of the environment.
    :param device: The device to use for training.
    :param rng: The random number generator.
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

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
        hidden_state: np.ndarray,
        next_hidden_state: np.ndarray,
    ) -> None:
        self.hidden_states[self.pos] = hidden_state
        self.next_hidden_states[self.pos] = next_hidden_state
        super().add(obs, next_obs, action, reward, done)

    def _get_samples(self, batch_inds: np.ndarray) -> HiddenReplayBufferSamples:
        """
        Get a batch of samples from the replay buffer.

        :param batch_inds: The indices of the samples to get.
        :return: A batch of samples.
        """
        samples = super()._get_samples(batch_inds)
        hidden_states = self.hidden_states[batch_inds]
        next_hidden_states = self.next_hidden_states[batch_inds]

        return HiddenReplayBufferSamples(
            *samples,
            hidden_states=th.as_tensor(hidden_states).to(self.device),
            next_hidden_states=th.as_tensor(next_hidden_states).to(self.device),
        )
