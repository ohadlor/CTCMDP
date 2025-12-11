from typing import NamedTuple, Optional, Union

import numpy as np
import torch as th
from gymnasium import spaces


class BaseReplayBufferSamples(NamedTuple):
    """
    A sample of transitions from the replay buffer.

    :param observations: The observations.
    :param actions: The actions.
    :param next_observations: The next observations.
    :param dones: The dones.
    :param rewards: The rewards.
    """

    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor


class BaseBuffer:
    """
    Base class for replay buffers.

    :param buffer_size: The size of the replay buffer.
    :param observation_space: The observation space of the environment.
    :param action_space: The action space of the environment.
    :param device: The device to use for training.
    :param rng: The random number generator.
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str],
        rng: Optional[np.random.Generator] = None,
    ):
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = device
        self.pos = 0
        self.full = False
        self.rng = np.random.default_rng() if rng is None else rng

        self.observations = np.zeros((self.buffer_size, *observation_space.shape), dtype=observation_space.dtype)
        self.actions = np.zeros((self.buffer_size, *action_space.shape), dtype=action_space.dtype)
        self.rewards = np.zeros((self.buffer_size,), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size,), dtype=np.float32)
        self.next_observations = np.zeros((self.buffer_size, *observation_space.shape), dtype=observation_space.dtype)

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
    ) -> None:
        self.observations[self.pos] = obs
        self.next_observations[self.pos] = next_obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def reset(self, seed: Optional[int] = None):
        """
        Clear the replay buffer.
        """
        self.pos = 0
        self.full = False
        self.rng = np.random.default_rng(seed)

    def sample(self, batch_size: int) -> BaseReplayBufferSamples:
        """
        Sample a batch of transitions from the replay buffer.

        :param batch_size: The size of the batch to sample.
        :return: A batch of samples.
        """
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = self.rng.integers(0, upper_bound, size=batch_size)
        return self._get_samples(batch_inds)

    def _get_samples(self, batch_inds: np.ndarray) -> BaseReplayBufferSamples:
        """
        Get a batch of samples from the replay buffer.

        :param batch_inds: The indices of the samples to get.
        :return: A batch of samples.
        """
        obs = self.observations[batch_inds]
        next_obs = self.next_observations[batch_inds]
        actions = self.actions[batch_inds]
        rewards = self.rewards[batch_inds].reshape(-1, 1)
        dones = self.dones[batch_inds].reshape(-1, 1)

        return BaseReplayBufferSamples(
            observations=th.as_tensor(obs, dtype=th.float32).to(self.device),
            actions=th.as_tensor(actions, dtype=th.float32).to(self.device),
            next_observations=th.as_tensor(next_obs, dtype=th.float32).to(self.device),
            dones=th.as_tensor(dones).to(self.device),
            rewards=th.as_tensor(rewards).to(self.device),
        )

    def log(self, logger, step):
        """
        Log the buffer size.

        :param logger: The logger to use.
        :param step: The current step.
        """
        logger.add_scalar("buffer/size", self.size(), step)

    @property
    def size(self) -> int:
        """
        Get the current size of the replay buffer.
        """
        return self.buffer_size if self.full else self.pos


class TimeIndexedReplayBuffer(BaseBuffer):
    """
    A replay buffer that adds a time index to each sample.

    :param buffer_size: The size of the replay buffer.
    :param observation_space: The observation space of the environment.
    :param action_space: The action space of the environment.
    :param gamma: The discount factor for the time indices.
    :param device: The device to use for training.
    :param rng: The random number generator.
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        beta: float = 0.5,
        c: float = 1.0,
        reset_delay: int = 10_000,
        device: str = "auto",
        rng: Optional[np.random.Generator] = None,
    ):
        super().__init__(buffer_size, observation_space, action_space, device)
        if rng is None:
            rng = np.random.default_rng()
        self.rng = rng
        self.time_indices = np.zeros((self.buffer_size,), dtype=np.int32)
        self.beta = beta
        self.c = c
        # Reset delay should be expected time steps to traverse 'radius' of hidden observation space
        # ~ hidden observation space radius / hidden step size (depends on nature of hidden action selector)
        self.reset_delay = reset_delay

    def add(self, obs, next_obs, action, reward, done) -> None:
        """
        Add a transition to the replay buffer.

        :param obs: The observation.
        :param next_obs: The next observation.
        :param action: The action.
        :param reward: The reward.
        :param done: The done flag.
        """
        pos = self.pos
        super().add(obs, next_obs, action, reward, done)
        self.time_indices[pos] = 0

    def sample(self, batch_size: int) -> BaseReplayBufferSamples:
        """
        Sample a batch of transitions from the replay buffer.
        Samples are weighted by the time index and discount factor,
        more recent samples have greater weight.

        :param batch_size: The size of the batch to sample.
        :return: A batch of samples.
        """
        upper_bound = self.buffer_size if self.full else self.pos
        weights = 1 / (self.c + self.time_indices[:upper_bound]) ** self.beta
        weights /= weights.sum()
        batch_inds = self.rng.choice(upper_bound, size=batch_size, p=weights.flatten())
        return self._get_samples(batch_inds)

    def increment_time_indices(self) -> None:
        """
        Increment the time index of all transitions in the buffer by 1.
        """
        n_entries = self.buffer_size if self.full else self.pos
        self.time_indices[:n_entries] += 1

    def reset_from_env(self) -> None:
        # Call when env is reset to add distance between resets for time_indices,
        # such that samples from previous resets are less likely to be called
        n_entries = self.buffer_size if self.full else self.pos
        self.time_indices[:n_entries] = self.reset_delay
