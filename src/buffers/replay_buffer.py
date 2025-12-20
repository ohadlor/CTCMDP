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

        self.observations = np.zeros((self.buffer_size, *observation_space.shape), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, *action_space.shape), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size,), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size,), dtype=np.bool_)
        self.next_observations = np.zeros((self.buffer_size, *observation_space.shape), dtype=np.float32)

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
    A replay buffer that gives higher sampling priority to samples from the current episode.
    This is achieved by assigning a higher weight to these samples, which then decays over time.
    The implementation uses a sparse representation for weights and a two-stage sampling
    process to ensure computational efficiency, especially for large buffer sizes.

    :param buffer_size: The size of the replay buffer.
    :param observation_space: The observation space of the environment.
    :param action_space: The action space of the environment.
    :param current_episode_multiplier: The initial weight multiplier for samples in the current episode.
                                       A value of 1 means no extra weight.
    :param in_episode_increment_factor: The amount by which the weight of a sample is reduced at each step.
    :param device: The device to use for training.
    :param rng: The random number generator.
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        current_episode_multiplier: float = 1.0,
        in_episode_increment_factor: float = 0.001,
        device: str = "auto",
        rng: Optional[np.random.Generator] = None,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, rng)

        if current_episode_multiplier < 1.0:
            raise ValueError("current_episode_multiplier must be >= 1.0")

        self.episode_weights = {}  # Sparse storage for weights > 1.0
        self.current_episode_multiplier = current_episode_multiplier
        self.in_episode_increment_factor = in_episode_increment_factor

    def add(self, obs: np.ndarray, next_obs: np.ndarray, action: np.ndarray, reward: float, done: bool) -> None:
        pos = self.pos
        if self.full and pos in self.episode_weights:
            del self.episode_weights[pos]

        super().add(obs, next_obs, action, reward, done)

        if self.current_episode_multiplier > 1.0:
            self.episode_weights[pos] = self.current_episode_multiplier

    def sample(self, batch_size: int) -> BaseReplayBufferSamples:
        upper_bound = self.size()

        hot_indices = list(self.episode_weights.keys())
        num_hot = len(hot_indices)
        num_cold = upper_bound - num_hot

        batch_inds = []

        if num_hot > 0:
            hot_weights = np.array([self.episode_weights[i] for i in hot_indices])
            sum_hot_weights = np.sum(hot_weights)

            total_weight = num_cold + sum_hot_weights
            prob_hot_group = sum_hot_weights / total_weight

            num_to_sample_from_hot = self.rng.binomial(batch_size, prob_hot_group)

            if num_to_sample_from_hot > 0:
                hot_probabilities = hot_weights / sum_hot_weights
                hot_samples = self.rng.choice(
                    hot_indices, size=num_to_sample_from_hot, p=hot_probabilities, replace=True
                )
                batch_inds.extend(hot_samples)

        num_to_sample_from_cold = batch_size - len(batch_inds)

        if num_to_sample_from_cold > 0:
            # Uniformly sample from cold indices using rejection sampling
            hot_indices_set = set(hot_indices)
            cold_samples = []
            while len(cold_samples) < num_to_sample_from_cold:
                # Oversample to reduce the number of Python loops
                oversampling_factor = 1.1
                candidate_batch_size = int((num_to_sample_from_cold - len(cold_samples)) * oversampling_factor) + 10

                candidates = self.rng.integers(0, upper_bound, size=candidate_batch_size)
                new_samples = [c for c in candidates if c not in hot_indices_set]
                cold_samples.extend(new_samples)

            batch_inds.extend(cold_samples[:num_to_sample_from_cold])

        self.rng.shuffle(batch_inds)
        return self._get_samples(np.array(batch_inds))

    def end_episode(self) -> None:
        """Resets the weights of the current episode's samples back to 1."""
        self.episode_weights.clear()

    def update_episode_weights(self) -> None:
        """Decays the weights of the samples from the current episode."""
        indices_to_remove = []
        for idx, weight in self.episode_weights.items():
            new_weight = weight - self.in_episode_increment_factor
            if new_weight <= 1.0:
                indices_to_remove.append(idx)
            else:
                self.episode_weights[idx] = new_weight

        for idx in indices_to_remove:
            del self.episode_weights[idx]
