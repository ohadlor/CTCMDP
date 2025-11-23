from abc import ABC, abstractmethod


class Annealing(ABC):
    """
    Base class for annealing schedules.

    :param start: The initial value.
    :param end: The final value.
    :param steps: The number of steps to anneal over.
    """
    def __init__(self, start: float, end: float = None, steps: int = None):
        self.start = start
        self.end = end
        self.steps = steps
        self.t = 0

    @abstractmethod
    def _get_value(self) -> float:
        """
        Get the current value of the annealing schedule.
        """
        pass

    def __call__(self) -> float:
        """
        Get the current value of the annealing schedule and increment the step count.
        """
        val = self._get_value()
        self.t += 1
        return val

    def prune_steps(self) -> int:
        """
        Prune the number of steps to be within the allowed range.
        """
        if self.steps is None:
            return self.t
        return min(self.t, self.steps)


class LinearAnnealing(Annealing):
    """
    Linearly anneal a value from a start to an end value over a given number of steps.

    :param start: The initial value.
    :param end: The final value.
    :param steps: The number of steps to anneal over.
    """
    def __init__(self, start: float, end: float, steps: int):
        super().__init__(start, end, steps)

    def _get_value(self) -> float:
        """
        Get the current value of the annealing schedule.
        """
        if self.steps == 0:
            return self.end
        t = self.prune_steps()
        return self.start + (self.end - self.start) * t / self.steps


class ExponentialAnnealing(Annealing):
    """
    Exponentially anneal a value from a start value with a given decay rate.

    :param start: The initial value.
    :param decay_rate: The decay rate.
    """
    def __init__(self, start: float, decay_rate: float):
        super().__init__(start)
        self.decay_rate = decay_rate

    def _get_value(self) -> float:
        """
        Get the current value of the annealing schedule.
        """
        return self.start * self.decay_rate**self.t
