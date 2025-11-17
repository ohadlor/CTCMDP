from abc import ABC, abstractmethod


class Annealing(ABC):
    def __init__(self, start: float, end: float = None, steps: int = None):
        self.start = start
        self.end = end
        self.steps = steps
        self.t = 0

    @abstractmethod
    def _get_value(self) -> float:
        pass

    def __call__(self) -> float:
        val = self._get_value()
        self.t += 1
        return val

    def prune_steps(self) -> int:
        if self.steps is None:
            return self.t
        return min(self.t, self.steps)


class LinearAnnealing(Annealing):
    def __init__(self, start: float, end: float, steps: int):
        super().__init__(start, end, steps)

    def _get_value(self) -> float:
        if self.steps == 0:
            return self.end
        t = self.prune_steps()
        return self.start + (self.end - self.start) * t / self.steps


class ExponentialAnnealing(Annealing):
    def __init__(self, start: float, decay_rate: float):
        super().__init__(start)
        self.decay_rate = decay_rate

    def _get_value(self) -> float:
        return self.start * self.decay_rate**self.t
