"""Noise schedule interface without a concrete scientific schedule."""

from abc import ABC, abstractmethod


class NoiseSchedule(ABC):
    """Return a future schedule value at a timestep."""

    @abstractmethod
    def value(self, timestep: int) -> float:
        """Return the value for one timestep."""
