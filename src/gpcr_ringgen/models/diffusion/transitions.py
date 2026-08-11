"""Diffusion transition interface without forward or reverse algorithms."""

from abc import ABC, abstractmethod

from gpcr_ringgen.core.types import DiffusionState


class DiffusionTransition(ABC):
    """Apply a future transition to a shared state."""

    @abstractmethod
    def apply(self, state: DiffusionState, timestep: int) -> DiffusionState:
        """Return a transitioned state."""
