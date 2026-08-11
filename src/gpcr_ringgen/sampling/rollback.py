"""Rollback and re-noising boundary without a policy implementation."""

from abc import ABC, abstractmethod

from gpcr_ringgen.core.types import DiffusionState


class RollbackPolicy(ABC):
    """Future rollback decision and state transition contract."""

    @abstractmethod
    def apply(self, state: DiffusionState, timestep: int) -> DiffusionState:
        """Return a state after a future rollback policy is selected."""
