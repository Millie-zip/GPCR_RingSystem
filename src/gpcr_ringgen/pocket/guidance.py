"""Optional Pocket Guidance plugin boundary."""

from gpcr_ringgen.core.interfaces import Guidance
from gpcr_ringgen.core.types import DiffusionState, PocketRepresentation


class PocketGuidance(Guidance):
    """Placeholder plugin that never becomes a mandatory sampler dependency."""

    def __init__(self, pocket: PocketRepresentation) -> None:
        self.pocket = pocket

    def apply(self, state: DiffusionState, timestep: int) -> DiffusionState:
        raise NotImplementedError("Pocket Guidance is not implemented.")
