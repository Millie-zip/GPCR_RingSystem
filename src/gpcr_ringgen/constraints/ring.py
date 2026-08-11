"""Ring constraint placeholder."""

from gpcr_ringgen.core.types import DiffusionState, RingTopology


def apply_ring_constraints(
    state: DiffusionState,
    topology: RingTopology,
) -> DiffusionState:
    """Apply future topology preservation constraints."""
    raise NotImplementedError("Ring constraints are not implemented.")
