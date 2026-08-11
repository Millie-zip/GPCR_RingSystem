"""Topology capacity assessment placeholder."""

from gpcr_ringgen.core.types import PocketRepresentation, RingTopology


def assess_capacity(
    topology: RingTopology,
    pocket: PocketRepresentation | None = None,
) -> float:
    """Assess capacity after scoring and pocket matching are specified."""
    raise NotImplementedError("Topology capacity scoring is not implemented.")
