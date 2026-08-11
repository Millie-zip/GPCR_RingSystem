"""Topology evaluation placeholder."""

from typing import Any, Sequence

from gpcr_ringgen.core.types import GeneratedMolecule


def evaluate_topology(molecules: Sequence[GeneratedMolecule]) -> dict[str, Any]:
    """Evaluate topology after representations and metrics are defined."""
    raise NotImplementedError("Topology evaluation is not implemented.")
