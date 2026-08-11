"""Geometry evaluation placeholder."""

from typing import Any, Sequence

from gpcr_ringgen.core.types import GeneratedMolecule


def evaluate_geometry(molecules: Sequence[GeneratedMolecule]) -> dict[str, Any]:
    """Evaluate geometry after its criteria are defined."""
    raise NotImplementedError("Geometry evaluation is not implemented.")
