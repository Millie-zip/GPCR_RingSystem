"""Diversity evaluation placeholder."""

from typing import Any, Sequence

from gpcr_ringgen.core.types import GeneratedMolecule


def evaluate_diversity(molecules: Sequence[GeneratedMolecule]) -> dict[str, Any]:
    """Evaluate diversity after the molecular similarity basis is selected."""
    raise NotImplementedError("Diversity evaluation is not implemented.")
