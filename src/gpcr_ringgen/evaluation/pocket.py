"""Pocket and target evaluation placeholder."""

from typing import Any, Sequence

from gpcr_ringgen.core.types import GeneratedMolecule, PocketRepresentation


def evaluate_pocket(
    molecules: Sequence[GeneratedMolecule],
    pocket: PocketRepresentation,
) -> dict[str, Any]:
    """Evaluate pocket compatibility after target metrics are approved."""
    raise NotImplementedError("Pocket evaluation is not implemented.")
