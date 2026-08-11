"""Pocket field placeholder."""

from typing import Any

from gpcr_ringgen.core.types import PocketRepresentation


def compute_pocket_fields(pocket: PocketRepresentation) -> dict[str, Any]:
    """Compute future pocket fields after their definitions are approved."""
    raise NotImplementedError("Pocket fields are not implemented.")
