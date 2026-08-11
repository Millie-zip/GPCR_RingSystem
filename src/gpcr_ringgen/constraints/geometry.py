"""Geometry constraint placeholder."""

from gpcr_ringgen.core.types import DiffusionState


def apply_geometry_constraints(state: DiffusionState) -> DiffusionState:
    """Apply future bond, clash, angle, or torsion constraints."""
    raise NotImplementedError("Geometry constraints are not implemented.")
