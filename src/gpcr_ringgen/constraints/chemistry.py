"""Chemistry constraint placeholder."""

from gpcr_ringgen.core.types import DiffusionState


def apply_chemistry_constraints(state: DiffusionState) -> DiffusionState:
    """Apply future valence and validity rules."""
    raise NotImplementedError("Chemistry constraints are not implemented.")
