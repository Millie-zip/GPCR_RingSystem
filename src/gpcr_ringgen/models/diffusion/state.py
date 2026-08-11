"""Helpers around the schema-free shared diffusion state."""

from gpcr_ringgen.core.types import DiffusionState


def make_dummy_state() -> DiffusionState:
    """Create the input state used solely by the Dummy Pipeline."""
    return DiffusionState(
        payload={"status": "dummy-initialized"},
        timestep=0,
        metadata={"dummy": True, "scientific": False},
    )
