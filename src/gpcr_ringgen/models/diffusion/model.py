"""Diffusion model boundary implementation for the Dummy Pipeline only."""

from copy import deepcopy

from gpcr_ringgen.core.interfaces import DiffusionModel
from gpcr_ringgen.core.types import DiffusionState, ScaffoldCondition


class DummyDiffusionModel(DiffusionModel):
    """Copy state metadata without performing noise or denoising operations."""

    def forward(
        self,
        state: DiffusionState,
        condition: ScaffoldCondition,
        timestep: int,
    ) -> DiffusionState:
        payload = deepcopy(state.payload)
        payload.update(
            {
                "dummy_model_applied": True,
                "scaffold_id": condition.scaffold_id,
                "topology_id": condition.topology.topology_id,
            }
        )
        metadata = deepcopy(state.metadata)
        metadata.update({"dummy": True, "scientific": False})
        return DiffusionState(payload=payload, timestep=timestep, metadata=metadata)
