"""Model-decoupled sampler implementation for architecture validation."""

from __future__ import annotations

from typing import Sequence

from gpcr_ringgen.core.interfaces import DiffusionModel, Guidance, Sampler
from gpcr_ringgen.core.types import GeneratedMolecule, ScaffoldCondition
from gpcr_ringgen.models.diffusion.state import make_dummy_state


class DummySampler(Sampler):
    """Run one model call and wrap its state as non-scientific output records."""

    def sample(
        self,
        model: DiffusionModel,
        condition: ScaffoldCondition,
        guidance: Guidance | None = None,
        num_samples: int = 1,
    ) -> Sequence[GeneratedMolecule]:
        if isinstance(num_samples, bool) or not isinstance(num_samples, int):
            raise TypeError("num_samples must be an integer")
        if num_samples < 1:
            raise ValueError("num_samples must be at least 1")

        molecules: list[GeneratedMolecule] = []
        for index in range(num_samples):
            state = model.forward(make_dummy_state(), condition, timestep=0)
            if guidance is not None:
                state = guidance.apply(state, timestep=0)
            molecules.append(
                GeneratedMolecule(
                    molecule_id=f"dummy-molecule-{index + 1:04d}",
                    representation=dict(state.payload),
                    metadata={
                        "dummy": True,
                        "scientific": False,
                        "timestep": state.timestep,
                    },
                )
            )
        return molecules
