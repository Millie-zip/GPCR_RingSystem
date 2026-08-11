"""End-to-end architecture test for the only implemented pipeline."""

from gpcr_ringgen.models.diffusion.model import DummyDiffusionModel
from gpcr_ringgen.sampling.sampler import DummySampler
from gpcr_ringgen.scaffold.realization import DummyScaffoldProvider
from gpcr_ringgen.topology.sampling import DummyTopologyProvider


def test_dummy_pipeline_runs_with_optional_guidance_disabled() -> None:
    topology = DummyTopologyProvider().provide()
    condition = DummyScaffoldProvider().provide(topology)
    molecules = DummySampler().sample(
        model=DummyDiffusionModel(),
        condition=condition,
        guidance=None,
    )

    assert len(molecules) == 1
    result = molecules[0]
    assert result.metadata["dummy"] is True
    assert result.metadata["scientific"] is False
    assert result.representation["topology_id"] == "dummy-topology"
    assert result.representation["scaffold_id"] == "dummy-scaffold"
