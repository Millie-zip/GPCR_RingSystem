"""Tests that Dummy components honor the public ABC boundaries."""

from gpcr_ringgen.core.interfaces import (
    DiffusionModel,
    Sampler,
    ScaffoldProvider,
    TopologyProvider,
)
from gpcr_ringgen.models.diffusion.model import DummyDiffusionModel
from gpcr_ringgen.sampling.sampler import DummySampler
from gpcr_ringgen.scaffold.realization import DummyScaffoldProvider
from gpcr_ringgen.topology.sampling import DummyTopologyProvider


def test_dummy_components_subclass_public_interfaces() -> None:
    assert isinstance(DummyTopologyProvider(), TopologyProvider)
    assert isinstance(DummyScaffoldProvider(), ScaffoldProvider)
    assert isinstance(DummyDiffusionModel(), DiffusionModel)
    assert isinstance(DummySampler(), Sampler)
