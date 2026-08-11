"""Tests for shared dataclass construction."""

from gpcr_ringgen.core.types import (
    DiffusionState,
    GeneratedMolecule,
    PocketRepresentation,
    RingTopology,
    ScaffoldCondition,
)


def test_core_dataclasses_initialize() -> None:
    topology = RingTopology("topology")
    scaffold = ScaffoldCondition("scaffold", topology)
    state = DiffusionState()
    pocket = PocketRepresentation("pocket")
    molecule = GeneratedMolecule("molecule")

    assert scaffold.topology is topology
    assert state.timestep == 0
    assert pocket.features == {}
    assert molecule.representation == {}
