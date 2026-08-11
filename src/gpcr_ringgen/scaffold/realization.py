"""Scaffold realization boundary and Dummy provider."""

from typing import Any

from gpcr_ringgen.core.interfaces import ScaffoldProvider
from gpcr_ringgen.core.types import RingTopology, ScaffoldCondition


class DummyScaffoldProvider(ScaffoldProvider):
    """Create a fixed placeholder condition for architecture validation."""

    def provide(self, topology: RingTopology) -> ScaffoldCondition:
        return ScaffoldCondition(
            scaffold_id="dummy-scaffold",
            topology=topology,
            metadata={"dummy": True, "scientific": False},
        )


def realize_scaffold(condition: ScaffoldCondition) -> Any:
    """Realize a scaffold only after its molecular representation is decided."""
    raise NotImplementedError("Scaffold realization is not implemented.")
