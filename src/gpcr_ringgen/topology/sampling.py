"""Topology sampling boundaries and a clearly marked Dummy provider."""

from gpcr_ringgen.core.interfaces import TopologyProvider
from gpcr_ringgen.core.types import RingTopology


class DummyTopologyProvider(TopologyProvider):
    """Return a fixed placeholder topology for architecture validation only."""

    def provide(self) -> RingTopology:
        return RingTopology(
            topology_id="dummy-topology",
            metadata={"dummy": True, "scientific": False},
        )
