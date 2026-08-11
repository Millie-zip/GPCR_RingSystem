"""Shared fixtures that are genuinely reused across test modules."""

from pathlib import Path

import pytest

from gpcr_ringgen.core.types import RingTopology


@pytest.fixture
def dummy_topology() -> RingTopology:
    """Return a non-scientific topology fixture."""
    return RingTopology("fixture-topology", {"dummy": True})


@pytest.fixture
def dummy_config_path() -> Path:
    """Return the minimal fixture configuration path."""
    return Path(__file__).parent / "fixtures" / "dummy.yaml"
