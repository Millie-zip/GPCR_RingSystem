"""Tests for the lightweight registry."""

import pytest

from gpcr_ringgen.core.exceptions import RegistryError
from gpcr_ringgen.core.registry import Registry


def test_registry_registers_and_retrieves() -> None:
    registry = Registry[object]("model")
    component = object()
    registry.register("dummy", component)
    assert registry.get("dummy") is component
    assert registry.names() == ("dummy",)


def test_registry_rejects_duplicate_name() -> None:
    registry = Registry[int]("sampler")
    registry.register("dummy", 1)
    with pytest.raises(RegistryError, match="already registered"):
        registry.register("dummy", 2)


def test_registry_reports_unknown_name() -> None:
    registry = Registry[int]("guidance")
    with pytest.raises(RegistryError, match="Unknown"):
        registry.get("missing")
