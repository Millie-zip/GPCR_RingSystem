"""Small, stable data structures shared by all project modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RingTopology:
    """Placeholder identity and metadata for a ring topology."""

    topology_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScaffoldCondition:
    """Minimal conditioning object associated with a ring topology."""

    scaffold_id: str
    topology: RingTopology
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DiffusionState:
    """Schema-free state container for future diffusion implementations."""

    payload: dict[str, Any] = field(default_factory=dict)
    timestep: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PocketRepresentation:
    """Schema-free placeholder for a future GPCR pocket representation."""

    pocket_id: str
    features: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GeneratedMolecule:
    """Inspectable output record; not necessarily chemically meaningful."""

    molecule_id: str
    representation: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
