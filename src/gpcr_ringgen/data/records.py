"""JSONL records consumed by the real Stage 3 training path."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


def _rows(value: object, name: str, width: int | None = None) -> list[list[float]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a numeric row sequence")
    result: list[list[float]] = []
    for row in value:
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
            raise ValueError(f"{name} rows must be sequences")
        converted = [float(item) for item in row]
        if width is not None and len(converted) != width:
            raise ValueError(f"{name} rows must have width {width}")
        if not all(math.isfinite(item) for item in converted):
            raise ValueError(f"{name} contains NaN or Inf")
        result.append(converted)
    if result and any(len(row) != len(result[0]) for row in result):
        raise ValueError(f"{name} rows must have consistent widths")
    return result


def _ints(value: object, name: str) -> list[int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be an integer sequence")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise ValueError(f"{name} must contain integers")
    return [int(item) for item in value]


def _bools(value: object, name: str) -> list[bool]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a boolean sequence")
    if any(not isinstance(item, bool) for item in value):
        raise ValueError(f"{name} must contain booleans")
    return [bool(item) for item in value]


def _edge_index(value: object) -> list[list[int]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("edge_index must be a sequence")
    rows = list(value)
    if len(rows) == 2 and all(isinstance(row, Sequence) for row in rows):
        source, target = _ints(rows[0], "edge_index[0]"), _ints(rows[1], "edge_index[1]")
        if len(source) != len(target):
            raise ValueError("edge_index rows must have the same length")
        return [source, target]
    pairs = []
    for pair in rows:
        if not isinstance(pair, Sequence) or len(pair) != 2:
            raise ValueError("edge_index must have shape [2,E] or [E,2]")
        values = _ints(pair, "edge_index pair")
        pairs.append((values[0], values[1]))
    return [[pair[0] for pair in pairs], [pair[1] for pair in pairs]]


@dataclass(frozen=True)
class TrainingRecord:
    sample_id: str
    atom_features: list[list[float]]
    edge_index: list[list[int]]
    edge_features: list[list[float]]
    coordinates: list[list[float]]
    atom_types: list[int]
    bond_types: list[int]
    ring_atom_mask: list[bool] | None = None
    noise_mask: list[bool] | None = None
    descriptor: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TrainingRecord":
        graph = value.get("graph", value)
        if not isinstance(graph, Mapping):
            raise ValueError("record.graph must be a mapping")
        sample_id = value.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id.strip():
            raise ValueError("sample_id must be a non-empty string")
        record = cls(
            sample_id=sample_id,
            atom_features=_rows(graph.get("atom_features"), "atom_features"),
            edge_index=_edge_index(graph.get("edge_index")),
            edge_features=_rows(graph.get("edge_features"), "edge_features"),
            coordinates=_rows(graph.get("coordinates"), "coordinates", width=3),
            atom_types=_ints(graph.get("atom_types"), "atom_types"),
            bond_types=_ints(graph.get("bond_types"), "bond_types"),
            ring_atom_mask=None if value.get("ring_atom_mask") is None else _bools(value["ring_atom_mask"], "ring_atom_mask"),
            noise_mask=None if value.get("noise_mask") is None else _bools(value["noise_mask"], "noise_mask"),
            descriptor=[float(item) for item in value.get("descriptor", [])],
            metadata=dict(value.get("metadata", {})),
        )
        record.validate()
        return record

    def validate(self) -> None:
        atom_count = len(self.atom_features)
        if len(self.edge_index) != 2:
            raise ValueError("edge_index must have shape [2,E]")
        edge_count = len(self.edge_index[0])
        if not self.sample_id.strip() or atom_count < 1:
            raise ValueError("record needs a sample_id and at least one atom")
        if len(self.coordinates) != atom_count or len(self.atom_types) != atom_count:
            raise ValueError("atom features, coordinates, and atom types must align")
        if len(self.edge_index[1]) != edge_count or len(self.edge_features) != edge_count or len(self.bond_types) != edge_count:
            raise ValueError("edge features and bond types must align")
        if any(len(row) == 0 for row in self.atom_features):
            raise ValueError("atom feature rows cannot be empty")
        if self.edge_features and any(len(row) == 0 for row in self.edge_features):
            raise ValueError("edge feature rows cannot be empty")
        for source, target in zip(*self.edge_index):
            if source < 0 or target < 0 or source >= atom_count or target >= atom_count:
                raise ValueError("edge endpoint is out of range")
            if source == target:
                raise ValueError("self-loop edges are not supported")
        if len({tuple(sorted(edge)) for edge in zip(*self.edge_index)}) != edge_count:
            raise ValueError("each undirected edge must be stored exactly once")
        if any(item < 0 for item in (*self.atom_types, *self.bond_types)):
            raise ValueError("categorical labels must be non-negative")
        for mask, name in ((self.ring_atom_mask, "ring_atom_mask"), (self.noise_mask, "noise_mask")):
            if mask is not None and len(mask) != atom_count:
                raise ValueError(f"{name} must align with atoms")
        if self.ring_atom_mask is not None and self.noise_mask is not None:
            if any(a == b for a, b in zip(self.ring_atom_mask, self.noise_mask)):
                raise ValueError("ring_atom_mask and noise_mask must be complementary")
            if not any(self.noise_mask):
                raise ValueError("every sample needs at least one noise atom")
        if not all(math.isfinite(value) for value in self.descriptor):
            raise ValueError("descriptor contains NaN or Inf")

    def to_mapping(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": "stage3_training_record_v1",
            "sample_id": self.sample_id,
            "graph": {
                "atom_features": self.atom_features,
                "edge_index": self.edge_index,
                "edge_features": self.edge_features,
                "coordinates": self.coordinates,
                "atom_types": self.atom_types,
                "bond_types": self.bond_types,
            },
            "descriptor": self.descriptor,
            "ring_atom_mask": self.ring_atom_mask,
            "noise_mask": self.noise_mask,
            "metadata": self.metadata,
        }


__all__ = ["TrainingRecord"]
