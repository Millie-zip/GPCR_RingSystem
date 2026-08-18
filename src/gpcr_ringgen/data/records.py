"""Serializable records for the real Stage 3 training asset library.

The record format is deliberately independent from SQLite.  The database
converter writes one JSON object per line and the training loader converts
those objects into the existing typed Stage 3 contracts.  No feature meaning
is inferred here: ``atom_features`` and ``edge_features`` must already follow
the versioned manifest used to build the asset library.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

from gpcr_ringgen.core.types import (
    BatchMask,
    BatchMetadata,
    DescriptorBatch,
    MolecularGraph,
    PocketBatch,
    Stage2Output,
    Stage3Input,
)


def _as_float_rows(value: object, name: str, *, width: int | None = None) -> list[list[float]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of numeric rows")
    rows: list[list[float]] = []
    for row in value:
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
            raise ValueError(f"every {name} row must be a numeric sequence")
        converted = [float(item) for item in row]
        if width is not None and len(converted) != width:
            raise ValueError(f"every {name} row must have width {width}")
        if not all(math.isfinite(item) for item in converted):
            raise ValueError(f"{name} contains NaN or Inf")
        rows.append(converted)
    if rows and width is None:
        width = len(rows[0])
        if any(len(row) != width for row in rows):
            raise ValueError(f"{name} rows must have a consistent width")
    return rows


def _as_int_vector(value: object, name: str) -> list[int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be an integer sequence")
    result: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise ValueError(f"{name} must contain integers")
        result.append(int(item))
    return result


def _as_bool_vector(value: object, name: str) -> list[bool]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a boolean sequence")
    if any(not isinstance(item, bool) for item in value):
        raise ValueError(f"{name} must contain booleans")
    return [bool(item) for item in value]


def _edge_index(value: object) -> list[list[int]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("edge_index must be a sequence")
    rows = list(value)
    if not rows:
        return [[], []]
    if len(rows) == 2 and all(isinstance(row, Sequence) for row in rows):
        source = _as_int_vector(rows[0], "edge_index[0]")
        target = _as_int_vector(rows[1], "edge_index[1]")
        if len(source) != len(target):
            raise ValueError("edge_index source and target lengths differ")
        return [source, target]
    pairs: list[tuple[int, int]] = []
    for pair in rows:
        if not isinstance(pair, Sequence) or len(pair) != 2:
            raise ValueError("edge_index must have shape [2, E] or [E, 2]")
        source, target = pair
        if any(isinstance(item, bool) or not isinstance(item, int) for item in pair):
            raise ValueError("edge_index must contain integers")
        pairs.append((int(source), int(target)))
    return [
        [source for source, _ in pairs],
        [target for _, target in pairs],
    ]


@dataclass(frozen=True)
class TrainingRecord:
    """One molecule plus one ring-system training view.

    ``ring_atom_mask`` and ``noise_mask`` are persisted for test-time audit and
    weak-constraint evaluation.  The default training collate intentionally
    does not place them in ``BatchMask`` and therefore does not apply them to
    the training loss.
    """

    sample_id: str
    atom_features: list[list[float]]
    edge_index: list[list[int]]
    edge_features: list[list[float]]
    coordinates: list[list[float]]
    atom_types: list[int]
    bond_types: list[int]
    descriptor: list[float] = field(default_factory=list)
    descriptor_mask: list[bool] = field(default_factory=list)
    pocket_coordinates: list[list[float]] = field(default_factory=list)
    pocket_features: list[list[float]] = field(default_factory=list)
    pocket_mask: list[bool] = field(default_factory=list)
    pocket_feature_dim: int = 1
    ring_atom_mask: list[bool] | None = None
    noise_mask: list[bool] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TrainingRecord":
        if not isinstance(value, Mapping):
            raise TypeError("training record must be a mapping")
        graph = value.get("graph", value)
        if not isinstance(graph, Mapping):
            raise ValueError("record.graph must be a mapping")
        descriptor = value.get("descriptor", {})
        if not isinstance(descriptor, Mapping):
            raise ValueError("record.descriptor must be a mapping")
        pocket = value.get("pocket", {})
        if not isinstance(pocket, Mapping):
            raise ValueError("record.pocket must be a mapping")
        sample_id = value.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id.strip():
            raise ValueError("sample_id must be a non-empty string")
        descriptor_values = descriptor.get("values", [])
        descriptor_mask = descriptor.get("mask", [True] * len(descriptor_values))
        pocket_coordinates = _as_float_rows(
            pocket.get("coordinates", []), "pocket.coordinates", width=3
        )
        pocket_features = _as_float_rows(
            pocket.get("features", []), "pocket.features"
        )
        pocket_feature_dim = int(pocket.get("feature_dim", 1))
        record = cls(
            sample_id=sample_id,
            atom_features=_as_float_rows(graph.get("atom_features"), "atom_features"),
            edge_index=_edge_index(graph.get("edge_index")),
            edge_features=_as_float_rows(graph.get("edge_features"), "edge_features"),
            coordinates=_as_float_rows(graph.get("coordinates"), "coordinates", width=3),
            atom_types=_as_int_vector(graph.get("atom_types"), "atom_types"),
            bond_types=_as_int_vector(graph.get("bond_types"), "bond_types"),
            descriptor=[float(item) for item in descriptor_values],
            descriptor_mask=_as_bool_vector(descriptor_mask, "descriptor.mask"),
            pocket_coordinates=pocket_coordinates,
            pocket_features=pocket_features,
            pocket_mask=_as_bool_vector(
                pocket.get("mask", [True] * len(pocket_coordinates)),
                "pocket.mask",
            ),
            pocket_feature_dim=pocket_feature_dim,
            ring_atom_mask=(
                None
                if value.get("ring_atom_mask") is None
                else _as_bool_vector(value["ring_atom_mask"], "ring_atom_mask")
            ),
            noise_mask=(
                None
                if value.get("noise_mask") is None
                else _as_bool_vector(value["noise_mask"], "noise_mask")
            ),
            metadata=dict(value.get("metadata", {})),
        )
        record.validate()
        return record

    def validate(self) -> None:
        if not self.sample_id.strip():
            raise ValueError("sample_id must be non-empty")
        atom_count = len(self.atom_features)
        if atom_count < 1:
            raise ValueError("training records must contain at least one atom")
        if len(self.coordinates) != atom_count or len(self.atom_types) != atom_count:
            raise ValueError("coordinates and atom_types must align with atoms")
        atom_width = len(self.atom_features[0]) if self.atom_features else 0
        if atom_width < 1 or any(len(row) != atom_width for row in self.atom_features):
            raise ValueError("atom_features must have a positive consistent width")
        edge_count = len(self.edge_index[0])
        if len(self.edge_index) != 2 or len(self.edge_index[1]) != edge_count:
            raise ValueError("edge_index must have shape [2, E]")
        if len(self.edge_features) != edge_count or len(self.bond_types) != edge_count:
            raise ValueError("edge features and bond types must align with edges")
        edge_width = len(self.edge_features[0]) if self.edge_features else 0
        if edge_count and (edge_width < 1 or any(len(row) != edge_width for row in self.edge_features)):
            raise ValueError("edge_features must have a positive consistent width")
        for source, target in zip(*self.edge_index):
            if source < 0 or target < 0 or source >= atom_count or target >= atom_count:
                raise ValueError("edge_index contains an out-of-range atom")
            if source == target:
                raise ValueError("training graph cannot contain self-loop edges")
        if any(value < 0 for value in (*self.atom_types, *self.bond_types)):
            raise ValueError("categorical labels must be non-negative")
        if len(self.descriptor_mask) != len(self.descriptor):
            raise ValueError("descriptor and descriptor_mask must align")
        if self.pocket_feature_dim < 1:
            raise ValueError("pocket_feature_dim must be positive")
        if len(self.pocket_coordinates) != len(self.pocket_mask):
            raise ValueError("pocket coordinates and mask must align")
        if self.pocket_coordinates and len(self.pocket_features) != len(self.pocket_coordinates):
            raise ValueError("pocket features and coordinates must align")
        if self.pocket_features and any(
            len(row) != self.pocket_feature_dim for row in self.pocket_features
        ):
            raise ValueError("pocket features do not match pocket_feature_dim")
        if self.ring_atom_mask is not None and len(self.ring_atom_mask) != atom_count:
            raise ValueError("ring_atom_mask must align with atoms")
        if self.noise_mask is not None:
            if len(self.noise_mask) != atom_count:
                raise ValueError("noise_mask must align with atoms")
            if self.ring_atom_mask is not None and any(
                noise == ring for noise, ring in zip(self.noise_mask, self.ring_atom_mask)
            ):
                raise ValueError("noise_mask must be the complement of ring_atom_mask")
            if not any(self.noise_mask):
                raise ValueError("every training view needs at least one noise atom")

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
            "descriptor": {
                "values": self.descriptor,
                "mask": self.descriptor_mask,
            },
            "pocket": {
                "coordinates": self.pocket_coordinates,
                "features": self.pocket_features,
                "mask": self.pocket_mask,
                "feature_dim": self.pocket_feature_dim,
            },
            "ring_atom_mask": self.ring_atom_mask,
            "noise_mask": self.noise_mask,
            "metadata": self.metadata,
        }

    def to_stage3_input(self, *, dtype: torch.dtype = torch.float32) -> Stage3Input:
        self.validate()
        device = torch.device("cpu")
        atom_features = torch.tensor(self.atom_features, dtype=dtype, device=device).unsqueeze(0)
        edge_index = torch.tensor(self.edge_index, dtype=torch.long, device=device).unsqueeze(0)
        edge_width = len(self.edge_features[0]) if self.edge_features else 1
        edge_features = torch.tensor(
            self.edge_features, dtype=dtype, device=device
        ).reshape(1, len(self.edge_features), edge_width)
        graph = MolecularGraph(
            atom_features=atom_features,
            edge_index=edge_index,
            edge_features=edge_features,
            coordinates=torch.tensor(self.coordinates, dtype=dtype, device=device).unsqueeze(0),
            atom_types=torch.tensor(self.atom_types, dtype=torch.long, device=device).unsqueeze(0),
            bond_types=torch.tensor(self.bond_types, dtype=torch.long, device=device).unsqueeze(0),
        )
        mask = BatchMask(
            atom_mask=torch.ones((1, len(self.atom_features)), dtype=torch.bool),
            edge_mask=torch.ones((1, len(self.edge_index[0])), dtype=torch.bool),
            descriptor_mask=torch.tensor(self.descriptor_mask, dtype=torch.bool).unsqueeze(0),
            pocket_mask=torch.tensor(self.pocket_mask, dtype=torch.bool).unsqueeze(0),
        )
        pocket_features = self.pocket_features
        if not pocket_features:
            pocket_features = []
        pocket = PocketBatch(
            coordinates=torch.tensor(self.pocket_coordinates, dtype=dtype).reshape(
                1, len(self.pocket_coordinates), 3
            ),
            features=torch.tensor(pocket_features, dtype=dtype).reshape(
                1, len(self.pocket_coordinates), self.pocket_feature_dim
            ),
        )
        stage2 = Stage2Output(
            scaffold=graph,
            mask=BatchMask(
                atom_mask=mask.atom_mask,
                edge_mask=mask.edge_mask,
            ),
            metadata=BatchMetadata(
                sample_ids=(self.sample_id,),
                records=(self.metadata,),
            ),
        )
        from gpcr_ringgen.data.transforms import stage2_output_to_stage3_input

        return stage2_output_to_stage3_input(
            stage2,
            descriptor=DescriptorBatch(
                torch.tensor(self.descriptor, dtype=dtype).reshape(1, len(self.descriptor))
            ),
            pocket=pocket,
            descriptor_mask=mask.descriptor_mask,
            pocket_mask=mask.pocket_mask,
        )


__all__ = ["TrainingRecord"]
