"""Padding-aware collation for real Stage 3 training records."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

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
from gpcr_ringgen.data.records import TrainingRecord
from gpcr_ringgen.data.transforms import stage2_output_to_stage3_input


def _record(value: TrainingRecord | Mapping[str, Any]) -> TrainingRecord:
    if isinstance(value, TrainingRecord):
        value.validate()
        return value
    if isinstance(value, Mapping):
        return TrainingRecord.from_mapping(value)
    raise TypeError("records must contain TrainingRecord objects or mappings")


def _same_width(records: Sequence[TrainingRecord], field: str) -> int:
    widths = {
        len(getattr(record, field)[0])
        for record in records
        if getattr(record, field)
    }
    if len(widths) != 1:
        raise ValueError(f"all records must use the same {field} width")
    return next(iter(widths))


def collate_molecules(
    records: Sequence[TrainingRecord | Mapping[str, Any]],
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> Stage3Input:
    """Collate variable-size records into the typed Stage 3 input contract.

    The collator pads only graph and pocket storage.  ``ring_atom_mask`` and
    ``noise_mask`` are retained in each record for test-time validation but are
    intentionally not copied into ``BatchMask``; training therefore has no
    weak ring/geometry constraint.
    """

    if not records:
        raise ValueError("cannot collate an empty record sequence")
    normalized = tuple(_record(item) for item in records)
    atom_feature_dim = _same_width(normalized, "atom_features")
    edge_feature_widths = {
        len(record.edge_features[0])
        for record in normalized
        if record.edge_features
    }
    if len(edge_feature_widths) != 1:
        raise ValueError("all records with edges must use the same edge feature width")
    edge_feature_dim = next(iter(edge_feature_widths), 1)
    descriptor_dims = {len(record.descriptor) for record in normalized}
    if len(descriptor_dims) != 1:
        raise ValueError("all records must use the same descriptor dimension")
    descriptor_dim = next(iter(descriptor_dims))
    pocket_feature_dims = {record.pocket_feature_dim for record in normalized}
    if len(pocket_feature_dims) != 1:
        raise ValueError("all records must use the same pocket feature dimension")
    pocket_feature_dim = next(iter(pocket_feature_dims))

    batch_size = len(normalized)
    max_atoms = max(len(record.atom_features) for record in normalized)
    max_edges = max(len(record.edge_index[0]) for record in normalized)
    max_pocket = max(len(record.pocket_coordinates) for record in normalized)
    target_device = torch.device(device)
    atom_features = torch.zeros(
        (batch_size, max_atoms, atom_feature_dim), dtype=dtype, device=target_device
    )
    edge_features = torch.zeros(
        (batch_size, max_edges, edge_feature_dim), dtype=dtype, device=target_device
    )
    edge_index = torch.zeros(
        (batch_size, 2, max_edges), dtype=torch.long, device=target_device
    )
    coordinates = torch.zeros(
        (batch_size, max_atoms, 3), dtype=dtype, device=target_device
    )
    atom_types = torch.zeros((batch_size, max_atoms), dtype=torch.long, device=target_device)
    bond_types = torch.zeros((batch_size, max_edges), dtype=torch.long, device=target_device)
    atom_mask = torch.zeros((batch_size, max_atoms), dtype=torch.bool, device=target_device)
    edge_mask = torch.zeros((batch_size, max_edges), dtype=torch.bool, device=target_device)
    descriptor = torch.zeros((batch_size, descriptor_dim), dtype=dtype, device=target_device)
    descriptor_mask = torch.zeros(
        (batch_size, descriptor_dim), dtype=torch.bool, device=target_device
    )
    pocket_coordinates = torch.zeros(
        (batch_size, max_pocket, 3), dtype=dtype, device=target_device
    )
    pocket_features = torch.zeros(
        (batch_size, max_pocket, pocket_feature_dim), dtype=dtype, device=target_device
    )
    pocket_mask = torch.zeros((batch_size, max_pocket), dtype=torch.bool, device=target_device)

    for batch_index, record in enumerate(normalized):
        atom_count = len(record.atom_features)
        edge_count = len(record.edge_index[0])
        pocket_count = len(record.pocket_coordinates)
        atom_features[batch_index, :atom_count] = torch.tensor(
            record.atom_features, dtype=dtype, device=target_device
        )
        coordinates[batch_index, :atom_count] = torch.tensor(
            record.coordinates, dtype=dtype, device=target_device
        )
        atom_types[batch_index, :atom_count] = torch.tensor(
            record.atom_types, dtype=torch.long, device=target_device
        )
        atom_mask[batch_index, :atom_count] = True
        if edge_count:
            edge_index[batch_index, :, :edge_count] = torch.tensor(
                record.edge_index, dtype=torch.long, device=target_device
            )
            edge_features[batch_index, :edge_count] = torch.tensor(
                record.edge_features, dtype=dtype, device=target_device
            )
            bond_types[batch_index, :edge_count] = torch.tensor(
                record.bond_types, dtype=torch.long, device=target_device
            )
            edge_mask[batch_index, :edge_count] = True
        if descriptor_dim:
            descriptor[batch_index] = torch.tensor(
                record.descriptor, dtype=dtype, device=target_device
            )
            descriptor_mask[batch_index] = torch.tensor(
                record.descriptor_mask, dtype=torch.bool, device=target_device
            )
        if pocket_count:
            pocket_coordinates[batch_index, :pocket_count] = torch.tensor(
                record.pocket_coordinates, dtype=dtype, device=target_device
            )
            pocket_features[batch_index, :pocket_count] = torch.tensor(
                record.pocket_features, dtype=dtype, device=target_device
            )
            pocket_mask[batch_index, :pocket_count] = torch.tensor(
                record.pocket_mask, dtype=torch.bool, device=target_device
            )

    graph = MolecularGraph(
        atom_features=atom_features,
        edge_index=edge_index,
        edge_features=edge_features,
        coordinates=coordinates,
        atom_types=atom_types,
        bond_types=bond_types,
    )
    stage2 = Stage2Output(
        scaffold=graph,
        mask=BatchMask(atom_mask=atom_mask, edge_mask=edge_mask),
        metadata=BatchMetadata(
            sample_ids=tuple(record.sample_id for record in normalized),
            records=tuple(record.metadata for record in normalized),
        ),
    )
    return stage2_output_to_stage3_input(
        stage2,
        descriptor=DescriptorBatch(descriptor),
        pocket=PocketBatch(
            coordinates=pocket_coordinates,
            features=pocket_features,
        ),
        descriptor_mask=descriptor_mask,
        pocket_mask=pocket_mask,
    )


__all__ = ["collate_molecules"]
