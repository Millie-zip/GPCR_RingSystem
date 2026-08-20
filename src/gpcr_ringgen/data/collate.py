"""Padding-aware collation for real Stage 3 records."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from gpcr_ringgen.core.types import (
    BatchMask, BatchMetadata, DescriptorBatch, MolecularGraph, PocketBatch,
    Stage2Output, Stage3Input,
)
from gpcr_ringgen.data.records import TrainingRecord
from gpcr_ringgen.data.transforms import stage2_output_to_stage3_input


def collate_molecules(
    records: Sequence[TrainingRecord],
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> Stage3Input:
    """Pad variable-size real records without introducing pocket features."""
    if not records:
        raise ValueError("cannot collate an empty record sequence")
    normalized = [
        record if isinstance(record, TrainingRecord) else TrainingRecord.from_mapping(record)
        for record in records
    ]
    for record in normalized:
        record.validate()
    atom_widths = {len(row) for record in normalized for row in record.atom_features}
    edge_widths = {len(row) for record in normalized for row in record.edge_features}
    descriptor_widths = {len(record.descriptor) for record in normalized}
    if len(atom_widths) != 1 or len(edge_widths) > 1 or len(descriptor_widths) != 1:
        raise ValueError("batch records must use consistent feature dimensions")
    atom_width = next(iter(atom_widths))
    edge_width = next(iter(edge_widths), 1)
    descriptor_width = next(iter(descriptor_widths))
    batch_size = len(normalized)
    max_atoms = max(len(record.atom_features) for record in normalized)
    max_edges = max(len(record.edge_index[0]) for record in normalized)
    target = torch.device(device)
    atom_features = torch.zeros((batch_size, max_atoms, atom_width), dtype=dtype, device=target)
    edge_features = torch.zeros((batch_size, max_edges, edge_width), dtype=dtype, device=target)
    edge_index = torch.zeros((batch_size, 2, max_edges), dtype=torch.long, device=target)
    coordinates = torch.zeros((batch_size, max_atoms, 3), dtype=dtype, device=target)
    atom_mask = torch.zeros((batch_size, max_atoms), dtype=torch.bool, device=target)
    edge_mask = torch.zeros((batch_size, max_edges), dtype=torch.bool, device=target)
    descriptor = torch.zeros((batch_size, descriptor_width), dtype=dtype, device=target)
    descriptor_mask = torch.zeros((batch_size, descriptor_width), dtype=torch.bool, device=target)
    for batch_index, record in enumerate(normalized):
        atoms = len(record.atom_features)
        edges = len(record.edge_index[0])
        atom_features[batch_index, :atoms] = torch.tensor(record.atom_features, dtype=dtype, device=target)
        coordinates[batch_index, :atoms] = torch.tensor(record.coordinates, dtype=dtype, device=target)
        atom_mask[batch_index, :atoms] = True
        if edges:
            edge_index[batch_index, :, :edges] = torch.tensor(record.edge_index, dtype=torch.long, device=target)
            edge_features[batch_index, :edges] = torch.tensor(record.edge_features, dtype=dtype, device=target)
            edge_mask[batch_index, :edges] = True
        if descriptor_width:
            descriptor[batch_index] = torch.tensor(record.descriptor, dtype=dtype, device=target)
            descriptor_mask[batch_index] = True
    stage2 = Stage2Output(
        scaffold=MolecularGraph(
            atom_features=atom_features,
            edge_index=edge_index,
            edge_features=edge_features,
            coordinates=coordinates,
        ),
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
            coordinates=torch.zeros((batch_size, 0, 3), dtype=dtype, device=target),
            features=torch.zeros((batch_size, 0, 1), dtype=dtype, device=target),
        ),
        descriptor_mask=descriptor_mask,
        pocket_mask=torch.zeros((batch_size, 0), dtype=torch.bool, device=target),
    )


__all__ = ["collate_molecules"]
