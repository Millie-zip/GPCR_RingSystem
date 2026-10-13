"""Transform boundary with no atom or graph schema assumption."""

from abc import ABC, abstractmethod
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
    PocketField,
    Stage2Output,
    Stage3Input,
)


class MolecularTransform(ABC):
    """Transform one future molecular record."""

    @abstractmethod
    def __call__(self, record: Any) -> Any:
        """Return a transformed record."""


def _coerce_stage2_output(
    output: Stage2Output | Mapping[str, Any],
) -> Stage2Output:
    if isinstance(output, Stage2Output):
        return output
    if not isinstance(output, Mapping):
        raise TypeError("stage2_output must be Stage2Output or a mapping.")

    scaffold = output.get("scaffold")
    if not isinstance(scaffold, MolecularGraph):
        raise TypeError("stage2_output['scaffold'] must be MolecularGraph.")

    mask_value = output.get("mask")
    if isinstance(mask_value, BatchMask):
        mask = mask_value
    else:
        atom_mask = output.get("atom_mask")
        edge_mask = output.get("edge_mask")
        diffusion_mask = output.get("diffusion_mask")
        if not isinstance(atom_mask, Tensor) or not isinstance(edge_mask, Tensor):
            raise TypeError(
                "Stage 2 mock output must provide BatchMask or tensor "
                "atom_mask and edge_mask values."
            )
        if diffusion_mask is not None and not isinstance(diffusion_mask, Tensor):
            raise TypeError("Stage 2 mock diffusion_mask must be a tensor.")
        mask = BatchMask(
            atom_mask=atom_mask,
            edge_mask=edge_mask,
            diffusion_mask=diffusion_mask,
        )

    metadata_value = output.get("metadata")
    if isinstance(metadata_value, BatchMetadata):
        metadata = metadata_value
    else:
        sample_ids_value = output.get("sample_ids")
        if not isinstance(sample_ids_value, Sequence) or isinstance(
            sample_ids_value, (str, bytes)
        ):
            raise TypeError("Stage 2 mock output must provide sample_ids.")
        sample_ids = tuple(sample_ids_value)
        if metadata_value is None:
            records: tuple[Mapping[str, Any], ...] = tuple(
                {} for _ in sample_ids
            )
        elif isinstance(metadata_value, Sequence) and not isinstance(
            metadata_value, (str, bytes)
        ):
            records = tuple(metadata_value)
        else:
            raise TypeError("Stage 2 mock metadata must be a sequence of mappings.")
        metadata = BatchMetadata(sample_ids=sample_ids, records=records)

    return Stage2Output(scaffold=scaffold, mask=mask, metadata=metadata)


def stage2_output_to_stage3_input(
    stage2_output: Stage2Output | Mapping[str, Any],
    *,
    descriptor: DescriptorBatch,
    pocket: PocketBatch,
    descriptor_mask: Tensor | None = None,
    pocket_mask: Tensor | None = None,
    pocket_field: PocketField | None = None,
) -> Stage3Input:
    """Adapt typed or mock Stage 2 output to a validated Stage 3 input.

    Missing descriptor/pocket masks default to all-valid boolean masks. This
    adapter performs no descriptor calculation, pocket processing, field
    construction, voxelization, diffusion, or sampling.
    """
    stage2 = _coerce_stage2_output(stage2_output)
    stage2.validate()
    descriptor.validate()
    pocket.validate()

    if descriptor_mask is None:
        descriptor_mask = torch.ones(
            descriptor.values.shape,
            dtype=torch.bool,
            device=descriptor.values.device,
        )
    if pocket_mask is None:
        pocket_mask = torch.ones(
            pocket.coordinates.shape[:2],
            dtype=torch.bool,
            device=pocket.coordinates.device,
        )

    combined_mask = BatchMask(
        atom_mask=stage2.mask.atom_mask,
        edge_mask=stage2.mask.edge_mask,
        diffusion_mask=stage2.mask.diffusion_mask,
        descriptor_mask=descriptor_mask,
        pocket_mask=pocket_mask,
        ring_atom_mask=stage2.mask.ring_atom_mask,
        ring_pair_mask=stage2.mask.ring_pair_mask,
        ring_edge_mask=stage2.mask.ring_edge_mask,
    )
    result = Stage3Input(
        scaffold=stage2.scaffold,
        descriptor=descriptor,
        pocket=pocket,
        mask=combined_mask,
        metadata=stage2.metadata,
        pocket_field=pocket_field,
    )
    result.validate()
    return result


def add_ring_core_masks(
    stage2_output: Stage2Output,
    ring_atom_mask: Tensor,
) -> Stage2Output:
    """Attach ring-core masks to a sparse real-edge Stage 2 graph.

    ``ring_atom_mask`` is ``[B, N]`` bool. Stage 2 edges are assumed to be real
    chemical bonds, so every valid core-core edge is fixed-present. Route-A
    complete candidate pairs are constructed later by the training or inference
    graph builder.
    """

    if not isinstance(stage2_output, Stage2Output):
        raise TypeError("stage2_output must be a Stage2Output")
    stage2_output.validate()
    if ring_atom_mask.shape != stage2_output.mask.atom_mask.shape:
        raise ValueError("ring_atom_mask must have shape [B, N]")
    if ring_atom_mask.dtype != torch.bool:
        raise TypeError("ring_atom_mask must have dtype torch.bool")
    if ring_atom_mask.device != stage2_output.scaffold.atom_features.device:
        raise ValueError("ring_atom_mask must be on the graph device")
    if bool((ring_atom_mask & ~stage2_output.mask.atom_mask).any()):
        raise ValueError("ring_atom_mask must be a subset of atom_mask")
    batch_size, _, num_edges = stage2_output.scaffold.edge_index.shape
    ring_pair_mask = torch.zeros(
        (batch_size, num_edges), dtype=torch.bool, device=ring_atom_mask.device
    )
    for batch_index in range(batch_size):
        valid = stage2_output.mask.edge_mask[batch_index]
        endpoints = stage2_output.scaffold.edge_index[batch_index, :, valid]
        ring_pair_mask[batch_index, valid] = (
            ring_atom_mask[batch_index, endpoints[0]]
            & ring_atom_mask[batch_index, endpoints[1]]
        )
    result = Stage2Output(
        scaffold=stage2_output.scaffold,
        mask=BatchMask(
            atom_mask=stage2_output.mask.atom_mask,
            edge_mask=stage2_output.mask.edge_mask,
            descriptor_mask=stage2_output.mask.descriptor_mask,
            pocket_mask=stage2_output.mask.pocket_mask,
            ring_atom_mask=ring_atom_mask,
            ring_pair_mask=ring_pair_mask,
            ring_edge_mask=ring_pair_mask.clone(),
            diffusion_mask=stage2_output.mask.diffusion_mask,
        ),
        metadata=stage2_output.metadata,
    )
    result.validate()
    return result
