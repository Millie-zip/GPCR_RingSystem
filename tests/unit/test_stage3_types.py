"""Tests for the Stage 2/Descriptor/Pocket/Field to Stage 3 contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pytest
import torch

from gpcr_ringgen.core.types import (
    BatchMask,
    BatchMetadata,
    DescriptorBatch,
    MolecularGraph,
    PocketBatch,
    Stage2Output,
    Stage3DiffusionState,
)
from gpcr_ringgen.data.transforms import stage2_output_to_stage3_input


@dataclass
class DummyPocketField:
    """Test-only field contract implementation with no scientific behavior."""

    _batch_size: int
    _device: torch.device

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def device(self) -> torch.device:
        return self._device

    def validate(self, *, batch_size: int, device: torch.device) -> None:
        if self.batch_size != batch_size:
            raise ValueError("Dummy field batch size mismatch.")
        if self.device != device:
            raise ValueError("Dummy field device mismatch.")


def _make_contract_parts(
    batch_size: int = 1,
) -> tuple[Stage2Output, DescriptorBatch, PocketBatch, torch.Tensor, torch.Tensor]:
    num_atoms, num_edges = 3, 2
    atom_features = torch.randn(batch_size, num_atoms, 4)
    edge_index = torch.tensor(
        [[[0, 1], [1, 2]]] * batch_size,
        dtype=torch.long,
    )
    edge_features = torch.randn(batch_size, num_edges, 2)
    coordinates = torch.randn(batch_size, num_atoms, 3)
    atom_mask = torch.ones(batch_size, num_atoms, dtype=torch.bool)
    edge_mask = torch.ones(batch_size, num_edges, dtype=torch.bool)

    if batch_size > 1:
        atom_mask[-1, -1] = False
        edge_mask[-1, -1] = False
        edge_index[-1, :, -1] = -1

    scaffold = MolecularGraph(
        atom_features=atom_features,
        edge_index=edge_index,
        edge_features=edge_features,
        coordinates=coordinates,
    )
    metadata = BatchMetadata(
        sample_ids=tuple(f"sample-{index}" for index in range(batch_size)),
        records=tuple({"source": "mock-stage2"} for _ in range(batch_size)),
    )
    stage2 = Stage2Output(
        scaffold=scaffold,
        mask=BatchMask(atom_mask=atom_mask, edge_mask=edge_mask),
        metadata=metadata,
    )
    descriptor = DescriptorBatch(torch.randn(batch_size, 5))
    pocket = PocketBatch(
        coordinates=torch.randn(batch_size, 4, 3),
        features=torch.randn(batch_size, 4, 3),
    )
    descriptor_mask = torch.ones(batch_size, 5, dtype=torch.bool)
    pocket_mask = torch.ones(batch_size, 4, dtype=torch.bool)
    return stage2, descriptor, pocket, descriptor_mask, pocket_mask


def test_stage3_input_builds_from_stage2_output() -> None:
    stage2, descriptor, pocket, descriptor_mask, pocket_mask = _make_contract_parts()
    pocket_field = DummyPocketField(1, torch.device("cpu"))

    result = stage2_output_to_stage3_input(
        stage2,
        descriptor=descriptor,
        pocket=pocket,
        descriptor_mask=descriptor_mask,
        pocket_mask=pocket_mask,
        pocket_field=pocket_field,
    )

    assert result.batch_size == 1
    assert result.metadata.sample_ids == ("sample-0",)
    assert result.pocket_field is pocket_field


def test_adapter_accepts_mock_mapping_without_stage2_model() -> None:
    stage2, descriptor, pocket, _, _ = _make_contract_parts()
    mock_output: Mapping[str, Any] = {
        "scaffold": stage2.scaffold,
        "atom_mask": stage2.mask.atom_mask,
        "edge_mask": stage2.mask.edge_mask,
        "sample_ids": ["mapping-sample"],
        "metadata": [{"source": "plain-mapping"}],
    }

    result = stage2_output_to_stage3_input(
        mock_output,
        descriptor=descriptor,
        pocket=pocket,
    )

    assert result.metadata.sample_ids == ("mapping-sample",)
    assert result.mask.descriptor_mask is not None
    assert bool(result.mask.descriptor_mask.all())


def test_batch_size_greater_than_one_preserves_mask_semantics() -> None:
    stage2, descriptor, pocket, descriptor_mask, pocket_mask = _make_contract_parts(2)
    descriptor_mask[1, -1] = False
    pocket_mask[1, -2:] = False

    result = stage2_output_to_stage3_input(
        stage2,
        descriptor=descriptor,
        pocket=pocket,
        descriptor_mask=descriptor_mask,
        pocket_mask=pocket_mask,
    )

    assert result.batch_size == 2
    assert not bool(result.mask.atom_mask[1, -1])
    assert not bool(result.mask.edge_mask[1, -1])
    assert not bool(result.mask.descriptor_mask[1, -1])
    assert result.mask.pocket_mask[1].tolist() == [True, True, False, False]


def test_shape_validation_reports_descriptor_mask_mismatch() -> None:
    stage2, descriptor, pocket, _, pocket_mask = _make_contract_parts()
    wrong_shape = torch.ones(1, 6, dtype=torch.bool)

    with pytest.raises(ValueError, match="descriptor_mask shape"):
        stage2_output_to_stage3_input(
            stage2,
            descriptor=descriptor,
            pocket=pocket,
            descriptor_mask=wrong_shape,
            pocket_mask=pocket_mask,
        )


def test_dtype_validation_rejects_integer_atom_features() -> None:
    stage2, descriptor, pocket, _, _ = _make_contract_parts()
    stage2.scaffold.atom_features = stage2.scaffold.atom_features.to(torch.long)

    with pytest.raises(TypeError, match="atom_features.*floating-point"):
        stage2_output_to_stage3_input(
            stage2,
            descriptor=descriptor,
            pocket=pocket,
        )


def test_mask_validation_rejects_non_boolean_mask() -> None:
    stage2, descriptor, pocket, _, _ = _make_contract_parts()
    stage2.mask.atom_mask = stage2.mask.atom_mask.to(torch.float32)

    with pytest.raises(TypeError, match="atom_mask.*torch.bool"):
        stage2_output_to_stage3_input(
            stage2,
            descriptor=descriptor,
            pocket=pocket,
        )


def test_invalid_edge_endpoint_has_clear_error() -> None:
    stage2, descriptor, pocket, _, _ = _make_contract_parts()
    stage2.scaffold.edge_index[0, 1, 0] = 99

    with pytest.raises(ValueError, match="out-of-range endpoint"):
        stage2_output_to_stage3_input(
            stage2,
            descriptor=descriptor,
            pocket=pocket,
        )


def test_stage3_diffusion_state_contract_validates_timestep_dtype() -> None:
    stage2, descriptor, pocket, descriptor_mask, pocket_mask = _make_contract_parts()
    stage3_input = stage2_output_to_stage3_input(
        stage2,
        descriptor=descriptor,
        pocket=pocket,
        descriptor_mask=descriptor_mask,
        pocket_mask=pocket_mask,
    )
    state = Stage3DiffusionState(
        atom_features=stage3_input.scaffold.atom_features,
        edge_index=stage3_input.scaffold.edge_index,
        edge_features=stage3_input.scaffold.edge_features,
        coordinates=stage3_input.scaffold.coordinates,
        descriptor=stage3_input.descriptor.values,
        timestep=torch.zeros(1, dtype=torch.long),
        mask=stage3_input.mask,
        metadata=stage3_input.metadata,
    )
    state.validate()

    state.timestep = state.timestep.to(torch.float32)
    with pytest.raises(TypeError, match="timestep.*torch.long"):
        state.validate()


def test_stage3_diffusion_state_requires_coordinates() -> None:
    stage2, descriptor, pocket, _, _ = _make_contract_parts()
    stage3_input = stage2_output_to_stage3_input(
        stage2,
        descriptor=descriptor,
        pocket=pocket,
    )
    state = Stage3DiffusionState(
        atom_features=stage3_input.scaffold.atom_features,
        edge_index=stage3_input.scaffold.edge_index,
        edge_features=stage3_input.scaffold.edge_features,
        coordinates=None,
        descriptor=stage3_input.descriptor.values,
        timestep=torch.zeros(1, dtype=torch.long),
        mask=stage3_input.mask,
        metadata=stage3_input.metadata,
    )

    with pytest.raises(ValueError, match="coordinates are required"):
        state.validate()
