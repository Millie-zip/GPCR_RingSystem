"""Tests for masked isotropic-Gaussian Stage 3 coordinate initialization."""

from __future__ import annotations

import pytest
import torch

from gpcr_ringgen.core.types import (
    BatchMask,
    BatchMetadata,
    DescriptorBatch,
    MolecularGraph,
    PocketBatch,
    Stage2Output,
    Stage3Input,
)
from gpcr_ringgen.data.transforms import stage2_output_to_stage3_input
from gpcr_ringgen.models.diffusion import initialize_stage3_state


def _make_stage3_input(batch_size: int = 2) -> Stage3Input:
    if batch_size not in (1, 2):
        raise ValueError("test helper supports batch sizes 1 and 2")

    atom_features = torch.arange(
        batch_size * 4 * 3,
        dtype=torch.float32,
    ).reshape(batch_size, 4, 3)
    edge_index = torch.tensor(
        [[[0, 1, 2], [1, 2, 3]]] * batch_size,
        dtype=torch.long,
    )
    edge_features = torch.arange(
        batch_size * 3 * 2,
        dtype=torch.float32,
    ).reshape(batch_size, 3, 2)
    source_coordinates = torch.full((batch_size, 4, 3), 17.0)
    atom_mask = torch.ones(batch_size, 4, dtype=torch.bool)
    edge_mask = torch.ones(batch_size, 3, dtype=torch.bool)
    if batch_size == 2:
        atom_mask[1, 2:] = False
        edge_mask[1, 1:] = False
        edge_index[1, :, 1:] = -1

    metadata = BatchMetadata(
        sample_ids=tuple(f"sample-{index}" for index in range(batch_size)),
        records=tuple({"row": index} for index in range(batch_size)),
    )
    stage2 = Stage2Output(
        scaffold=MolecularGraph(
            atom_features=atom_features,
            atom_types=torch.tensor(
                [[0, 1, 2, 3]] * batch_size,
                dtype=torch.long,
            ).masked_fill(~atom_mask, 0),
            edge_index=edge_index,
            bond_types=torch.tensor(
                [[0, 1, 2]] * batch_size,
                dtype=torch.long,
            ).masked_fill(~edge_mask, 0),
            edge_features=edge_features,
            coordinates=source_coordinates,
        ),
        mask=BatchMask(
            atom_mask=atom_mask,
            edge_mask=edge_mask,
            diffusion_mask=atom_mask.clone(),
        ),
        metadata=metadata,
    )
    descriptor = DescriptorBatch(
        torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]][:batch_size],
            dtype=torch.float32,
        )
    )
    descriptor_mask = torch.ones(batch_size, 3, dtype=torch.bool)
    if batch_size == 2:
        descriptor_mask[1, -1] = False
    pocket = PocketBatch(
        coordinates=torch.zeros(batch_size, 2, 3),
        features=torch.ones(batch_size, 2, 2),
    )
    return stage2_output_to_stage3_input(
        stage2,
        descriptor=descriptor,
        pocket=pocket,
        descriptor_mask=descriptor_mask,
    )


def test_fixed_seed_produces_reproducible_coordinates() -> None:
    stage3_input = _make_stage3_input()

    first = initialize_stage3_state(stage3_input, seed=123)
    second = initialize_stage3_state(stage3_input, seed=123)

    assert torch.equal(first.coordinates, second.coordinates)


def test_initialization_does_not_mutate_global_torch_rng_state() -> None:
    stage3_input = _make_stage3_input()
    torch.manual_seed(999)
    expected_next_value = torch.rand(1)
    torch.manual_seed(999)

    initialize_stage3_state(stage3_input, seed=123)
    actual_next_value = torch.rand(1)

    assert torch.equal(actual_next_value, expected_next_value)


def test_different_seeds_usually_produce_different_coordinates() -> None:
    stage3_input = _make_stage3_input()

    first = initialize_stage3_state(stage3_input, seed=123)
    second = initialize_stage3_state(stage3_input, seed=456)

    valid_coordinates_differ = first.coordinates[stage3_input.mask.atom_mask] != second.coordinates[
        stage3_input.mask.atom_mask
    ]
    assert bool(valid_coordinates_differ.any())


def test_single_sample_coordinate_shape_dtype_device_and_timestep() -> None:
    stage3_input = _make_stage3_input(batch_size=1)

    state = initialize_stage3_state(stage3_input, seed=7, timestep=5)

    assert state.coordinates.shape == (1, 4, 3)
    assert state.coordinates.dtype == stage3_input.scaffold.atom_features.dtype
    assert state.coordinates.device == stage3_input.device
    assert state.coordinates.device.type == "cpu"
    assert state.timestep.tolist() == [5]
    assert state.timestep.dtype == torch.long


def test_valid_coordinates_are_centered_per_sample() -> None:
    stage3_input = _make_stage3_input()

    state = initialize_stage3_state(stage3_input, seed=22)
    weights = state.mask.atom_mask.to(state.coordinates.dtype).unsqueeze(-1)
    valid_center = (state.coordinates * weights).sum(dim=1) / weights.sum(dim=1)

    assert torch.allclose(valid_center, torch.zeros_like(valid_center), atol=1.0e-6)


def test_variable_size_batch_mask_and_padding_are_preserved() -> None:
    stage3_input = _make_stage3_input()

    state = initialize_stage3_state(stage3_input, seed=88)

    assert torch.equal(state.mask.atom_mask, stage3_input.mask.atom_mask)
    assert torch.equal(state.mask.edge_mask, stage3_input.mask.edge_mask)
    assert torch.equal(state.mask.descriptor_mask, stage3_input.mask.descriptor_mask)
    assert torch.equal(state.mask.pocket_mask, stage3_input.mask.pocket_mask)
    assert torch.count_nonzero(state.coordinates[~state.mask.atom_mask]) == 0


def test_initialization_does_not_modify_graph_topology_or_features() -> None:
    stage3_input = _make_stage3_input()
    original_atom_features = stage3_input.scaffold.atom_features.clone()
    original_atom_types = stage3_input.scaffold.atom_types.clone()
    original_edge_index = stage3_input.scaffold.edge_index.clone()
    original_bond_types = stage3_input.scaffold.bond_types.clone()
    original_edge_features = stage3_input.scaffold.edge_features.clone()
    original_coordinates = stage3_input.scaffold.coordinates.clone()

    state = initialize_stage3_state(stage3_input, seed=31)

    assert torch.equal(stage3_input.scaffold.atom_features, original_atom_features)
    assert torch.equal(stage3_input.scaffold.atom_types, original_atom_types)
    assert torch.equal(stage3_input.scaffold.edge_index, original_edge_index)
    assert torch.equal(stage3_input.scaffold.bond_types, original_bond_types)
    assert torch.equal(stage3_input.scaffold.edge_features, original_edge_features)
    assert torch.equal(stage3_input.scaffold.coordinates, original_coordinates)
    assert torch.equal(state.atom_features, original_atom_features)
    assert torch.equal(state.atom_types, original_atom_types)
    assert torch.equal(state.edge_index, original_edge_index)
    assert torch.equal(state.bond_types, original_bond_types)
    assert torch.equal(state.edge_features, original_edge_features)
    assert not torch.equal(state.coordinates, original_coordinates)


def test_descriptor_and_sample_order_are_preserved() -> None:
    stage3_input = _make_stage3_input()

    state = initialize_stage3_state(stage3_input, seed=13)

    assert torch.equal(state.descriptor, stage3_input.descriptor.values)
    assert state.metadata.sample_ids == stage3_input.metadata.sample_ids
    assert state.metadata.records == stage3_input.metadata.records


def test_initialization_copies_fixed_ring_coordinates_exactly() -> None:
    stage3_input = _make_stage3_input(batch_size=1)
    stage3_input.mask.diffusion_mask[0] = torch.tensor(
        [False, False, True, True]
    )

    state = initialize_stage3_state(stage3_input, seed=17)

    fixed_ring = ~stage3_input.mask.diffusion_mask & stage3_input.mask.atom_mask
    assert torch.equal(
        state.coordinates[fixed_ring],
        stage3_input.scaffold.coordinates[fixed_ring],
    )


def test_empty_ligand_sample_and_invalid_arguments_have_clear_errors() -> None:
    stage3_input = _make_stage3_input()
    stage3_input.mask.atom_mask[1] = False
    stage3_input.mask.edge_mask[1] = False
    stage3_input.mask.diffusion_mask[1] = False

    with pytest.raises(ValueError, match="at least one valid ligand atom"):
        initialize_stage3_state(stage3_input, seed=1)
    with pytest.raises(TypeError, match="seed must be an integer"):
        initialize_stage3_state(_make_stage3_input(), seed=True)
    with pytest.raises(ValueError, match="timestep must be non-negative"):
        initialize_stage3_state(_make_stage3_input(), seed=1, timestep=-1)
