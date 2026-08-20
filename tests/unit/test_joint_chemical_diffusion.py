"""Tests for fixed-topology atom/bond/coordinate joint diffusion."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from gpcr_ringgen.core.types import (
    BatchMask,
    BatchMetadata,
    DescriptorBatch,
    MolecularGraph,
    PocketBatch,
    Stage3DiffusionState,
    Stage3Input,
)
from gpcr_ringgen.models.diffusion import (
    DDPMConfig,
    DDPMNoiseSchedule,
    categorical_reverse_step,
    q_sample_joint,
)
from gpcr_ringgen.models.equivariant import (
    EquivariantDenoiserBackbone,
    EquivariantDenoiserConfig,
)
from gpcr_ringgen.sampling import JointDDPMStage3Sampler
from gpcr_ringgen.training import JointDiffusionLoss
from gpcr_ringgen.utils.seed import make_torch_generator


def _schedule(*, num_timesteps: int = 6) -> DDPMNoiseSchedule:
    return DDPMNoiseSchedule(
        DDPMConfig(
            num_timesteps=num_timesteps,
            beta_schedule="linear",
            beta_start=1.0e-3,
            beta_end=0.1,
            prediction_type="epsilon",
            loss_reduction="mean",
            mode="joint_chemical",
            num_atom_types=5,
            num_bond_types=4,
        )
    )


def _state(*, dtype: torch.dtype = torch.float32) -> Stage3DiffusionState:
    state = Stage3DiffusionState(
        atom_features=torch.tensor(
            [
                [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
                [[1.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            ],
            dtype=dtype,
        ),
        edge_index=torch.tensor(
            [
                [[0, 1, 2, 3], [1, 2, 3, 0]],
                [[0, -1, -1, -1], [1, -1, -1, -1]],
            ],
            dtype=torch.long,
        ),
        edge_features=torch.tensor(
            [
                [[1.0], [1.0], [1.0], [1.0]],
                [[1.0], [0.0], [0.0], [0.0]],
            ],
            dtype=dtype,
        ),
        coordinates=torch.tensor(
            [
                [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]],
                [[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ],
            dtype=dtype,
        ),
        descriptor=torch.tensor([[1.0, 2.0], [3.0, 0.0]], dtype=dtype),
        timestep=torch.zeros(2, dtype=torch.long),
        mask=BatchMask(
            atom_mask=torch.tensor(
                [[True, True, True, True], [True, True, False, False]]
            ),
            edge_mask=torch.tensor(
                [[True, True, True, True], [True, False, False, False]]
            ),
            descriptor_mask=torch.tensor([[True, True], [True, False]]),
        ),
        metadata=BatchMetadata(
            sample_ids=("joint-0", "joint-1"),
            records=({}, {}),
        ),
        atom_types=torch.tensor([[0, 1, 2, 3], [1, 2, 0, 0]]),
        bond_types=torch.tensor([[0, 1, 2, 3], [1, 0, 0, 0]]),
    )
    state.validate()
    return state


def _model(*, dtype: torch.dtype = torch.float32) -> EquivariantDenoiserBackbone:
    torch.manual_seed(123)
    return EquivariantDenoiserBackbone(
        EquivariantDenoiserConfig(
            atom_feature_dim=2,
            edge_feature_dim=1,
            hidden_dim=16,
            num_layers=2,
            timestep_embedding_dim=8,
            coordinate_scale=0.1,
            epsilon=1.0e-8,
            max_time_period=10_000.0,
            descriptor_dim=2,
            num_atom_types=5,
            num_bond_types=4,
        )
    ).to(dtype=dtype)


def _stage3_input(*, dtype: torch.dtype = torch.float32) -> Stage3Input:
    state = _state(dtype=dtype)
    return Stage3Input(
        scaffold=MolecularGraph(
            atom_features=state.atom_features,
            edge_index=state.edge_index,
            edge_features=state.edge_features,
            atom_types=state.atom_types,
            bond_types=state.bond_types,
        ),
        descriptor=DescriptorBatch(state.descriptor),
        pocket=PocketBatch(
            coordinates=torch.tensor(
                [
                    [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                    [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                ],
                dtype=dtype,
            ),
            features=torch.ones(2, 2, 1, dtype=dtype),
        ),
        mask=BatchMask(
            atom_mask=state.mask.atom_mask,
            edge_mask=state.mask.edge_mask,
            descriptor_mask=state.mask.descriptor_mask,
            pocket_mask=torch.ones(2, 2, dtype=torch.bool),
        ),
        metadata=state.metadata,
    )


def _rotation(dtype: torch.dtype) -> torch.Tensor:
    matrix = torch.randn(3, 3, dtype=dtype, generator=make_torch_generator(77))
    rotation, _ = torch.linalg.qr(matrix)
    if torch.linalg.det(rotation) < 0:
        rotation[:, 0] = -rotation[:, 0]
    return rotation


def test_joint_forward_noising_preserves_topology_and_padding() -> None:
    state = _state()
    schedule = _schedule()
    result = q_sample_joint(
        state,
        torch.tensor([1, 5]),
        schedule,
        generator=make_torch_generator(9),
    )

    assert torch.equal(result.noisy_state.edge_index, state.edge_index)
    assert torch.equal(result.noisy_state.mask.atom_mask, state.mask.atom_mask)
    assert torch.equal(result.noisy_state.mask.edge_mask, state.mask.edge_mask)
    assert result.noisy_state.atom_types.shape == (2, 4)
    assert result.noisy_state.bond_types.shape == (2, 4)
    assert torch.count_nonzero(result.noisy_state.atom_types[1, 2:]) == 0
    assert torch.count_nonzero(result.noisy_state.bond_types[1, 1:]) == 0
    assert result.coordinate_noise.shape == (2, 4, 3)


def test_joint_forward_noising_is_seed_reproducible() -> None:
    state = _state()
    timestep = torch.tensor([3, 5])
    first = q_sample_joint(
        state,
        timestep,
        _schedule(),
        generator=make_torch_generator(42),
    )
    second = q_sample_joint(
        state,
        timestep,
        _schedule(),
        generator=make_torch_generator(42),
    )

    assert torch.equal(first.noisy_state.coordinates, second.noisy_state.coordinates)
    assert torch.equal(first.noisy_state.atom_types, second.noisy_state.atom_types)
    assert torch.equal(first.noisy_state.bond_types, second.noisy_state.bond_types)


def test_joint_denoiser_shapes_and_loss_backward() -> None:
    state = _state()
    schedule = _schedule()
    target = q_sample_joint(
        state,
        torch.tensor([2, 4]),
        schedule,
        generator=make_torch_generator(12),
    )
    model = _model()
    prediction = model(target.noisy_state)
    losses = JointDiffusionLoss(
        coordinate_weight=1.0,
        atom_weight=0.5,
        bond_weight=0.5,
    )(prediction, target)
    losses.total.backward()

    assert prediction.coordinate_noise.shape == (2, 4, 3)
    assert prediction.atom_type_logits is not None
    assert prediction.bond_type_logits is not None
    assert prediction.atom_type_logits.shape == (2, 4, 5)
    assert prediction.bond_type_logits.shape == (2, 4, 4)
    assert losses.total.ndim == 0
    assert torch.isfinite(losses.total)
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_joint_predictions_preserve_rotation_rules() -> None:
    dtype = torch.float64
    state = _state(dtype=dtype)
    rotation = _rotation(dtype)
    rotated = replace(
        state,
        coordinates=state.coordinates @ rotation.transpose(0, 1),
    )
    model = _model(dtype=dtype).eval()

    original = model(state)
    transformed = model(rotated)

    assert torch.allclose(
        transformed.coordinate_noise,
        original.coordinate_noise @ rotation.transpose(0, 1),
        atol=1.0e-9,
        rtol=1.0e-7,
    )
    assert torch.allclose(transformed.atom_type_logits, original.atom_type_logits)
    assert torch.allclose(transformed.bond_type_logits, original.bond_type_logits)


def test_bond_logits_do_not_depend_on_stored_edge_orientation() -> None:
    state = _state(dtype=torch.float64)
    reversed_edges = state.edge_index.clone()
    reversed_edges[:, 0, :], reversed_edges[:, 1, :] = (
        state.edge_index[:, 1, :],
        state.edge_index[:, 0, :],
    )
    reversed_state = replace(state, edge_index=reversed_edges)
    model = _model(dtype=torch.float64).eval()

    original = model(state)
    reversed_output = model(reversed_state)

    assert torch.allclose(reversed_output.coordinate_noise, original.coordinate_noise)
    assert torch.allclose(reversed_output.atom_type_logits, original.atom_type_logits)
    assert torch.allclose(reversed_output.bond_type_logits, original.bond_type_logits)


def test_categorical_reverse_step_keeps_fixed_validity_masks() -> None:
    state = _state()
    logits = torch.randn(2, 4, 5, generator=make_torch_generator(8))
    sampled = categorical_reverse_step(
        state.atom_types,
        logits,
        torch.tensor([0, 4]),
        _schedule(),
        state.mask.atom_mask,
        5,
        generator=make_torch_generator(4),
    )

    assert sampled.shape == state.atom_types.shape
    assert torch.count_nonzero(sampled[~state.mask.atom_mask]) == 0
    assert bool((sampled[state.mask.atom_mask] < 5).all())


def test_joint_sampler_is_reproducible_and_freezes_topology() -> None:
    stage3_input = _stage3_input()
    sampler = JointDDPMStage3Sampler(_schedule(num_timesteps=3))
    model = _model().eval()

    first = sampler.sample(stage3_input, model, seed=21)
    second = sampler.sample(stage3_input, model, seed=21)

    assert torch.equal(first.final_state.edge_index, stage3_input.scaffold.edge_index)
    assert torch.equal(first.final_state.mask.edge_mask, stage3_input.mask.edge_mask)
    assert torch.equal(first.coordinates, second.coordinates)
    assert torch.equal(first.final_state.atom_types, second.final_state.atom_types)
    assert torch.equal(first.final_state.bond_types, second.final_state.bond_types)
    assert first.final_state.atom_types is not None
    assert first.final_state.bond_types is not None
    assert bool((first.final_state.atom_types[first.final_state.mask.atom_mask] < 5).all())
    assert bool((first.final_state.bond_types[first.final_state.mask.edge_mask] < 4).all())


def test_joint_diffusion_rejects_duplicate_undirected_topology_edges() -> None:
    state = _state()
    duplicated_index = state.edge_index.clone()
    duplicated_index[0, :, 1] = torch.tensor([1, 0])
    invalid = replace(state, edge_index=duplicated_index)

    with pytest.raises(ValueError, match="store each undirected edge once"):
        q_sample_joint(
            invalid,
            torch.tensor([1, 1]),
            _schedule(),
            generator=make_torch_generator(1),
        )
