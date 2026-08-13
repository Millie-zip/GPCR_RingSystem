"""Tests for the pocket-independent E(n)-equivariant denoiser backbone."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch

from gpcr_ringgen.core.types import (
    BatchMask,
    BatchMetadata,
    Stage3DiffusionState,
)
from gpcr_ringgen.models.equivariant import (
    EquivariantDenoiserBackbone,
    EquivariantDenoiserConfig,
    load_equivariant_denoiser_config,
)


def _make_state(
    *,
    batch_size: int = 2,
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cpu"),
) -> Stage3DiffusionState:
    if batch_size not in (1, 2):
        raise ValueError("test helper supports batch sizes 1 and 2")

    atom_features = torch.tensor(
        [
            [
                [1.0, 0.0, 0.5, -0.5],
                [0.0, 1.0, -0.5, 0.5],
                [1.0, 1.0, 0.25, -0.25],
                [0.5, -0.5, 1.0, 0.0],
            ],
            [
                [0.2, 0.8, 0.1, -0.1],
                [0.8, 0.2, -0.1, 0.1],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
        ][:batch_size],
        dtype=dtype,
        device=device,
    )
    coordinates = torch.tensor(
        [
            [
                [-1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.5],
                [0.0, -1.0, -0.5],
            ],
            [
                [-0.5, 0.25, 0.75],
                [0.5, -0.25, -0.75],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
        ][:batch_size],
        dtype=dtype,
        device=device,
    )
    edge_index = torch.tensor(
        [
            [[0, 1, 2, 3], [1, 2, 3, 0]],
            [[0, -1, -1, -1], [1, -1, -1, -1]],
        ][:batch_size],
        dtype=torch.long,
        device=device,
    )
    edge_features = torch.tensor(
        [
            [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0], [0.25, 0.75]],
            [[1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        ][:batch_size],
        dtype=dtype,
        device=device,
    )
    atom_mask = torch.tensor(
        [[True, True, True, True], [True, True, False, False]][:batch_size],
        device=device,
    )
    edge_mask = torch.tensor(
        [[True, True, True, True], [True, False, False, False]][:batch_size],
        device=device,
    )
    descriptor = torch.tensor(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 0.0]][:batch_size],
        dtype=dtype,
        device=device,
    )
    descriptor_mask = torch.tensor(
        [[True, True, True], [True, True, False]][:batch_size],
        device=device,
    )
    state = Stage3DiffusionState(
        atom_features=atom_features,
        edge_index=edge_index,
        edge_features=edge_features,
        coordinates=coordinates,
        descriptor=descriptor,
        timestep=torch.tensor([2, 7][:batch_size], dtype=torch.long, device=device),
        mask=BatchMask(
            atom_mask=atom_mask,
            edge_mask=edge_mask,
            descriptor_mask=descriptor_mask,
        ),
        metadata=BatchMetadata(
            sample_ids=tuple(f"sample-{index}" for index in range(batch_size)),
            records=tuple({"row": index} for index in range(batch_size)),
        ),
    )
    state.validate()
    return state


def _make_model(
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cpu"),
    descriptor_dim: int | None = 3,
) -> EquivariantDenoiserBackbone:
    torch.manual_seed(1234)
    config = EquivariantDenoiserConfig(
        atom_feature_dim=4,
        edge_feature_dim=2,
        hidden_dim=16,
        num_layers=2,
        timestep_embedding_dim=8,
        coordinate_scale=0.1,
        epsilon=1.0e-8,
        max_time_period=10_000.0,
        descriptor_dim=descriptor_dim,
    )
    return EquivariantDenoiserBackbone(config).to(device=device, dtype=dtype)


def _random_rotation(dtype: torch.dtype) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(99)
    matrix = torch.randn(3, 3, dtype=dtype, generator=generator)
    rotation, _ = torch.linalg.qr(matrix)
    if torch.linalg.det(rotation) < 0:
        rotation[:, 0] = -rotation[:, 0]
    return rotation


def test_forward_shape_and_masked_output() -> None:
    state = _make_state()
    model = _make_model().eval()

    output = model(state, state.timestep)

    assert output.coordinate_noise.shape == state.coordinates.shape
    assert output.coordinate_noise.dtype == state.coordinates.dtype
    assert output.coordinate_noise.device == state.coordinates.device
    assert torch.count_nonzero(output.coordinate_noise[~state.mask.atom_mask]) == 0
    assert torch.isfinite(output.coordinate_noise).all()


def test_single_and_batched_forward() -> None:
    model = _make_model().eval()

    single_output = model(_make_state(batch_size=1), timestep=3)
    batch_output = model(_make_state(batch_size=2), timestep=3)

    assert single_output.coordinate_noise.shape == (1, 4, 3)
    assert batch_output.coordinate_noise.shape == (2, 4, 3)


def test_variable_atom_count_batch_uses_masks() -> None:
    state = _make_state(batch_size=2)
    model = _make_model().eval()

    output = model(state)
    weights = state.mask.atom_mask.to(state.coordinates.dtype).unsqueeze(-1)
    center = (output.coordinate_noise * weights).sum(dim=1) / weights.sum(dim=1)

    assert state.mask.atom_mask.sum(dim=1).tolist() == [4, 2]
    assert torch.allclose(center, torch.zeros_like(center), atol=1.0e-6)
    assert torch.count_nonzero(output.coordinate_noise[1, 2:]) == 0


def test_backward_reaches_coordinates_and_parameters() -> None:
    state = _make_state()
    state.coordinates = state.coordinates.clone().requires_grad_(True)
    model = _make_model()

    loss = model(state).coordinate_noise.square().sum()
    loss.backward()

    assert state.coordinates.grad is not None
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_all_computed_gradients_are_finite() -> None:
    state = _make_state()
    state.coordinates = state.coordinates.clone().requires_grad_(True)
    model = _make_model()

    model(state).coordinate_noise.square().mean().backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]

    assert gradients
    assert state.coordinates.grad is not None
    assert torch.isfinite(state.coordinates.grad).all()
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_random_rotation_equivariance() -> None:
    dtype = torch.float64
    state = _make_state(dtype=dtype)
    model = _make_model(dtype=dtype).eval()
    rotation = _random_rotation(dtype)
    rotated_state = replace(
        state,
        coordinates=state.coordinates @ rotation.transpose(0, 1),
    )

    prediction = model(state).coordinate_noise
    rotated_prediction = model(rotated_state).coordinate_noise
    expected = prediction @ rotation.transpose(0, 1)
    max_error = torch.max(torch.abs(rotated_prediction - expected)).item()

    print(f"rotation_max_error={max_error:.3e}")
    assert torch.allclose(rotated_prediction, expected, atol=1.0e-9, rtol=1.0e-7)


def test_translation_consistency() -> None:
    dtype = torch.float64
    state = _make_state(dtype=dtype)
    model = _make_model(dtype=dtype).eval()
    translation = torch.tensor([7.0, -4.0, 2.5], dtype=dtype)
    translated_state = replace(
        state,
        coordinates=state.coordinates + translation,
    )

    prediction = model(state).coordinate_noise
    translated_prediction = model(translated_state).coordinate_noise
    max_error = torch.max(torch.abs(translated_prediction - prediction)).item()

    print(f"translation_max_error={max_error:.3e}")
    assert torch.allclose(translated_prediction, prediction, atol=1.0e-9, rtol=1.0e-7)


def test_float64_cpu_smoke_and_optional_descriptor_path() -> None:
    dtype = torch.float64
    state = _make_state(dtype=dtype)
    conditioned_model = _make_model(dtype=dtype, descriptor_dim=3).eval()
    unconditioned_model = _make_model(dtype=dtype, descriptor_dim=None).eval()

    conditioned = conditioned_model(state)
    unconditioned = unconditioned_model(state)

    assert conditioned.coordinate_noise.dtype == dtype
    assert conditioned.coordinate_noise.device.type == "cpu"
    assert unconditioned.coordinate_noise.shape == state.coordinates.shape


def test_repository_denoiser_config_loads() -> None:
    config_path = (
        Path(__file__).parents[2]
        / "configs"
        / "model"
        / "equivariant_denoiser.yaml"
    )

    config = load_equivariant_denoiser_config(config_path)

    assert config.hidden_dim == 64
    assert config.num_layers == 3
    assert config.descriptor_dim is None
