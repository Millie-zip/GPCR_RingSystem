"""Tests for Stage 3 forward DDPM noising and coordinate-noise loss."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from gpcr_ringgen.core.types import BatchMask, BatchMetadata, Stage3DiffusionState
from gpcr_ringgen.models.diffusion import (
    DDPMConfig,
    DDPMNoiseSchedule,
    load_ddpm_config,
    project_zero_center_of_mass,
    q_sample,
    sample_timesteps,
    training_target,
)
from gpcr_ringgen.models.equivariant import (
    EquivariantDenoiserBackbone,
    EquivariantDenoiserConfig,
)
from gpcr_ringgen.training import NoisePredictionLoss, masked_coordinate_mse
from gpcr_ringgen.utils.seed import make_torch_generator


def _make_config(*, loss_reduction: str = "mean") -> DDPMConfig:
    return DDPMConfig(
        num_timesteps=20,
        beta_schedule="linear",
        beta_start=1.0e-3,
        beta_end=0.1,
        prediction_type="epsilon",
        loss_reduction=loss_reduction,
    )


def _make_state(
    *,
    dtype: torch.dtype = torch.float32,
    zero_coordinates: bool = False,
) -> Stage3DiffusionState:
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
        ],
        dtype=dtype,
    )
    if zero_coordinates:
        coordinates.zero_()
    state = Stage3DiffusionState(
        atom_features=torch.tensor(
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
            ],
            dtype=dtype,
        ),
        atom_types=torch.tensor(
            [[0, 1, 2, 3], [0, 1, 0, 0]],
            dtype=torch.long,
        ),
        edge_index=torch.tensor(
            [
                [[0, 1, 2, 3], [1, 2, 3, 0]],
                [[0, -1, -1, -1], [1, -1, -1, -1]],
            ],
            dtype=torch.long,
        ),
        bond_types=torch.tensor(
            [[0, 1, 2, 3], [0, 0, 0, 0]],
            dtype=torch.long,
        ),
        edge_features=torch.tensor(
            [
                [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0], [0.25, 0.75]],
                [[1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            ],
            dtype=dtype,
        ),
        coordinates=coordinates,
        descriptor=torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 0.0]],
            dtype=dtype,
        ),
        timestep=torch.zeros(2, dtype=torch.long),
        mask=BatchMask(
            atom_mask=torch.tensor(
                [[True, True, True, True], [True, True, False, False]]
            ),
            diffusion_mask=torch.tensor(
                [[True, True, True, True], [True, True, False, False]]
            ),
            edge_mask=torch.tensor(
                [[True, True, True, True], [True, False, False, False]]
            ),
            descriptor_mask=torch.tensor(
                [[True, True, True], [True, True, False]]
            ),
        ),
        metadata=BatchMetadata(
            sample_ids=("sample-0", "sample-1"),
            records=({"row": 0}, {"row": 1}),
        ),
    )
    state.validate()
    return state


def _make_denoiser(dtype: torch.dtype) -> EquivariantDenoiserBackbone:
    torch.manual_seed(123)
    model_config = EquivariantDenoiserConfig(
        atom_feature_dim=4,
        edge_feature_dim=2,
        hidden_dim=16,
        num_layers=2,
        timestep_embedding_dim=8,
        coordinate_scale=0.1,
        epsilon=1.0e-8,
        max_time_period=10_000.0,
        descriptor_dim=3,
    )
    return EquivariantDenoiserBackbone(model_config).to(dtype=dtype)


def test_forward_diffusion_leaves_ring_coordinates_unchanged() -> None:
    state = _make_state(dtype=torch.float64)
    state.mask.diffusion_mask = torch.tensor(
        [[False, False, True, True], [False, True, False, False]]
    )
    timestep = torch.tensor([5, 8], dtype=torch.long)
    explicit_noise = torch.randn_like(state.coordinates)

    output = q_sample(
        state,
        timestep,
        DDPMNoiseSchedule(_make_config()),
        noise=explicit_noise,
    )

    fixed_ring = state.mask.atom_mask & ~state.mask.diffusion_mask
    assert torch.equal(
        output.noisy_state.coordinates[fixed_ring],
        state.coordinates[fixed_ring],
    )
    assert torch.equal(
        output.noise[fixed_ring],
        torch.zeros_like(output.noise[fixed_ring]),
    )


def _valid_centers(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.unsqueeze(-1).to(values.dtype)
    return (values * weights).sum(dim=1) / weights.sum(dim=1)


def test_minimum_timestep_matches_standard_ddpm_formula() -> None:
    state = _make_state(dtype=torch.float64)
    schedule = DDPMNoiseSchedule(_make_config())
    timestep = torch.zeros(state.batch_size, dtype=torch.long)
    explicit_noise = torch.arange(
        state.coordinates.numel(), dtype=state.coordinates.dtype
    ).reshape_as(state.coordinates)

    result = q_sample(state, timestep, schedule, noise=explicit_noise)

    centered_clean = project_zero_center_of_mass(
        state.coordinates, state.mask.atom_mask
    )
    centered_noise = project_zero_center_of_mass(
        explicit_noise, state.mask.atom_mask
    )
    signal, noise_scale = schedule.coefficients(timestep, state.coordinates)
    expected = signal * centered_clean + noise_scale * centered_noise
    expected = expected * state.mask.atom_mask.unsqueeze(-1)
    assert schedule.value(0) == pytest.approx(_make_config().beta_start)
    assert torch.allclose(result.noisy_state.coordinates, expected)
    assert torch.allclose(result.noise, centered_noise)


def test_noise_influence_increases_with_timestep() -> None:
    state = _make_state(dtype=torch.float64, zero_coordinates=True)
    schedule = DDPMNoiseSchedule(_make_config())
    explicit_noise = torch.tensor(
        [
            [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 1.0, 0.0]],
            [[-1.0, 0.5, 0.0], [1.0, -0.5, 0.0], [9.0, 9.0, 9.0], [9.0, 9.0, 9.0]],
        ],
        dtype=torch.float64,
    )
    low_timestep = torch.zeros(2, dtype=torch.long)
    high_timestep = torch.full((2,), schedule.num_timesteps - 1, dtype=torch.long)

    low = q_sample(state, low_timestep, schedule, noise=explicit_noise)
    high = q_sample(state, high_timestep, schedule, noise=explicit_noise)

    assert torch.linalg.vector_norm(high.noisy_state.coordinates) > torch.linalg.vector_norm(
        low.noisy_state.coordinates
    )
    low_signal, low_noise = schedule.coefficients(low_timestep, state.coordinates)
    high_signal, high_noise = schedule.coefficients(high_timestep, state.coordinates)
    assert torch.all(high_noise > low_noise)
    assert torch.all(high_signal < low_signal)


def test_q_sample_batch_shape_mask_dtype_and_cpu() -> None:
    dtype = torch.float64
    state = _make_state(dtype=dtype)
    schedule = DDPMNoiseSchedule(_make_config())
    timestep = torch.tensor([3, 17], dtype=torch.long)

    result = q_sample(
        state,
        timestep,
        schedule,
        generator=make_torch_generator(42),
    )

    assert result.noisy_state.coordinates.shape == (2, 4, 3)
    assert result.noise.shape == (2, 4, 3)
    assert result.noisy_state.coordinates.dtype == dtype
    assert result.noisy_state.coordinates.device.type == "cpu"
    assert result.noisy_state.timestep.tolist() == [3, 17]
    assert state.mask.atom_mask.sum(dim=1).tolist() == [4, 2]
    assert torch.count_nonzero(result.noisy_state.coordinates[1, 2:]) == 0
    assert torch.count_nonzero(result.noise[1, 2:]) == 0
    assert torch.allclose(
        _valid_centers(result.noisy_state.coordinates, state.mask.atom_mask),
        torch.zeros((2, 3), dtype=dtype),
        atol=1.0e-12,
    )
    assert torch.allclose(
        _valid_centers(result.noise, state.mask.atom_mask),
        torch.zeros((2, 3), dtype=dtype),
        atol=1.0e-12,
    )


def test_seeded_timestep_and_noise_sampling_are_reproducible() -> None:
    state = _make_state()
    schedule = DDPMNoiseSchedule(_make_config())
    first_timestep = sample_timesteps(
        state.batch_size,
        schedule,
        device=state.coordinates.device,
        generator=make_torch_generator(7),
    )
    second_timestep = sample_timesteps(
        state.batch_size,
        schedule,
        device=state.coordinates.device,
        generator=make_torch_generator(7),
    )
    first = q_sample(
        state,
        first_timestep,
        schedule,
        generator=make_torch_generator(99),
    )
    second = q_sample(
        state,
        second_timestep,
        schedule,
        generator=make_torch_generator(99),
    )

    assert torch.equal(first_timestep, second_timestep)
    assert torch.equal(first.noise, second.noise)
    assert torch.equal(first.noisy_state.coordinates, second.noisy_state.coordinates)
    assert bool(((first_timestep >= 0) & (first_timestep < 20)).all())


def test_training_target_is_centered_epsilon() -> None:
    state = _make_state()
    schedule = DDPMNoiseSchedule(_make_config())
    result = q_sample(
        state,
        torch.tensor([1, 9], dtype=torch.long),
        schedule,
        generator=make_torch_generator(5),
    )

    target = training_target(result, prediction_type="epsilon")

    assert target is result.noise
    assert torch.count_nonzero(target[~state.mask.atom_mask]) == 0


def test_masked_loss_is_finite_scalar_and_ignores_padding() -> None:
    state = _make_state()
    target = torch.zeros_like(state.coordinates)
    prediction = torch.ones_like(state.coordinates)
    prediction[1, 2:] = 1.0e6

    mean_loss = masked_coordinate_mse(
        prediction,
        target,
        state.mask.atom_mask,
        reduction="mean",
    )
    sum_loss = NoisePredictionLoss(reduction="sum")(
        prediction,
        target,
        state.mask.atom_mask,
    )

    assert mean_loss.ndim == 0
    assert torch.isfinite(mean_loss)
    assert mean_loss.item() == pytest.approx(1.0)
    assert sum_loss.item() == pytest.approx(18.0)


def test_denoiser_loss_backward_has_finite_gradients() -> None:
    state = _make_state()
    schedule = DDPMNoiseSchedule(_make_config())
    forward = q_sample(
        state,
        torch.tensor([4, 12], dtype=torch.long),
        schedule,
        generator=make_torch_generator(101),
    )
    model = _make_denoiser(state.coordinates.dtype)
    criterion = NoisePredictionLoss(_make_config().loss_reduction)

    prediction = model(forward.noisy_state).coordinate_noise
    loss = criterion(prediction, training_target(forward), state.mask.atom_mask)
    loss.backward()
    gradients = [
        parameter.grad for parameter in model.parameters() if parameter.grad is not None
    ]

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_invalid_timestep_and_empty_mask_raise_clear_errors() -> None:
    state = _make_state()
    schedule = DDPMNoiseSchedule(_make_config())
    with pytest.raises(ValueError, match="timestep values"):
        q_sample(state, torch.tensor([0, 20]), schedule)

    empty_mask = torch.zeros_like(state.mask.atom_mask)
    with pytest.raises(ValueError, match="at least one valid atom"):
        project_zero_center_of_mass(state.coordinates, empty_mask)


def test_repository_ddpm_config_loads() -> None:
    config_path = Path(__file__).parents[2] / "configs" / "diffusion" / "ddpm.yaml"

    config = load_ddpm_config(config_path)
    schedule = DDPMNoiseSchedule(config)

    assert config.num_timesteps == 1000
    assert config.beta_schedule == "linear"
    assert config.prediction_type == "epsilon"
    assert config.loss_reduction == "mean"
    assert schedule.alpha_bars.shape == (1000,)
    assert torch.all(schedule.alpha_bars[1:] < schedule.alpha_bars[:-1])
