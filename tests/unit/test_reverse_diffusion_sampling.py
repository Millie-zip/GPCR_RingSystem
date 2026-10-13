"""Tests for Stage 3 DDPM reverse diffusion and sampling outputs."""

from __future__ import annotations

import pytest
import torch

from gpcr_ringgen.core.types import (
    BatchMask,
    BatchMetadata,
    DescriptorBatch,
    MolecularGraph,
    Stage2Output,
    Stage3Input,
)
from gpcr_ringgen.data.transforms import stage2_output_to_stage3_input
from gpcr_ringgen.models.diffusion import DDPMConfig, DDPMNoiseSchedule
from gpcr_ringgen.models.equivariant import (
    EquivariantDenoiserBackbone,
    EquivariantDenoiserConfig,
    FieldConditionedEquivariantDenoiser,
    FieldConditioningConfig,
)
from gpcr_ringgen.pocket import (
    GeometryFieldConfig,
    adapt_pocket_atoms,
    build_geometry_field,
)
from gpcr_ringgen.sampling import (
    DDPMStage3Sampler,
    ddpm_reverse_step,
    restore_global_coordinates,
)
from gpcr_ringgen.utils.seed import make_torch_generator


def _processed_pocket(dtype: torch.dtype = torch.float32):
    coordinates = torch.tensor(
        [
            [[9.0, -5.0, 2.0], [13.0, -5.0, 2.0], [11.0, -2.0, 2.0]],
            [[-4.0, 6.0, 1.0], [-2.0, 6.0, -1.0], [-3.0, 6.0, 0.0]],
        ],
        dtype=dtype,
    )
    pocket_mask = torch.tensor(
        [[True, True, True], [True, True, False]],
    )

    def rows(prefix: str) -> list[list[str]]:
        return [
            [
                f"{prefix}{point}" if pocket_mask[batch, point] else ""
                for point in range(coordinates.shape[1])
            ]
            for batch in range(coordinates.shape[0])
        ]

    return adapt_pocket_atoms(
        coordinates=coordinates,
        features=torch.ones(2, 3, 2, dtype=dtype),
        atom_types=rows("atom"),
        elements=rows("C"),
        residue_names=rows("RES"),
        residue_ids=rows("id"),
        sample_ids=["sample-0", "sample-1"],
        pocket_mask=pocket_mask,
    )


def _field(processed_pocket):
    return build_geometry_field(
        processed_pocket,
        GeometryFieldConfig(
            sigma=1.3,
            cutoff=20.0,
            epsilon=1.0e-8,
            occupancy_enabled=True,
            nearest_distance_enabled=True,
        ),
    )


def _stage3_input(
    *,
    dtype: torch.dtype = torch.float32,
    include_field: bool = False,
) -> tuple[Stage3Input, object]:
    processed = _processed_pocket(dtype)
    atom_mask = torch.tensor(
        [[True, True, True, True], [True, True, False, False]],
    )
    edge_mask = torch.tensor(
        [[True, True, True, True], [True, False, False, False]],
    )
    stage2 = Stage2Output(
        scaffold=MolecularGraph(
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
            coordinates=torch.zeros(2, 4, 3, dtype=dtype),
        ),
        mask=BatchMask(
            atom_mask=atom_mask,
            edge_mask=edge_mask,
            diffusion_mask=atom_mask.clone(),
        ),
        metadata=BatchMetadata(
            sample_ids=("sample-0", "sample-1"),
            records=({"row": 0}, {"row": 1}),
        ),
    )
    pocket_field = _field(processed)
    stage3_input = stage2_output_to_stage3_input(
        stage2,
        descriptor=DescriptorBatch(
            torch.tensor(
                [[1.0, 2.0, 3.0], [4.0, 5.0, 0.0]],
                dtype=dtype,
            )
        ),
        pocket=processed.pocket,
        descriptor_mask=torch.tensor(
            [[True, True, True], [True, True, False]],
        ),
        pocket_mask=processed.pocket_mask,
        pocket_field=pocket_field if include_field else None,
    )
    return stage3_input, processed


def _schedule(num_timesteps: int = 4) -> DDPMNoiseSchedule:
    return DDPMNoiseSchedule(
        DDPMConfig(
            num_timesteps=num_timesteps,
            beta_schedule="linear",
            beta_start=1.0e-3,
            beta_end=0.02,
            prediction_type="epsilon",
            loss_reduction="mean",
        )
    )


def _backbone(dtype: torch.dtype = torch.float32) -> EquivariantDenoiserBackbone:
    torch.manual_seed(1357)
    return EquivariantDenoiserBackbone(
        EquivariantDenoiserConfig(
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
    ).to(dtype=dtype)


def _conditioned_model(
    mode: str,
    dtype: torch.dtype = torch.float32,
) -> FieldConditionedEquivariantDenoiser:
    return FieldConditionedEquivariantDenoiser(
        _backbone(dtype),
        FieldConditioningConfig(
            mode=mode,
            channels=("occupancy", "nearest_distance"),
        ),
    )


def _valid_centers(output) -> torch.Tensor:
    mask = output.final_state.mask.atom_mask
    weights = mask.unsqueeze(-1).to(output.coordinates.dtype)
    return (output.coordinates * weights).sum(dim=1) / weights.sum(dim=1)


def test_toy_reverse_sampling_runs_with_batch_mask_and_descriptor() -> None:
    stage3_input, _ = _stage3_input()
    model = _conditioned_model("none")
    sampler = DDPMStage3Sampler(_schedule())

    output = sampler.sample(stage3_input, model, seed=41)

    assert output.coordinates.shape == (2, 4, 3)
    assert output.coordinates.dtype == torch.float32
    assert output.coordinates.device.type == "cpu"
    assert output.coordinate_frame == "local"
    assert output.trajectory is None
    assert output.final_state.timestep.tolist() == [0, 0]
    assert torch.equal(output.graph.edge_index, stage3_input.scaffold.edge_index)
    assert torch.equal(output.graph.atom_features, stage3_input.scaffold.atom_features)
    assert torch.equal(output.final_state.descriptor, stage3_input.descriptor.values)
    assert output.metadata.sample_ids == stage3_input.metadata.sample_ids
    assert torch.count_nonzero(output.coordinates[1, 2:]) == 0
    assert torch.isfinite(output.coordinates).all()
    assert torch.allclose(
        _valid_centers(output),
        torch.zeros((2, 3)),
        atol=1.0e-6,
    )


def test_every_reverse_step_keeps_ring_coordinates_fixed() -> None:
    stage3_input, _ = _stage3_input()
    stage3_input.mask.diffusion_mask = torch.tensor(
        [[False, False, True, True], [False, True, False, False]]
    )
    stage3_input.scaffold.coordinates[0, 0] = torch.tensor([1.0, 2.0, 3.0])
    stage3_input.scaffold.coordinates[0, 1] = torch.tensor([-1.0, 0.5, 2.0])
    stage3_input.scaffold.coordinates[1, 0] = torch.tensor([4.0, -2.0, 1.0])

    output = DDPMStage3Sampler(_schedule(num_timesteps=3)).sample(
        stage3_input,
        _conditioned_model("none").eval(),
        seed=29,
        return_trajectory=True,
    )

    assert output.trajectory is not None
    fixed_ring = stage3_input.mask.atom_mask & ~stage3_input.mask.diffusion_mask
    expected = stage3_input.scaffold.coordinates[fixed_ring]
    for coordinates in output.trajectory:
        assert torch.equal(coordinates[fixed_ring], expected)


def test_fixed_seed_sampling_is_reproducible_and_local_rng_only() -> None:
    stage3_input, _ = _stage3_input()
    model = _conditioned_model("none").eval()
    sampler = DDPMStage3Sampler(_schedule())
    torch.manual_seed(999)
    expected_next_value = torch.rand(1)
    torch.manual_seed(999)

    first = sampler.sample(stage3_input, model, seed=77)
    second = sampler.sample(stage3_input, model, seed=77)
    actual_next_value = torch.rand(1)

    assert torch.equal(first.coordinates, second.coordinates)
    assert torch.equal(actual_next_value, expected_next_value)


def test_different_seeds_change_sampled_coordinates() -> None:
    stage3_input, _ = _stage3_input()
    model = _conditioned_model("none").eval()
    sampler = DDPMStage3Sampler(_schedule())

    first = sampler.sample(stage3_input, model, seed=11)
    second = sampler.sample(stage3_input, model, seed=12)

    assert not torch.equal(first.coordinates, second.coordinates)


def test_no_field_and_field_modes_both_sample() -> None:
    no_field_input, _ = _stage3_input()
    field_input, _ = _stage3_input(include_field=True)
    sampler = DDPMStage3Sampler(_schedule(num_timesteps=3))

    no_field_output = sampler.sample(
        no_field_input,
        _conditioned_model("none").eval(),
        seed=5,
    )
    field_output = sampler.sample(
        field_input,
        _conditioned_model("additive").eval(),
        seed=5,
    )

    assert no_field_output.coordinates.shape == field_output.coordinates.shape
    assert torch.isfinite(no_field_output.coordinates).all()
    assert torch.isfinite(field_output.coordinates).all()
    assert not torch.equal(no_field_output.coordinates, field_output.coordinates)


def test_trajectory_switch_and_model_mode_are_handled_safely() -> None:
    stage3_input, _ = _stage3_input()
    sampler = DDPMStage3Sampler(_schedule(num_timesteps=3))
    model = _conditioned_model("none")
    model.train()
    forward_modes: list[bool] = []
    hook = model.register_forward_pre_hook(
        lambda module, _inputs: forward_modes.append(module.training)
    )
    try:
        without_trajectory = sampler.sample(
            stage3_input,
            model,
            seed=8,
            return_trajectory=False,
        )
        with_trajectory = sampler.sample(
            stage3_input,
            model,
            seed=8,
            return_trajectory=True,
        )
    finally:
        hook.remove()

    assert without_trajectory.trajectory is None
    assert with_trajectory.trajectory is not None
    assert len(with_trajectory.trajectory) == 4
    assert torch.equal(with_trajectory.trajectory[-1], with_trajectory.coordinates)
    assert all(not coordinates.requires_grad for coordinates in with_trajectory.trajectory)
    assert forward_modes and not any(forward_modes)
    assert model.training


def test_reverse_step_t_zero_is_deterministic_and_zero_variance() -> None:
    stage3_input, _ = _stage3_input(dtype=torch.float64)
    from gpcr_ringgen.models.diffusion import initialize_stage3_state

    state = initialize_stage3_state(stage3_input, seed=3, timestep=0)
    predicted_noise = torch.zeros_like(state.coordinates)
    schedule = _schedule()

    first = ddpm_reverse_step(
        state,
        predicted_noise,
        schedule,
        generator=make_torch_generator(1),
    )
    second = ddpm_reverse_step(
        state,
        predicted_noise,
        schedule,
        generator=make_torch_generator(2),
    )

    assert torch.equal(first.previous_state.coordinates, first.posterior_mean)
    assert torch.equal(first.previous_state.coordinates, second.previous_state.coordinates)
    assert torch.count_nonzero(first.posterior_variance) == 0
    assert first.previous_state.timestep.tolist() == [0, 0]


def test_reverse_step_matches_ddpm_posterior_coefficients() -> None:
    stage3_input, _ = _stage3_input(dtype=torch.float64)
    from gpcr_ringgen.models.diffusion import initialize_stage3_state

    state = initialize_stage3_state(stage3_input, seed=13, timestep=2)
    predicted_noise = torch.zeros_like(state.coordinates)
    schedule = _schedule()

    result = ddpm_reverse_step(
        state,
        predicted_noise,
        schedule,
        generator=make_torch_generator(6),
    )

    alpha_t = schedule.alphas[2].to(torch.float64)
    beta_t = schedule.betas[2].to(torch.float64)
    alpha_bar_t = schedule.alpha_bars[2].to(torch.float64)
    alpha_bar_previous = schedule.alpha_bars[1].to(torch.float64)
    expected_mean = state.coordinates / torch.sqrt(alpha_t)
    expected_variance = beta_t * (1.0 - alpha_bar_previous) / (1.0 - alpha_bar_t)
    assert torch.allclose(result.posterior_mean, expected_mean, atol=1.0e-12)
    assert torch.allclose(
        result.posterior_variance,
        torch.full_like(result.posterior_variance, expected_variance),
        atol=1.0e-12,
    )
    assert result.previous_state.timestep.tolist() == [1, 1]


def test_local_to_global_transform_restores_valid_atoms_and_keeps_padding_zero() -> None:
    stage3_input, processed = _stage3_input(dtype=torch.float64)
    output = DDPMStage3Sampler(_schedule(num_timesteps=2)).sample(
        stage3_input,
        _conditioned_model("none", dtype=torch.float64).eval(),
        seed=9,
    )

    global_coordinates = restore_global_coordinates(output, processed.transform)
    valid = output.final_state.mask.atom_mask
    expected = output.coordinates + processed.transform.center[:, None, :]

    assert global_coordinates.shape == output.coordinates.shape
    assert torch.allclose(global_coordinates[valid], expected[valid])
    assert torch.count_nonzero(global_coordinates[~valid]) == 0


def test_additive_mode_requires_field_and_invalid_prediction_is_rejected() -> None:
    stage3_input, _ = _stage3_input()
    sampler = DDPMStage3Sampler(_schedule(num_timesteps=2))
    with pytest.raises(ValueError, match="pocket_field is required"):
        sampler.sample(
            stage3_input,
            _conditioned_model("additive"),
            seed=4,
        )

    from gpcr_ringgen.models.diffusion import initialize_stage3_state

    state = initialize_stage3_state(stage3_input, seed=4, timestep=1)
    with pytest.raises(ValueError, match=r"shape \[B, N, 3\]"):
        ddpm_reverse_step(
            state,
            torch.zeros(1, 2, 3),
            _schedule(num_timesteps=2),
        )
