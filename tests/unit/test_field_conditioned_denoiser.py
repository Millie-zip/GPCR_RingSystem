"""Tests for Pocket Field-conditioned equivariant Stage 3 denoising."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from gpcr_ringgen.core.types import BatchMask, BatchMetadata, Stage3DiffusionState
from gpcr_ringgen.models.diffusion import (
    DDPMConfig,
    DDPMNoiseSchedule,
    q_sample,
    training_target,
)
from gpcr_ringgen.models.equivariant import (
    EquivariantDenoiserBackbone,
    EquivariantDenoiserConfig,
    FieldConditionedEquivariantDenoiser,
    FieldConditioningConfig,
    load_field_conditioning_config,
)
from gpcr_ringgen.pocket import (
    GeometryFieldConfig,
    adapt_pocket_atoms,
    build_geometry_field,
    transform_coordinates,
)
from gpcr_ringgen.training import NoisePredictionLoss
from gpcr_ringgen.utils.seed import make_torch_generator


def _make_state(*, dtype: torch.dtype = torch.float32) -> Stage3DiffusionState:
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
        coordinates=torch.tensor(
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
        ),
        descriptor=torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 0.0]],
            dtype=dtype,
        ),
        timestep=torch.tensor([2, 7], dtype=torch.long),
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


def _raw_pocket_coordinates(dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor(
        [
            [[-2.0, -1.0, 0.0], [2.0, -1.0, 0.0], [0.0, 2.0, 0.0]],
            [[-1.0, 0.0, 1.0], [1.0, 0.0, -1.0], [0.0, 0.0, 0.0]],
        ],
        dtype=dtype,
    )


def _processed_pocket(coordinates: torch.Tensor):
    batch_size, num_points, _ = coordinates.shape
    pocket_mask = torch.tensor(
        [[True, True, True], [True, True, False]],
        device=coordinates.device,
    )

    def rows(prefix: str) -> list[list[str]]:
        return [
            [
                f"{prefix}{point}" if pocket_mask[batch, point] else ""
                for point in range(num_points)
            ]
            for batch in range(batch_size)
        ]

    return adapt_pocket_atoms(
        coordinates=coordinates,
        features=torch.ones(
            batch_size,
            num_points,
            2,
            dtype=coordinates.dtype,
            device=coordinates.device,
        ),
        atom_types=rows("atom"),
        elements=rows("C"),
        residue_names=rows("RES"),
        residue_ids=rows("id"),
        sample_ids=["sample-0", "sample-1"],
        pocket_mask=pocket_mask,
    )


def _field(coordinates: torch.Tensor):
    return build_geometry_field(
        _processed_pocket(coordinates),
        GeometryFieldConfig(
            sigma=1.3,
            cutoff=20.0,
            epsilon=1.0e-8,
            occupancy_enabled=True,
            nearest_distance_enabled=True,
        ),
    )


def _backbone(dtype: torch.dtype) -> EquivariantDenoiserBackbone:
    torch.manual_seed(2468)
    config = EquivariantDenoiserConfig(
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
    return EquivariantDenoiserBackbone(config).to(dtype=dtype)


def _model(
    mode: str,
    *,
    dtype: torch.dtype = torch.float32,
) -> FieldConditionedEquivariantDenoiser:
    backbone = _backbone(dtype)
    config = FieldConditioningConfig(
        mode=mode,
        channels=("occupancy", "nearest_distance"),
    )
    return FieldConditionedEquivariantDenoiser(backbone, config).to(dtype=dtype)


def _random_rotation(dtype: torch.dtype) -> torch.Tensor:
    generator = make_torch_generator(91)
    matrix = torch.randn(3, 3, dtype=dtype, generator=generator)
    rotation, _ = torch.linalg.qr(matrix)
    if torch.linalg.det(rotation) < 0:
        rotation[:, 0] = -rotation[:, 0]
    return rotation


def test_none_mode_exactly_matches_base_denoiser() -> None:
    state = _make_state()
    backbone = _backbone(state.coordinates.dtype).eval()
    model = FieldConditionedEquivariantDenoiser(
        backbone,
        FieldConditioningConfig(
            mode="none",
            channels=("occupancy", "nearest_distance"),
        ),
    ).eval()

    expected = backbone(state).coordinate_noise
    actual = model(state).coordinate_noise

    assert torch.equal(actual, expected)
    assert model.field_encoder is None


def test_additive_and_none_modes_have_matching_batch_output_shape() -> None:
    state = _make_state()
    pocket_field = _field(_raw_pocket_coordinates(state.coordinates.dtype))
    none_output = _model("none").eval()(state)
    additive_model = _model("additive").eval()

    field_result = additive_model.query_field(state, pocket_field)
    additive_output = additive_model(state, pocket_field=pocket_field)

    assert none_output.coordinate_noise.shape == additive_output.coordinate_noise.shape
    assert additive_output.coordinate_noise.shape == (2, 4, 3)
    assert field_result.values.shape == (2, 4, 2)
    assert field_result.channels == ("occupancy", "nearest_distance")
    assert torch.equal(field_result.query_mask, state.mask.atom_mask)
    assert torch.count_nonzero(additive_output.coordinate_noise[1, 2:]) == 0
    assert additive_output.coordinate_noise.device.type == "cpu"
    assert torch.isfinite(additive_output.coordinate_noise).all()


def test_changing_field_changes_conditioned_prediction() -> None:
    state = _make_state(dtype=torch.float64)
    original_coordinates = _raw_pocket_coordinates(torch.float64)
    altered_coordinates = original_coordinates.clone()
    altered_coordinates[0, 0] += torch.tensor([0.75, 0.4, 0.2], dtype=torch.float64)
    original_field = _field(original_coordinates)
    altered_field = _field(altered_coordinates)
    model = _model("additive", dtype=torch.float64).eval()

    original_query = model.query_field(state, original_field).values
    altered_query = model.query_field(state, altered_field).values
    original_prediction = model(state, pocket_field=original_field).coordinate_noise
    altered_prediction = model(state, pocket_field=altered_field).coordinate_noise

    assert not torch.allclose(original_query, altered_query)
    assert torch.max(torch.abs(original_prediction - altered_prediction)) > 1.0e-10


def test_diffusion_loss_backward_has_finite_conditioned_gradients() -> None:
    state = _make_state()
    schedule = DDPMNoiseSchedule(
        DDPMConfig(
            num_timesteps=20,
            beta_schedule="linear",
            beta_start=1.0e-3,
            beta_end=0.1,
            prediction_type="epsilon",
            loss_reduction="mean",
        )
    )
    forward = q_sample(
        state,
        torch.tensor([4, 12], dtype=torch.long),
        schedule,
        generator=make_torch_generator(32),
    )
    model = _model("additive")
    pocket_field = _field(_raw_pocket_coordinates(state.coordinates.dtype))

    prediction = model(
        forward.noisy_state,
        pocket_field=pocket_field,
    ).coordinate_noise
    loss = NoisePredictionLoss("mean")(
        prediction,
        training_target(forward),
        state.mask.atom_mask,
    )
    loss.backward()
    gradients = [
        parameter.grad for parameter in model.parameters() if parameter.grad is not None
    ]

    assert torch.isfinite(loss)
    assert gradients
    assert model.field_encoder is not None
    assert all(parameter.grad is not None for parameter in model.field_encoder.parameters())
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_joint_rotation_equivariance() -> None:
    dtype = torch.float64
    state = _make_state(dtype=dtype)
    pocket_coordinates = _raw_pocket_coordinates(dtype)
    rotation = _random_rotation(dtype)
    rotated_state = replace(
        state,
        coordinates=state.coordinates @ rotation.transpose(0, 1),
    )
    rotated_pocket_coordinates = pocket_coordinates @ rotation.transpose(0, 1)
    model = _model("additive", dtype=dtype).eval()

    prediction = model(
        state,
        pocket_field=_field(pocket_coordinates),
    ).coordinate_noise
    rotated_prediction = model(
        rotated_state,
        pocket_field=_field(rotated_pocket_coordinates),
    ).coordinate_noise
    expected = prediction @ rotation.transpose(0, 1)
    max_error = torch.max(torch.abs(rotated_prediction - expected)).item()

    print(f"field_conditioned_rotation_max_error={max_error:.3e}")
    assert torch.allclose(rotated_prediction, expected, atol=1.0e-9, rtol=1.0e-7)


def test_joint_translation_consistency_in_local_frame() -> None:
    dtype = torch.float64
    state = _make_state(dtype=dtype)
    pocket_coordinates = _raw_pocket_coordinates(dtype)
    translation = torch.tensor([8.0, -5.0, 2.5], dtype=dtype)
    original_pocket = _processed_pocket(pocket_coordinates)
    translated_pocket = _processed_pocket(pocket_coordinates + translation)
    original_state = replace(
        state,
        coordinates=transform_coordinates(
            state.coordinates,
            original_pocket.transform,
        ),
    )
    translated_state = replace(
        state,
        coordinates=transform_coordinates(
            state.coordinates + translation,
            translated_pocket.transform,
        ),
    )
    model = _model("additive", dtype=dtype).eval()

    original_prediction = model(
        original_state,
        pocket_field=build_geometry_field(
            original_pocket,
            GeometryFieldConfig(1.3, 20.0, 1.0e-8, True, True),
        ),
    ).coordinate_noise
    translated_prediction = model(
        translated_state,
        pocket_field=build_geometry_field(
            translated_pocket,
            GeometryFieldConfig(1.3, 20.0, 1.0e-8, True, True),
        ),
    ).coordinate_noise
    max_error = torch.max(
        torch.abs(translated_prediction - original_prediction)
    ).item()

    print(f"field_conditioned_translation_max_error={max_error:.3e}")
    assert torch.allclose(
        translated_prediction,
        original_prediction,
        atol=1.0e-9,
        rtol=1.0e-7,
    )


def test_descriptor_rows_remain_aligned_with_batch_samples() -> None:
    dtype = torch.float64
    state = _make_state(dtype=dtype)
    changed_descriptor = state.descriptor.clone()
    changed_descriptor[0] += torch.tensor([3.0, -2.0, 1.0], dtype=dtype)
    changed_state = replace(state, descriptor=changed_descriptor)
    pocket_field = _field(_raw_pocket_coordinates(dtype))
    model = _model("additive", dtype=dtype).eval()

    original = model(state, pocket_field=pocket_field).coordinate_noise
    changed = model(changed_state, pocket_field=pocket_field).coordinate_noise

    assert torch.max(torch.abs(changed[0] - original[0])) > 1.0e-10
    assert torch.equal(changed[1], original[1])


def test_config_loading_and_channel_mismatch_error() -> None:
    config_path = (
        Path(__file__).parents[2]
        / "configs"
        / "model"
        / "field_conditioning.yaml"
    )
    config = load_field_conditioning_config(config_path)
    state = _make_state()
    occupancy_only_field = build_geometry_field(
        _processed_pocket(_raw_pocket_coordinates(state.coordinates.dtype)),
        GeometryFieldConfig(1.3, 20.0, 1.0e-8, True, False),
    )
    model = FieldConditionedEquivariantDenoiser(_backbone(torch.float32), config)

    assert config.mode == "additive"
    assert config.channels == ("occupancy", "nearest_distance")
    with pytest.raises(ValueError, match="channels do not match"):
        model(state, pocket_field=occupancy_only_field)
