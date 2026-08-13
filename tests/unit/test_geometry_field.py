"""Tests for the basic vectorized pocket geometry field."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from gpcr_ringgen.pocket import (
    GeometryFieldConfig,
    QueryablePocketField,
    adapt_pocket_atoms,
    build_geometry_field,
    load_geometry_field_config,
    transform_coordinates,
)


def _config(
    *,
    sigma: float = 1.0,
    cutoff: float = 6.0,
    occupancy: bool = True,
    nearest_distance: bool = True,
) -> GeometryFieldConfig:
    return GeometryFieldConfig(
        sigma=sigma,
        cutoff=cutoff,
        epsilon=1.0e-8,
        occupancy_enabled=occupancy,
        nearest_distance_enabled=nearest_distance,
    )


def _processed_pocket(
    coordinates: torch.Tensor,
    *,
    pocket_mask: torch.Tensor | None = None,
    sample_ids: list[str] | None = None,
):
    batch_size, num_points, _ = coordinates.shape
    if pocket_mask is None:
        pocket_mask = torch.ones(
            batch_size,
            num_points,
            dtype=torch.bool,
            device=coordinates.device,
        )
    if sample_ids is None:
        sample_ids = [f"pocket-{index}" for index in range(batch_size)]

    def rows(prefix: str) -> list[list[str]]:
        return [
            [f"{prefix}{point}" if pocket_mask[batch, point] else "" for point in range(num_points)]
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
        sample_ids=sample_ids,
        pocket_mask=pocket_mask,
    )


def test_atom_location_has_higher_occupancy_than_far_point() -> None:
    pocket = _processed_pocket(
        torch.tensor([[[0.0, 0.0, 0.0]]], dtype=torch.float64)
    )
    field = build_geometry_field(pocket, _config(cutoff=20.0))
    result = field.query(
        torch.tensor([[[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]], dtype=torch.float64)
    )

    occupancy = result.values[0, :, result.channels.index("occupancy")]
    assert occupancy[0] > occupancy[1]


def test_single_point_unbatched_query_returns_q_by_c() -> None:
    pocket = _processed_pocket(
        torch.tensor([[[0.0, 0.0, 0.0]]], dtype=torch.float64)
    )
    field = build_geometry_field(pocket, _config())

    result = field.query(torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64))

    assert result.values.shape == (1, 2)
    assert result.query_mask.shape == (1,)
    assert not result.is_batched
    assert torch.allclose(
        result.select_channel("occupancy"),
        torch.ones(1, dtype=torch.float64),
    )
    assert torch.allclose(
        result.select_channel("nearest_distance"),
        torch.zeros(1, dtype=torch.float64),
    )


def test_multiple_points_unbatched_query_exposes_channel_metadata() -> None:
    pocket = _processed_pocket(
        torch.tensor([[[0.0, 0.0, 0.0]]], dtype=torch.float32)
    )
    field = build_geometry_field(pocket, _config(cutoff=10.0))
    query_mask = torch.tensor([True, False, True])

    result = field.query(
        torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            dtype=torch.float32,
        ),
        query_mask=query_mask,
    )

    assert result.values.shape == (3, 2)
    assert result.query_mask.shape == (3,)
    assert result.channels == ("occupancy", "nearest_distance")
    assert tuple(channel.name for channel in result.channel_metadata) == result.channels
    assert result.channel_index("nearest_distance") == 1
    assert result.channel_metadata[0].to_metadata()["unit"] == "dimensionless"
    assert torch.count_nonzero(result.values[~query_mask]) == 0


def test_gaussian_occupancy_decreases_with_distance() -> None:
    pocket = _processed_pocket(
        torch.tensor([[[0.0, 0.0, 0.0]]], dtype=torch.float64)
    )
    field = build_geometry_field(pocket, _config(sigma=1.0, cutoff=10.0))
    result = field.query(
        torch.tensor(
            [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]],
            dtype=torch.float64,
        )
    )

    occupancy = result.values[0, :, result.channels.index("occupancy")]
    expected = torch.exp(-torch.tensor([0.0, 0.5, 2.0], dtype=torch.float64))
    assert occupancy[0] > occupancy[1] > occupancy[2]
    assert torch.allclose(occupancy, expected, atol=1.0e-12)


def test_pocket_and_query_translation_preserves_field_values() -> None:
    coordinates = torch.tensor(
        [[[1.0, 2.0, 3.0], [3.0, 2.0, 3.0]]], dtype=torch.float64
    )
    translation = torch.tensor([9.0, -4.0, 2.5], dtype=torch.float64)
    global_queries = torch.tensor(
        [[[2.0, 2.0, 3.0], [5.0, 2.0, 3.0]]], dtype=torch.float64
    )
    original = _processed_pocket(coordinates)
    translated = _processed_pocket(coordinates + translation)

    original_values = build_geometry_field(original, _config()).query(
        transform_coordinates(global_queries, original.transform)
    ).values
    translated_values = build_geometry_field(translated, _config()).query(
        transform_coordinates(global_queries + translation, translated.transform)
    ).values

    assert torch.allclose(original_values, translated_values, atol=1.0e-12)


def test_output_shape_channel_order_and_query_mask() -> None:
    coordinates = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            [[5.0, 1.0, 0.0], [7.0, 1.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    pocket = _processed_pocket(coordinates)
    field = build_geometry_field(pocket, _config())
    queries = torch.zeros(2, 3, 3, dtype=torch.float32)
    query_mask = torch.tensor([[True, True, False], [True, False, False]])

    assert isinstance(field, QueryablePocketField)
    field.validate(batch_size=2, device=queries.device)
    result = field.query(queries, query_mask=query_mask)

    assert result.values.shape == (2, 3, 2)
    assert result.query_mask.shape == (2, 3)
    assert result.channels == ("occupancy", "nearest_distance")
    assert result.values.dtype == queries.dtype
    assert result.values.device.type == "cpu"
    assert torch.count_nonzero(result.values[~query_mask]) == 0
    assert torch.isfinite(result.values).all()


def test_padded_pocket_atoms_do_not_contribute_in_batch() -> None:
    coordinates = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            [[10.0, 0.0, 0.0], [10_000.0, 0.0, 0.0]],
        ],
        dtype=torch.float64,
    )
    pocket_mask = torch.tensor([[True, True], [True, False]])
    pocket = _processed_pocket(coordinates, pocket_mask=pocket_mask)
    field = build_geometry_field(pocket, _config(cutoff=20.0))
    queries = torch.tensor(
        [
            [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        ],
        dtype=torch.float64,
    )

    result = field.query(queries)
    occupancy = result.values[..., result.channels.index("occupancy")]
    nearest = result.values[..., result.channels.index("nearest_distance")]

    assert result.values.shape == (2, 2, 2)
    assert torch.allclose(occupancy[1, 0], torch.tensor(1.0, dtype=torch.float64))
    assert torch.allclose(nearest[1], torch.tensor([0.0, 1.0], dtype=torch.float64))


def test_cutoff_excludes_distant_atoms_from_occupancy() -> None:
    pocket = _processed_pocket(
        torch.tensor([[[0.0, 0.0, 0.0]]], dtype=torch.float64)
    )
    field = build_geometry_field(
        pocket,
        _config(sigma=10.0, cutoff=1.5, nearest_distance=False),
    )
    result = field.query(
        torch.tensor([[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]], dtype=torch.float64)
    )

    assert result.channels == ("occupancy",)
    assert result.values[0, 0, 0] > 0
    assert result.values[0, 1, 0] == 0


def test_extremely_far_finite_query_stays_finite_on_cpu() -> None:
    pocket = _processed_pocket(
        torch.tensor([[[0.0, 0.0, 0.0]]], dtype=torch.float32)
    )
    field = build_geometry_field(pocket, _config())

    result = field.query(torch.full((1, 2, 3), 1.0e20, dtype=torch.float32))

    assert result.values.device.type == "cpu"
    assert torch.isfinite(result.values).all()
    assert result.values[..., result.channels.index("occupancy")].eq(0).all()


def test_empty_single_and_batched_queries_return_empty_features() -> None:
    pocket = _processed_pocket(
        torch.tensor([[[0.0, 0.0, 0.0]]], dtype=torch.float32)
    )
    field = build_geometry_field(pocket, _config())

    single_result = field.query(torch.empty(0, 3, dtype=torch.float32))
    batched_result = field.query(torch.empty(1, 0, 3, dtype=torch.float32))

    assert single_result.values.shape == (0, 2)
    assert single_result.query_mask.shape == (0,)
    assert not single_result.is_batched
    assert batched_result.values.shape == (1, 0, 2)
    assert batched_result.query_mask.shape == (1, 0)
    assert batched_result.is_batched
    assert torch.isfinite(single_result.values).all()
    assert torch.isfinite(batched_result.values).all()


def test_empty_processed_pocket_is_rejected_when_building_field() -> None:
    pocket = _processed_pocket(
        torch.tensor([[[0.0, 0.0, 0.0]]], dtype=torch.float32)
    )
    pocket.pocket_mask[:] = False

    with pytest.raises(ValueError, match="at least one valid atom"):
        build_geometry_field(pocket, _config())


def test_invalid_query_shape_and_dtype_have_clear_errors() -> None:
    pocket = _processed_pocket(
        torch.tensor([[[0.0, 0.0, 0.0]]], dtype=torch.float32)
    )
    field = build_geometry_field(pocket, _config())

    with pytest.raises(ValueError, match=r"shape \[Q, 3\] or \[B, Q, 3\]"):
        field.query(torch.zeros(2, 4, dtype=torch.float32))
    with pytest.raises(TypeError, match="dtype must match"):
        field.query(torch.zeros(1, 2, 3, dtype=torch.float64))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is optional")
def test_cuda_query_preserves_device_and_dtype_when_available() -> None:
    device = torch.device("cuda")
    coordinates = torch.tensor(
        [[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]],
        dtype=torch.float32,
        device=device,
    )
    pocket = _processed_pocket(coordinates)
    field = build_geometry_field(pocket, _config())
    queries = torch.zeros(1, 3, 3, dtype=torch.float32, device=device)

    result = field.query(queries)

    assert result.values.device.type == "cuda"
    assert result.values.dtype == torch.float32
    assert result.query_mask.device.type == "cuda"


def test_repository_geometry_field_config_loads() -> None:
    config_path = Path(__file__).parents[2] / "configs" / "pocket" / "geometry_field.yaml"

    config = load_geometry_field_config(config_path)

    assert config.sigma == 1.0
    assert config.cutoff == 6.0
    assert config.epsilon == 1.0e-8
    assert config.occupancy_enabled
    assert config.nearest_distance_enabled
