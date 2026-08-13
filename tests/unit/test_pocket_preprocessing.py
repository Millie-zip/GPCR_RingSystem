"""Tests for reversible GPCR pocket geometry preprocessing."""

from __future__ import annotations

import pytest
import torch

from gpcr_ringgen.pocket.preprocessing import (
    adapt_pocket_atoms,
    inverse_transform_coordinates,
    transform_coordinates,
)


def _single_pocket_inputs() -> dict[str, object]:
    return {
        "coordinates": torch.tensor(
            [[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]],
            dtype=torch.float64,
        ),
        "features": torch.tensor(
            [[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]],
            dtype=torch.float64,
        ),
        "atom_types": [["N", "CA", "C"]],
        "elements": [["N", "C", "C"]],
        "residue_names": [["ALA", "ALA", "ALA"]],
        "residue_ids": [["42", "42", "42"]],
        "chain_ids": [["A", "A", "A"]],
        "sample_ids": ["pocket-1"],
    }


def test_known_coordinates_are_centered_near_zero() -> None:
    inputs = _single_pocket_inputs()
    result = adapt_pocket_atoms(**inputs)

    local_center = result.pocket.coordinates.mean(dim=1)
    expected_center = torch.tensor([[2.0 / 3.0, 2.0 / 3.0, 0.0]], dtype=torch.float64)
    assert torch.allclose(local_center, torch.zeros_like(local_center), atol=1e-12)
    assert torch.allclose(result.transform.center, expected_center, atol=1e-12)
    assert result.metadata.sample_ids == ("pocket-1",)
    assert result.atom_metadata[0].elements == ("N", "C", "C")


def test_global_local_global_transform_is_reversible() -> None:
    inputs = _single_pocket_inputs()
    global_coordinates = inputs["coordinates"]
    result = adapt_pocket_atoms(**inputs)

    local_from_function = transform_coordinates(
        global_coordinates,
        result.transform,
    )
    restored = inverse_transform_coordinates(local_from_function, result.transform)

    assert torch.allclose(local_from_function, result.pocket.coordinates, atol=1e-12)
    assert torch.allclose(restored, global_coordinates, atol=1e-12)
    restored_single = inverse_transform_coordinates(
        result.pocket.coordinates[0],
        result.transform,
    )
    assert torch.allclose(restored_single, global_coordinates[0], atol=1e-12)


def test_translation_produces_identical_local_coordinates() -> None:
    inputs = _single_pocket_inputs()
    original = adapt_pocket_atoms(**inputs)
    translation = torch.tensor([10.0, -3.0, 7.5], dtype=torch.float64)
    translated_inputs = dict(inputs)
    translated_inputs["coordinates"] = inputs["coordinates"] + translation

    translated = adapt_pocket_atoms(**translated_inputs)

    assert torch.allclose(
        translated.pocket.coordinates,
        original.pocket.coordinates,
        atol=1e-12,
    )
    assert torch.allclose(
        translated.transform.center,
        original.transform.center + translation,
        atol=1e-12,
    )


def test_bounding_box_and_radius_have_expected_shapes_and_values() -> None:
    result = adapt_pocket_atoms(**_single_pocket_inputs())

    assert result.geometry.bounding_box_min.shape == (1, 3)
    assert result.geometry.bounding_box_max.shape == (1, 3)
    assert result.geometry.bounding_radius.shape == (1,)
    expected_min = torch.tensor([[-2.0 / 3.0, -2.0 / 3.0, 0.0]], dtype=torch.float64)
    expected_max = torch.tensor([[4.0 / 3.0, 4.0 / 3.0, 0.0]], dtype=torch.float64)
    expected_radius = torch.tensor([(20.0 / 9.0) ** 0.5], dtype=torch.float64)
    assert torch.allclose(result.geometry.bounding_box_min, expected_min, atol=1e-12)
    assert torch.allclose(result.geometry.bounding_box_max, expected_max, atol=1e-12)
    assert torch.allclose(result.geometry.bounding_radius, expected_radius, atol=1e-12)


def test_empty_pocket_has_clear_error() -> None:
    inputs = _single_pocket_inputs()
    inputs["pocket_mask"] = torch.zeros(1, 3, dtype=torch.bool)

    with pytest.raises(ValueError, match="at least one valid atom"):
        adapt_pocket_atoms(**inputs)


def test_invalid_coordinate_shape_has_clear_error() -> None:
    inputs = _single_pocket_inputs()
    inputs["coordinates"] = inputs["coordinates"][0]
    inputs["features"] = inputs["features"][0]

    with pytest.raises(ValueError, match="rank 3"):
        adapt_pocket_atoms(**inputs)


@pytest.mark.parametrize("invalid_value", [float("nan"), float("inf")])
def test_non_finite_coordinates_have_clear_error(invalid_value: float) -> None:
    inputs = _single_pocket_inputs()
    coordinates = inputs["coordinates"].clone()
    coordinates[0, 0, 0] = invalid_value
    inputs["coordinates"] = coordinates

    with pytest.raises(ValueError, match="NaN or Inf"):
        adapt_pocket_atoms(**inputs)


def test_batched_padded_cpu_smoke() -> None:
    coordinates = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
            [[1.0, 1.0, 1.0], [3.0, 1.0, 1.0], [99.0, 99.0, 99.0]],
        ],
        dtype=torch.float32,
    )
    mask = torch.tensor([[True, True, True], [True, True, False]])
    result = adapt_pocket_atoms(
        coordinates=coordinates,
        features=torch.ones(2, 3, 2),
        atom_types=[["N", "CA", "C"], ["N", "CA", ""]],
        elements=[["N", "C", "C"], ["N", "C", ""]],
        residue_names=[["ALA", "ALA", "ALA"], ["GLY", "GLY", ""]],
        residue_ids=[["42", "42", "42"], ["7", "7", ""]],
        sample_ids=["pocket-a", "pocket-b"],
        pocket_mask=mask,
    )

    weights = mask.to(result.pocket.coordinates.dtype)[..., None]
    valid_local_center = (result.pocket.coordinates * weights).sum(dim=1) / weights.sum(dim=1)
    assert result.batch_size == 2
    assert result.pocket.coordinates.device.type == "cpu"
    assert torch.allclose(valid_local_center, torch.zeros_like(valid_local_center), atol=1e-6)
    assert result.geometry.bounding_radius.shape == (2,)
