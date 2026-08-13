"""Reversible GPCR pocket geometry preprocessing without field construction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from gpcr_ringgen.core.types import BatchMetadata, PocketBatch
from gpcr_ringgen.pocket.representation import (
    PocketAtomMetadata,
    PocketCoordinateTransform,
    PocketGeometry,
    ProcessedPocketBatch,
)


def read_pocket_file(path: str | Path) -> PocketBatch:
    """Reserve a parser boundary without implementing a PDB parser.

    No RDKit, BioPython, MDAnalysis, ProDy, or equivalent parser is currently
    configured. Downstream code should adapt coordinates produced by an
    upstream parser with :func:`adapt_pocket_atoms`.
    """
    raise NotImplementedError(
        f"No protein structure parser is configured for {Path(path)}. "
        "Pass upstream parser output to adapt_pocket_atoms instead."
    )


def _validate_coordinates_for_transform(
    coordinates: Tensor,
    transform: PocketCoordinateTransform,
) -> None:
    if not isinstance(coordinates, Tensor):
        raise TypeError("coordinates must be a torch.Tensor.")
    if coordinates.ndim not in (2, 3) or coordinates.shape[-1] != 3:
        raise ValueError("coordinates must have shape [N, 3] or [B, N, 3].")
    if not torch.is_floating_point(coordinates):
        raise TypeError("coordinates must use a floating-point dtype.")
    if not bool(torch.isfinite(coordinates).all()):
        raise ValueError("coordinates contain NaN or Inf.")
    transform.validate()
    if transform.center.dtype != coordinates.dtype:
        raise TypeError("coordinates and transform center must share a dtype.")
    if transform.device != coordinates.device:
        raise ValueError("coordinates and transform center must share a device.")
    coordinate_batch = 1 if coordinates.ndim == 2 else coordinates.shape[0]
    if transform.batch_size != coordinate_batch:
        raise ValueError("coordinates and transform batch sizes differ.")


def transform_coordinates(
    coordinates: Tensor,
    transform: PocketCoordinateTransform,
) -> Tensor:
    """Apply ``r_local = r_global - center`` to one or more pockets."""
    _validate_coordinates_for_transform(coordinates, transform)
    if coordinates.ndim == 2:
        return coordinates - transform.center[0]
    return coordinates - transform.center[:, None, :]


def inverse_transform_coordinates(
    coordinates: Tensor,
    transform: PocketCoordinateTransform,
) -> Tensor:
    """Apply ``r_global = r_local + center`` using saved metadata."""
    _validate_coordinates_for_transform(coordinates, transform)
    if coordinates.ndim == 2:
        return coordinates + transform.center[0]
    return coordinates + transform.center[:, None, :]


def _validate_pocket_mask(pocket: PocketBatch, pocket_mask: Tensor) -> None:
    batch_size, num_points, _ = pocket.coordinates.shape
    if not isinstance(pocket_mask, Tensor):
        raise TypeError("pocket_mask must be a torch.Tensor.")
    if tuple(pocket_mask.shape) != (batch_size, num_points):
        raise ValueError("pocket_mask must have shape [B, P].")
    if pocket_mask.dtype != torch.bool:
        raise TypeError("pocket_mask must have dtype torch.bool.")
    if pocket_mask.device != pocket.coordinates.device:
        raise ValueError("pocket_mask must be on the coordinate device.")
    if bool((pocket_mask.sum(dim=1) == 0).any()):
        raise ValueError("Every pocket must contain at least one valid atom.")


def _compute_geometry(
    local_coordinates: Tensor,
    pocket_mask: Tensor,
) -> PocketGeometry:
    expanded_mask = pocket_mask[..., None]
    positive_infinity = torch.full_like(local_coordinates, torch.inf)
    negative_infinity = torch.full_like(local_coordinates, -torch.inf)
    bounding_box_min = torch.where(
        expanded_mask,
        local_coordinates,
        positive_infinity,
    ).amin(dim=1)
    bounding_box_max = torch.where(
        expanded_mask,
        local_coordinates,
        negative_infinity,
    ).amax(dim=1)
    distances = torch.linalg.vector_norm(local_coordinates, dim=-1)
    bounding_radius = torch.where(
        pocket_mask,
        distances,
        torch.zeros_like(distances),
    ).amax(dim=1)
    return PocketGeometry(
        bounding_box_min=bounding_box_min,
        bounding_box_max=bounding_box_max,
        bounding_radius=bounding_radius,
    )


def preprocess_pocket(
    pocket: PocketBatch,
    *,
    pocket_mask: Tensor,
    atom_metadata: Sequence[PocketAtomMetadata],
    metadata: BatchMetadata,
) -> ProcessedPocketBatch:
    """Center a batched pocket and compute reversible geometric metadata.

    Centering uses only atoms where ``pocket_mask=True``. The feature tensor and
    all atom/sample metadata are passed through unchanged. No PCA alignment,
    field construction, voxelization, ligand initialization, or model call is
    performed.
    """
    pocket.validate()
    _validate_pocket_mask(pocket, pocket_mask)
    if not bool(torch.isfinite(pocket.coordinates).all()):
        raise ValueError("pocket coordinates contain NaN or Inf.")
    if not bool(torch.isfinite(pocket.features).all()):
        raise ValueError("pocket features contain NaN or Inf.")

    batch_size = pocket.batch_size
    atom_metadata_tuple = tuple(atom_metadata)
    if len(atom_metadata_tuple) != batch_size:
        raise ValueError("atom_metadata length must equal pocket batch size.")
    metadata.validate(batch_size)

    weights = pocket_mask.to(dtype=pocket.coordinates.dtype)[..., None]
    counts = weights.sum(dim=1)
    center = (pocket.coordinates * weights).sum(dim=1) / counts
    transform = PocketCoordinateTransform(center=center)
    local_coordinates = transform_coordinates(pocket.coordinates, transform)
    geometry = _compute_geometry(local_coordinates, pocket_mask)
    result = ProcessedPocketBatch(
        pocket=PocketBatch(
            coordinates=local_coordinates,
            features=pocket.features,
        ),
        pocket_mask=pocket_mask,
        atom_metadata=atom_metadata_tuple,
        transform=transform,
        geometry=geometry,
        metadata=metadata,
    )
    result.validate()
    return result


def _normalize_string_rows(
    name: str,
    values: Sequence[Sequence[str]],
    *,
    batch_size: int,
    num_points: int,
) -> tuple[tuple[str, ...], ...]:
    if len(values) != batch_size:
        raise ValueError(f"{name} must contain one row per pocket.")
    rows = tuple(tuple(row) for row in values)
    if any(len(row) != num_points for row in rows):
        raise ValueError(f"Every {name} row must have padded length P.")
    return rows


def adapt_pocket_atoms(
    *,
    coordinates: Tensor,
    features: Tensor,
    atom_types: Sequence[Sequence[str]],
    elements: Sequence[Sequence[str]],
    residue_names: Sequence[Sequence[str]],
    residue_ids: Sequence[Sequence[str]],
    sample_ids: Sequence[str],
    pocket_mask: Tensor | None = None,
    chain_ids: Sequence[Sequence[str]] | None = None,
    metadata_records: Sequence[Mapping[str, Any]] | None = None,
) -> ProcessedPocketBatch:
    """Adapt batched output from an external protein parser.

    ``coordinates`` must have shape ``[B, P, 3]`` and ``features`` must have
    shape ``[B, P, F_pocket]``. Both are floating-point tensors on the same
    device. This function organizes annotations and geometry but does not parse
    PDB text or encode atom types/elements into numerical features.
    """
    pocket = PocketBatch(coordinates=coordinates, features=features)
    pocket.validate()
    batch_size, num_points, _ = coordinates.shape
    if pocket_mask is None:
        pocket_mask = torch.ones(
            (batch_size, num_points),
            dtype=torch.bool,
            device=coordinates.device,
        )

    normalized_atom_types = _normalize_string_rows(
        "atom_types",
        atom_types,
        batch_size=batch_size,
        num_points=num_points,
    )
    normalized_elements = _normalize_string_rows(
        "elements",
        elements,
        batch_size=batch_size,
        num_points=num_points,
    )
    normalized_residue_names = _normalize_string_rows(
        "residue_names",
        residue_names,
        batch_size=batch_size,
        num_points=num_points,
    )
    normalized_residue_ids = _normalize_string_rows(
        "residue_ids",
        residue_ids,
        batch_size=batch_size,
        num_points=num_points,
    )
    normalized_chain_ids = (
        None
        if chain_ids is None
        else _normalize_string_rows(
            "chain_ids",
            chain_ids,
            batch_size=batch_size,
            num_points=num_points,
        )
    )
    atom_metadata = tuple(
        PocketAtomMetadata(
            atom_types=normalized_atom_types[index],
            elements=normalized_elements[index],
            residue_names=normalized_residue_names[index],
            residue_ids=normalized_residue_ids[index],
            chain_ids=(
                None
                if normalized_chain_ids is None
                else normalized_chain_ids[index]
            ),
        )
        for index in range(batch_size)
    )
    records = (
        tuple({} for _ in range(batch_size))
        if metadata_records is None
        else tuple(metadata_records)
    )
    metadata = BatchMetadata(sample_ids=tuple(sample_ids), records=records)
    return preprocess_pocket(
        pocket,
        pocket_mask=pocket_mask,
        atom_metadata=atom_metadata,
        metadata=metadata,
    )


__all__ = [
    "adapt_pocket_atoms",
    "inverse_transform_coordinates",
    "preprocess_pocket",
    "read_pocket_file",
    "transform_coordinates",
]
