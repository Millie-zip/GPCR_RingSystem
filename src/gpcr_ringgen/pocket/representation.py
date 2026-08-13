"""Pocket geometry data structures built on the shared Stage 3 contract."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from gpcr_ringgen.core.types import BatchMetadata, PocketBatch, PocketRepresentation


@dataclass
class PocketAtomMetadata:
    """Padded non-tensor atom annotations supplied by an upstream parser.

    Each tuple has length ``P`` and uses an empty string only for padded atoms.
    No atom, residue, or element vocabulary is imposed here.
    """

    atom_types: tuple[str, ...]
    elements: tuple[str, ...]
    residue_names: tuple[str, ...]
    residue_ids: tuple[str, ...]
    chain_ids: tuple[str, ...] | None = None

    def validate(self, num_points: int, valid_mask: Tensor) -> None:
        """Validate annotation lengths and required values for real atoms."""
        fields = {
            "atom_types": self.atom_types,
            "elements": self.elements,
            "residue_names": self.residue_names,
            "residue_ids": self.residue_ids,
            "chain_ids": self.chain_ids,
        }
        for name, values in fields.items():
            if values is not None and len(values) != num_points:
                raise ValueError(f"{name} length must equal padded pocket size P.")
        for index, is_valid in enumerate(valid_mask.tolist()):
            if not is_valid:
                continue
            for name in ("atom_types", "elements", "residue_names", "residue_ids"):
                value = fields[name][index]
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"{name} must be non-empty for every valid atom.")


@dataclass
class PocketCoordinateTransform:
    """Translation-only global/local coordinate transform metadata.

    Tensor fields:
        center: Shape ``[B, 3]``; floating-point; same dtype and device as the
            pocket coordinates. Global-to-local subtracts this center and the
            inverse transform adds it. No rotation or PCA frame is stored.
    """

    center: Tensor
    kind: str = "translation"

    @property
    def batch_size(self) -> int:
        return int(self.center.shape[0])

    @property
    def device(self) -> torch.device:
        return self.center.device

    def validate(self) -> None:
        if not isinstance(self.center, Tensor):
            raise TypeError("transform center must be a torch.Tensor.")
        if self.center.ndim != 2 or self.center.shape[1] != 3:
            raise ValueError("transform center must have shape [B, 3].")
        if self.center.shape[0] < 1:
            raise ValueError("transform batch size must be positive.")
        if not torch.is_floating_point(self.center):
            raise TypeError("transform center must use a floating-point dtype.")
        if not bool(torch.isfinite(self.center).all()):
            raise ValueError("transform center contains NaN or Inf.")
        if self.kind != "translation":
            raise ValueError("Only translation transforms are supported in this stage.")


@dataclass
class PocketGeometry:
    """Basic local-coordinate geometric extent for each pocket.

    Tensor fields:
        bounding_box_min: Shape ``[B, 3]``; floating-point; pocket device/dtype;
            coordinate-wise minima over atoms with ``pocket_mask=True``.
        bounding_box_max: Shape ``[B, 3]``; floating-point; pocket device/dtype;
            coordinate-wise maxima over atoms with ``pocket_mask=True``.
        bounding_radius: Shape ``[B]``; floating-point; pocket device/dtype;
            maximum Euclidean distance from the saved center over valid atoms.
    """

    bounding_box_min: Tensor
    bounding_box_max: Tensor
    bounding_radius: Tensor

    def validate(
        self,
        *,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        tensors = {
            "bounding_box_min": (self.bounding_box_min, (batch_size, 3)),
            "bounding_box_max": (self.bounding_box_max, (batch_size, 3)),
            "bounding_radius": (self.bounding_radius, (batch_size,)),
        }
        for name, (tensor, expected_shape) in tensors.items():
            if not isinstance(tensor, Tensor):
                raise TypeError(f"{name} must be a torch.Tensor.")
            if tuple(tensor.shape) != expected_shape:
                raise ValueError(f"{name} must have shape {list(expected_shape)}.")
            if tensor.dtype != dtype or tensor.device != device:
                raise ValueError(f"{name} must match pocket coordinate dtype/device.")
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"{name} contains NaN or Inf.")
        if bool((self.bounding_box_min > self.bounding_box_max).any()):
            raise ValueError("bounding_box_min must not exceed bounding_box_max.")
        if bool((self.bounding_radius < 0).any()):
            raise ValueError("bounding_radius must be non-negative.")


@dataclass
class ProcessedPocketBatch:
    """Reversible local-coordinate pocket representation for Field input.

    Tensor fields:
        pocket_mask: Shape ``[B, P]``; ``torch.bool``; same device as
            ``pocket.coordinates``. ``True`` identifies a real atom and
            ``False`` identifies padding. Embedded ``PocketBatch`` coordinates
            are local coordinates with shape ``[B, P, 3]``.
    """

    pocket: PocketBatch
    pocket_mask: Tensor
    atom_metadata: tuple[PocketAtomMetadata, ...]
    transform: PocketCoordinateTransform
    geometry: PocketGeometry
    metadata: BatchMetadata

    @property
    def batch_size(self) -> int:
        return self.pocket.batch_size

    def validate(self) -> None:
        self.pocket.validate()
        coordinates = self.pocket.coordinates
        batch_size, num_points, _ = coordinates.shape
        if not isinstance(self.pocket_mask, Tensor):
            raise TypeError("pocket_mask must be a torch.Tensor.")
        if tuple(self.pocket_mask.shape) != (batch_size, num_points):
            raise ValueError("pocket_mask must have shape [B, P].")
        if self.pocket_mask.dtype != torch.bool:
            raise TypeError("pocket_mask must have dtype torch.bool.")
        if self.pocket_mask.device != coordinates.device:
            raise ValueError("pocket_mask must be on the pocket coordinate device.")
        if not bool(torch.isfinite(coordinates).all()):
            raise ValueError("pocket coordinates contain NaN or Inf.")
        if not bool(torch.isfinite(self.pocket.features).all()):
            raise ValueError("pocket features contain NaN or Inf.")
        if bool((self.pocket_mask.sum(dim=1) == 0).any()):
            raise ValueError("Every pocket must contain at least one valid atom.")
        if len(self.atom_metadata) != batch_size:
            raise ValueError("atom_metadata length must equal batch size.")
        for sample_metadata, valid_mask in zip(
            self.atom_metadata,
            self.pocket_mask,
            strict=True,
        ):
            sample_metadata.validate(num_points, valid_mask)
        self.metadata.validate(batch_size)
        self.transform.validate()
        if self.transform.batch_size != batch_size:
            raise ValueError("Transform batch size must equal pocket batch size.")
        if (
            self.transform.center.dtype != coordinates.dtype
            or self.transform.device != coordinates.device
        ):
            raise ValueError("Transform center must match pocket coordinate dtype/device.")
        self.geometry.validate(
            batch_size=batch_size,
            dtype=coordinates.dtype,
            device=coordinates.device,
        )


__all__ = [
    "PocketAtomMetadata",
    "PocketBatch",
    "PocketCoordinateTransform",
    "PocketGeometry",
    "PocketRepresentation",
    "ProcessedPocketBatch",
]
