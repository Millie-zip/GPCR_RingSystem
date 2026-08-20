"""Small, stable data structures shared by all project modules."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import torch
from torch import Tensor


def _require_rank(name: str, tensor: Tensor, rank: int) -> None:
    if not isinstance(tensor, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if tensor.ndim != rank:
        raise ValueError(
            f"{name} must have rank {rank}, got shape {tuple(tensor.shape)}."
        )


def _require_floating(name: str, tensor: Tensor) -> None:
    if not torch.is_floating_point(tensor):
        raise TypeError(f"{name} must use a floating-point dtype, got {tensor.dtype}.")


def _require_bool(name: str, tensor: Tensor) -> None:
    if tensor.dtype != torch.bool:
        raise TypeError(f"{name} must have dtype torch.bool, got {tensor.dtype}.")


def _validate_edge_references(
    edge_index: Tensor,
    atom_mask: Tensor,
    edge_mask: Tensor,
) -> None:
    """Validate only real edges; padded entries may use arbitrary sentinels."""
    num_atoms = atom_mask.shape[1]
    for batch_index in range(edge_index.shape[0]):
        valid_edges = edge_mask[batch_index]
        if not bool(valid_edges.any()):
            continue
        endpoints = edge_index[batch_index, :, valid_edges]
        if bool(((endpoints < 0) | (endpoints >= num_atoms)).any()):
            raise ValueError(
                "edge_index contains an out-of-range endpoint for a valid edge."
            )
        referenced_atoms = atom_mask[batch_index, endpoints]
        if not bool(referenced_atoms.all()):
            raise ValueError(
                "A valid edge references an atom marked as padding by atom_mask."
            )


@dataclass
class RingTopology:
    """Placeholder identity and metadata for a ring topology."""

    topology_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScaffoldCondition:
    """Minimal conditioning object associated with a ring topology."""

    scaffold_id: str
    topology: RingTopology
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DiffusionState:
    """Schema-free state container for future diffusion implementations."""

    payload: dict[str, Any] = field(default_factory=dict)
    timestep: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PocketRepresentation:
    """Schema-free placeholder for a future GPCR pocket representation."""

    pocket_id: str
    features: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GeneratedMolecule:
    """Inspectable output record; not necessarily chemically meaningful."""

    molecule_id: str
    representation: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MolecularGraph:
    """Padded batched molecular/scaffold graph without a fixed vocabulary.

    Tensor fields:
        atom_features: Shape ``[B, N, F_atom]``; floating-point; expected on the
            active model device. Rows with ``atom_mask=False`` are padding.
        edge_index: Shape ``[B, 2, E]``; ``torch.long``; same device as the
            features. Entries with ``edge_mask=False`` are padding and ignored.
        edge_features: Shape ``[B, E, F_edge]``; floating-point; same device and
            dtype as ``atom_features``. Rows with ``edge_mask=False`` are padding.
        coordinates: Optional shape ``[B, N, 3]``; floating-point; same device
            and dtype as ``atom_features``. Padded atoms have unspecified values.

    ``B`` is batch size, ``N`` padded atom capacity, and ``E`` padded edge
    capacity. The feature dimensions remain scientific design decisions.
    """

    atom_features: Tensor
    edge_index: Tensor
    edge_features: Tensor
    coordinates: Tensor | None = None

    @property
    def batch_size(self) -> int:
        return int(self.atom_features.shape[0])

    def validate(self) -> None:
        """Validate graph ranks, aligned dimensions, dtypes, and devices."""
        _require_rank("atom_features", self.atom_features, 3)
        _require_rank("edge_index", self.edge_index, 3)
        _require_rank("edge_features", self.edge_features, 3)
        _require_floating("atom_features", self.atom_features)
        _require_floating("edge_features", self.edge_features)
        if self.edge_index.dtype != torch.long:
            raise TypeError(
                f"edge_index must have dtype torch.long, got {self.edge_index.dtype}."
            )

        batch_size, num_atoms, atom_feature_dim = self.atom_features.shape
        edge_batch, index_axis, num_edges = self.edge_index.shape
        feature_batch, feature_edges, edge_feature_dim = self.edge_features.shape
        if batch_size < 1 or num_atoms < 1 or atom_feature_dim < 1:
            raise ValueError("atom_features dimensions B, N, and F_atom must be positive.")
        if index_axis != 2:
            raise ValueError("edge_index must have shape [B, 2, E].")
        if (edge_batch, num_edges) != (feature_batch, feature_edges):
            raise ValueError("edge_index and edge_features batch/edge dimensions differ.")
        if edge_feature_dim < 1:
            raise ValueError("edge_features dimension F_edge must be positive.")
        if batch_size != edge_batch:
            raise ValueError("Atom and edge batch sizes differ.")

        tensors = {
            "edge_index": self.edge_index,
            "edge_features": self.edge_features,
        }
        if self.coordinates is not None:
            _require_rank("coordinates", self.coordinates, 3)
            _require_floating("coordinates", self.coordinates)
            if tuple(self.coordinates.shape) != (batch_size, num_atoms, 3):
                raise ValueError("coordinates must have shape [B, N, 3].")
            tensors["coordinates"] = self.coordinates

        for name, tensor in tensors.items():
            if tensor.device != self.atom_features.device:
                raise ValueError(f"{name} must be on the atom_features device.")
        for name, tensor in (
            ("edge_features", self.edge_features),
            ("coordinates", self.coordinates),
        ):
            if tensor is not None and tensor.dtype != self.atom_features.dtype:
                raise TypeError(f"{name} must use the atom_features dtype.")


@dataclass
class DescriptorBatch:
    """Batched molecular descriptors without fixing descriptor semantics.

    Tensor fields:
        values: Shape ``[B, D]``; floating-point; expected on the active model
            device. ``descriptor_mask=False`` in :class:`BatchMask` marks a
            missing or padded descriptor value that consumers must ignore.
    """

    values: Tensor

    @property
    def batch_size(self) -> int:
        return int(self.values.shape[0])

    def validate(self) -> None:
        _require_rank("descriptor.values", self.values, 2)
        _require_floating("descriptor.values", self.values)
        if self.values.shape[0] < 1:
            raise ValueError("descriptor.values batch dimension must be positive.")


@dataclass
class PocketBatch:
    """Raw padded pocket atoms/features before any field construction.

    Tensor fields:
        coordinates: Shape ``[B, P, 3]``; floating-point; expected on the active
            model device. Entries with ``pocket_mask=False`` are padding.
        features: Shape ``[B, P, F_pocket]``; floating-point; same dtype/device
            as ``coordinates``. Entries with ``pocket_mask=False`` are padding.

    The feature meaning and pocket extraction method are intentionally unspecified.
    """

    coordinates: Tensor
    features: Tensor

    @property
    def batch_size(self) -> int:
        return int(self.coordinates.shape[0])

    def validate(self) -> None:
        _require_rank("pocket.coordinates", self.coordinates, 3)
        _require_rank("pocket.features", self.features, 3)
        _require_floating("pocket.coordinates", self.coordinates)
        _require_floating("pocket.features", self.features)
        batch_size, num_points, coordinate_dim = self.coordinates.shape
        feature_batch, feature_points, feature_dim = self.features.shape
        if coordinate_dim != 3:
            raise ValueError("pocket.coordinates must have shape [B, P, 3].")
        if (batch_size, num_points) != (feature_batch, feature_points):
            raise ValueError("Pocket coordinates and features dimensions differ.")
        if batch_size < 1 or num_points < 0 or feature_dim < 1:
            raise ValueError("Pocket dimensions B and F_pocket must be positive; P may be zero.")
        if self.features.device != self.coordinates.device:
            raise ValueError("Pocket coordinates and features must share a device.")
        if self.features.dtype != self.coordinates.dtype:
            raise TypeError("Pocket coordinates and features must share a dtype.")


@dataclass
class BatchMask:
    """Boolean validity masks for padded Stage 2/Stage 3 tensors.

    Tensor fields:
        atom_mask: Shape ``[B, N]``; ``torch.bool``; ``True`` means a real atom,
            ``False`` means padding; expected on the graph tensor device.
        edge_mask: Shape ``[B, E]``; ``torch.bool``; ``True`` means a real edge,
            ``False`` means padding; expected on the graph tensor device.
        descriptor_mask: Optional shape ``[B, D]``; ``torch.bool``; ``True`` means
            the descriptor value is present, ``False`` means missing/padding;
            expected on the same device as ``atom_mask``.
        pocket_mask: Optional shape ``[B, P]``; ``torch.bool``; ``True`` means a
            real pocket point/atom, ``False`` means padding; expected on the same
            device as ``atom_mask``.
    """

    atom_mask: Tensor
    edge_mask: Tensor
    descriptor_mask: Tensor | None = None
    pocket_mask: Tensor | None = None

    @property
    def batch_size(self) -> int:
        return int(self.atom_mask.shape[0])

    @property
    def device(self) -> torch.device:
        return self.atom_mask.device

    def validate(self) -> None:
        masks = {
            "atom_mask": self.atom_mask,
            "edge_mask": self.edge_mask,
            "descriptor_mask": self.descriptor_mask,
            "pocket_mask": self.pocket_mask,
        }
        for name, mask in masks.items():
            if mask is None:
                continue
            _require_rank(name, mask, 2)
            _require_bool(name, mask)
            if mask.shape[0] != self.atom_mask.shape[0]:
                raise ValueError(f"{name} has a different batch size.")
            if mask.device != self.atom_mask.device:
                raise ValueError(f"{name} must be on the atom_mask device.")
        if self.atom_mask.shape[0] < 1:
            raise ValueError("Mask batch size must be positive.")


@dataclass
class BatchMetadata:
    """Per-sample identifiers and free-form non-tensor metadata."""

    sample_ids: tuple[str, ...]
    records: tuple[Mapping[str, Any], ...]

    def validate(self, batch_size: int) -> None:
        if len(self.sample_ids) != batch_size:
            raise ValueError("sample_ids length must equal batch size.")
        if len(self.records) != batch_size:
            raise ValueError("metadata records length must equal batch size.")
        if any(not isinstance(sample_id, str) or not sample_id.strip() for sample_id in self.sample_ids):
            raise ValueError("Every sample_id must be a non-empty string.")
        if len(set(self.sample_ids)) != len(self.sample_ids):
            raise ValueError("sample_ids must be unique within a batch.")
        if any(not isinstance(record, Mapping) for record in self.records):
            raise TypeError("Every metadata record must be a mapping.")


@runtime_checkable
class PocketField(Protocol):
    """Stable interface for optional pocket fields without choosing a layout.

    Concrete implementations must document every tensor they expose, including
    shape, dtype, device, and mask semantics. No grid, voxel, Gaussian, or
    equivariant representation is implied by this protocol.
    """

    @property
    def batch_size(self) -> int:
        """Return the represented batch size."""

    @property
    def device(self) -> torch.device:
        """Return the device used by field data."""

    def validate(self, *, batch_size: int, device: torch.device) -> None:
        """Raise a clear error when field data is incompatible with the batch."""


@dataclass
class Stage2Output:
    """Minimal Stage 2 output contract consumed by the Stage 3 adapter."""

    scaffold: MolecularGraph
    mask: BatchMask
    metadata: BatchMetadata

    @property
    def batch_size(self) -> int:
        return self.scaffold.batch_size

    def validate(self) -> None:
        self.scaffold.validate()
        self.mask.validate()
        batch_size, num_atoms, _ = self.scaffold.atom_features.shape
        num_edges = self.scaffold.edge_index.shape[2]
        if tuple(self.mask.atom_mask.shape) != (batch_size, num_atoms):
            raise ValueError("atom_mask shape must match scaffold [B, N].")
        if tuple(self.mask.edge_mask.shape) != (batch_size, num_edges):
            raise ValueError("edge_mask shape must match scaffold [B, E].")
        if self.mask.device != self.scaffold.atom_features.device:
            raise ValueError("Stage 2 masks and scaffold tensors must share a device.")
        self.metadata.validate(batch_size)
        _validate_edge_references(
            self.scaffold.edge_index,
            self.mask.atom_mask,
            self.mask.edge_mask,
        )


@dataclass
class Stage3Input:
    """Validated boundary object joining Stage 2, descriptors, and pocket data."""

    scaffold: MolecularGraph
    descriptor: DescriptorBatch
    pocket: PocketBatch
    mask: BatchMask
    metadata: BatchMetadata
    pocket_field: PocketField | None = None

    @property
    def batch_size(self) -> int:
        return self.scaffold.batch_size

    @property
    def device(self) -> torch.device:
        return self.scaffold.atom_features.device

    def validate(self) -> None:
        self.scaffold.validate()
        self.descriptor.validate()
        self.pocket.validate()
        self.mask.validate()
        batch_size, num_atoms, _ = self.scaffold.atom_features.shape
        num_edges = self.scaffold.edge_index.shape[2]
        descriptor_dim = self.descriptor.values.shape[1]
        num_pocket_points = self.pocket.coordinates.shape[1]

        if self.descriptor.batch_size != batch_size or self.pocket.batch_size != batch_size:
            raise ValueError("Scaffold, descriptor, and pocket batch sizes must match.")
        expected_shapes = {
            "atom_mask": ((batch_size, num_atoms), self.mask.atom_mask),
            "edge_mask": ((batch_size, num_edges), self.mask.edge_mask),
            "descriptor_mask": (
                (batch_size, descriptor_dim),
                self.mask.descriptor_mask,
            ),
            "pocket_mask": (
                (batch_size, num_pocket_points),
                self.mask.pocket_mask,
            ),
        }
        for name, (expected, actual) in expected_shapes.items():
            if actual is None or tuple(actual.shape) != expected:
                raise ValueError(f"{name} shape must be {list(expected)}.")

        tensors = {
            "descriptor.values": self.descriptor.values,
            "pocket.coordinates": self.pocket.coordinates,
            "pocket.features": self.pocket.features,
            "mask": self.mask.atom_mask,
        }
        for name, tensor in tensors.items():
            if tensor.device != self.device:
                raise ValueError(f"{name} must be on the scaffold device.")
        self.metadata.validate(batch_size)
        _validate_edge_references(
            self.scaffold.edge_index,
            self.mask.atom_mask,
            self.mask.edge_mask,
        )
        if self.pocket_field is not None:
            self.pocket_field.validate(batch_size=batch_size, device=self.device)


@dataclass
class Stage3DiffusionState:
    """Stage 3 state contract; this class implements no diffusion algorithm.

    Tensor fields:
        atom_features: Shape ``[B, N, F_atom]``; floating-point; model device.
            Rows with ``atom_mask=False`` are padding and must be ignored.
        edge_index: Shape ``[B, 2, E]``; ``torch.long``; model device. Entries
            with ``edge_mask=False`` are padding and ignored.
        edge_features: Shape ``[B, E, F_edge]``; floating-point; model device.
            Rows with ``edge_mask=False`` are padding.
        coordinates: Shape ``[B, N, 3]``; floating-point; model device. Rows
            with ``atom_mask=False`` are padding with unspecified values.
        descriptor: Shape ``[B, D]``; floating-point; model device. Values with
            ``descriptor_mask=False`` are missing/padding and must be ignored.
        timestep: Shape ``[B]``; ``torch.long``; model device. One non-negative
            discrete timestep is stored per sample.
    """

    atom_features: Tensor
    edge_index: Tensor
    edge_features: Tensor
    coordinates: Tensor
    descriptor: Tensor
    timestep: Tensor
    mask: BatchMask
    metadata: BatchMetadata

    @property
    def batch_size(self) -> int:
        return int(self.atom_features.shape[0])

    def validate(self) -> None:
        if self.coordinates is None:
            raise ValueError("Stage3DiffusionState coordinates are required.")
        graph = MolecularGraph(
            atom_features=self.atom_features,
            edge_index=self.edge_index,
            edge_features=self.edge_features,
            coordinates=self.coordinates,
        )
        graph.validate()
        descriptor_batch = DescriptorBatch(self.descriptor)
        descriptor_batch.validate()
        self.mask.validate()
        _require_rank("timestep", self.timestep, 1)
        if self.timestep.dtype != torch.long:
            raise TypeError(
                f"timestep must have dtype torch.long, got {self.timestep.dtype}."
            )
        if self.timestep.shape[0] != self.batch_size:
            raise ValueError("timestep must have shape [B].")
        if bool((self.timestep < 0).any()):
            raise ValueError("timestep values must be non-negative.")
        if descriptor_batch.batch_size != self.batch_size:
            raise ValueError("descriptor batch size must match graph batch size.")
        if self.timestep.device != self.atom_features.device:
            raise ValueError("timestep must be on the graph device.")
        if self.descriptor.device != self.atom_features.device:
            raise ValueError("descriptor must be on the graph device.")
        if tuple(self.mask.atom_mask.shape) != tuple(self.atom_features.shape[:2]):
            raise ValueError("atom_mask shape must match atom_features [B, N].")
        if tuple(self.mask.edge_mask.shape) != (
            self.batch_size,
            self.edge_features.shape[1],
        ):
            raise ValueError("edge_mask shape must match edge_features [B, E].")
        if self.mask.descriptor_mask is None or tuple(self.mask.descriptor_mask.shape) != tuple(self.descriptor.shape):
            raise ValueError("descriptor_mask shape must match descriptor [B, D].")
        if self.mask.device != self.atom_features.device:
            raise ValueError("State masks must be on the graph device.")
        self.metadata.validate(self.batch_size)
        _validate_edge_references(self.edge_index, self.mask.atom_mask, self.mask.edge_mask)
