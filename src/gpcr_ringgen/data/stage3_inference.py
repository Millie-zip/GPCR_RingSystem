"""Standard JSON input adapter for independent Stage 3 inference."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from gpcr_ringgen.core.exceptions import ConfigurationError
from gpcr_ringgen.core.types import (
    BatchMask,
    BatchMetadata,
    DescriptorBatch,
    MolecularGraph,
    PocketBatch,
    Stage2Output,
)
from gpcr_ringgen.pocket import PocketAtomMetadata


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{name} must be a JSON object")
    return value


def _sequence(value: object, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ConfigurationError(f"{name} must be a JSON array")
    return value


def _string_tuple(value: object, name: str) -> tuple[str, ...]:
    values = tuple(_sequence(value, name))
    if any(not isinstance(item, str) for item in values):
        raise ConfigurationError(f"every {name} entry must be a string")
    return values


def _float_tensor(
    value: object,
    name: str,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    try:
        tensor = torch.tensor(value, dtype=dtype, device=device)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise ConfigurationError(f"{name} must contain numeric values") from exc
    if not bool(torch.isfinite(tensor).all()):
        raise ConfigurationError(f"{name} contains NaN or Inf")
    return tensor


def _bool_tensor(
    value: object | None,
    name: str,
    *,
    size: int,
    device: torch.device,
) -> Tensor:
    if value is None:
        return torch.ones((1, size), dtype=torch.bool, device=device)
    values = tuple(_sequence(value, name))
    if any(not isinstance(item, bool) for item in values):
        raise ConfigurationError(f"every {name} entry must be true or false")
    tensor = torch.tensor(values, dtype=torch.bool, device=device)
    if tensor.ndim != 1 or tensor.shape[0] != size:
        raise ConfigurationError(f"{name} must have shape [{size}]")
    return tensor.unsqueeze(0)


def _long_tensor(
    value: object,
    name: str,
    *,
    device: torch.device,
) -> Tensor:
    try:
        tensor = torch.tensor(value, device=device)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise ConfigurationError(f"{name} must contain integer values") from exc
    if tensor.dtype == torch.bool or torch.is_floating_point(tensor):
        raise ConfigurationError(f"{name} must contain integer values")
    return tensor.to(dtype=torch.long)


@dataclass(frozen=True)
class Stage3InferenceRequest:
    """Single-sample standard input before pocket preprocessing.

    Tensor fields:
        stage2_output.scaffold.atom_features: Shape ``[1, N, F_atom]``;
            floating-point on the configured device. ``atom_mask=False`` is
            scaffold padding.
        stage2_output.scaffold.edge_index: Shape ``[1, 2, E]``;
            ``torch.long``. ``edge_mask=False`` is edge padding.
        stage2_output.scaffold.edge_features: Shape ``[1, E, F_edge]``;
            same floating dtype/device as atom features.
        descriptor.values: Shape ``[1, D]``; same floating dtype/device.
            ``descriptor_mask`` shape ``[1, D]`` uses ``True`` for present data.
        pocket.coordinates: Shape ``[1, P, 3]`` global coordinates;
            floating-point on the configured device. ``pocket_mask=False`` is
            pocket padding. Pocket features are ``[1, P, F_pocket]``.

    This object contains no generated Stage 1/Stage 2 model output. Its fixed
    scaffold is adapted directly to the existing Stage 2 contract.
    """

    stage2_output: Stage2Output
    descriptor: DescriptorBatch
    descriptor_mask: Tensor
    pocket: PocketBatch
    pocket_mask: Tensor
    pocket_atom_metadata: tuple[PocketAtomMetadata, ...]

    def validate(self) -> None:
        """Validate all standard-input tensors and sample alignment."""

        self.stage2_output.validate()
        self.descriptor.validate()
        self.pocket.validate()
        batch_size = self.stage2_output.batch_size
        if batch_size != 1:
            raise ValueError("standard Stage 3 inference JSON must contain one sample")
        if self.descriptor.batch_size != batch_size or self.pocket.batch_size != batch_size:
            raise ValueError("scaffold, descriptor, and pocket batch sizes must match")
        if tuple(self.descriptor_mask.shape) != tuple(self.descriptor.values.shape):
            raise ValueError("descriptor_mask must match descriptor values [1, D]")
        if self.descriptor_mask.dtype != torch.bool:
            raise TypeError("descriptor_mask must have dtype torch.bool")
        if tuple(self.pocket_mask.shape) != tuple(self.pocket.coordinates.shape[:2]):
            raise ValueError("pocket_mask must match pocket coordinates [1, P]")
        if self.pocket_mask.dtype != torch.bool:
            raise TypeError("pocket_mask must have dtype torch.bool")
        device = self.stage2_output.scaffold.atom_features.device
        dtype = self.stage2_output.scaffold.atom_features.dtype
        for name, tensor in (
            ("descriptor", self.descriptor.values),
            ("descriptor_mask", self.descriptor_mask),
            ("pocket coordinates", self.pocket.coordinates),
            ("pocket features", self.pocket.features),
            ("pocket_mask", self.pocket_mask),
        ):
            if tensor.device != device:
                raise ValueError(f"{name} must be on the scaffold device")
        for name, tensor in (
            ("descriptor", self.descriptor.values),
            ("pocket coordinates", self.pocket.coordinates),
            ("pocket features", self.pocket.features),
        ):
            if tensor.dtype != dtype:
                raise TypeError(f"{name} must use the scaffold floating dtype")
        if bool((self.pocket_mask.sum(dim=1) == 0).any()):
            raise ValueError("pocket must contain at least one valid atom")
        if len(self.pocket_atom_metadata) != batch_size:
            raise ValueError("pocket_atom_metadata length must equal batch size")
        for metadata, mask in zip(
            self.pocket_atom_metadata,
            self.pocket_mask,
            strict=True,
        ):
            metadata.validate(self.pocket.coordinates.shape[1], mask)


def load_stage3_inference_request(
    path: str | Path,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> Stage3InferenceRequest:
    """Load one pocket, fixed scaffold, and descriptor from standard JSON.

    The JSON stores unbatched arrays. This adapter adds a leading batch axis and
    builds existing project data contracts; it performs no pocket centering,
    Field construction, descriptor calculation, or sampling.
    """

    input_path = Path(path)
    if not input_path.is_file():
        raise ConfigurationError(f"Stage 3 inference input not found: {input_path}")
    try:
        raw = json.loads(input_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigurationError(f"Invalid JSON in {input_path}: {exc}") from exc
    root = _mapping(raw, "input root")
    sample_id = root.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id.strip():
        raise ConfigurationError("sample_id must be a non-empty string")
    metadata_value = root.get("metadata", {})
    metadata_record = _mapping(metadata_value, "metadata")
    batch_metadata = BatchMetadata(
        sample_ids=(sample_id,),
        records=(metadata_record,),
    )

    scaffold = _mapping(root.get("scaffold"), "scaffold")
    atom_features = _float_tensor(
        scaffold.get("atom_features"),
        "scaffold.atom_features",
        dtype=dtype,
        device=device,
    )
    edge_index = _long_tensor(
        scaffold.get("edge_index"),
        "scaffold.edge_index",
        device=device,
    )
    edge_features = _float_tensor(
        scaffold.get("edge_features"),
        "scaffold.edge_features",
        dtype=dtype,
        device=device,
    )
    if atom_features.ndim != 2:
        raise ConfigurationError("scaffold.atom_features must have shape [N, F_atom]")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ConfigurationError("scaffold.edge_index must have shape [2, E]")
    if edge_features.ndim != 2:
        raise ConfigurationError("scaffold.edge_features must have shape [E, F_edge]")
    atom_mask = _bool_tensor(
        scaffold.get("atom_mask"),
        "scaffold.atom_mask",
        size=atom_features.shape[0],
        device=device,
    )
    edge_mask = _bool_tensor(
        scaffold.get("edge_mask"),
        "scaffold.edge_mask",
        size=edge_index.shape[1],
        device=device,
    )
    stage2_output = Stage2Output(
        scaffold=MolecularGraph(
            atom_features=atom_features.unsqueeze(0),
            edge_index=edge_index.unsqueeze(0),
            edge_features=edge_features.unsqueeze(0),
        ),
        mask=BatchMask(atom_mask=atom_mask, edge_mask=edge_mask),
        metadata=batch_metadata,
    )

    descriptor_mapping = _mapping(root.get("descriptor"), "descriptor")
    descriptor_values = _float_tensor(
        descriptor_mapping.get("values"),
        "descriptor.values",
        dtype=dtype,
        device=device,
    )
    if descriptor_values.ndim != 1:
        raise ConfigurationError("descriptor.values must have shape [D]")
    descriptor = DescriptorBatch(descriptor_values.unsqueeze(0))
    descriptor_mask = _bool_tensor(
        descriptor_mapping.get("mask"),
        "descriptor.mask",
        size=descriptor_values.shape[0],
        device=device,
    )

    pocket_mapping = _mapping(root.get("pocket"), "pocket")
    pocket_coordinates = _float_tensor(
        pocket_mapping.get("coordinates"),
        "pocket.coordinates",
        dtype=dtype,
        device=device,
    )
    pocket_features = _float_tensor(
        pocket_mapping.get("features"),
        "pocket.features",
        dtype=dtype,
        device=device,
    )
    if pocket_coordinates.ndim != 2 or pocket_coordinates.shape[-1] != 3:
        raise ConfigurationError("pocket.coordinates must have shape [P, 3]")
    if pocket_features.ndim != 2:
        raise ConfigurationError("pocket.features must have shape [P, F_pocket]")
    num_pocket_atoms = pocket_coordinates.shape[0]
    pocket_mask = _bool_tensor(
        pocket_mapping.get("mask"),
        "pocket.mask",
        size=num_pocket_atoms,
        device=device,
    )
    atom_types = _string_tuple(pocket_mapping.get("atom_types"), "pocket.atom_types")
    elements = _string_tuple(pocket_mapping.get("elements"), "pocket.elements")
    residue_names = _string_tuple(
        pocket_mapping.get("residue_names"),
        "pocket.residue_names",
    )
    residue_ids = _string_tuple(
        pocket_mapping.get("residue_ids"),
        "pocket.residue_ids",
    )
    chain_value = pocket_mapping.get("chain_ids")
    chain_ids = (
        None
        if chain_value is None
        else _string_tuple(chain_value, "pocket.chain_ids")
    )
    pocket_atom_metadata = (
        PocketAtomMetadata(
            atom_types=atom_types,
            elements=elements,
            residue_names=residue_names,
            residue_ids=residue_ids,
            chain_ids=chain_ids,
        ),
    )
    request = Stage3InferenceRequest(
        stage2_output=stage2_output,
        descriptor=descriptor,
        descriptor_mask=descriptor_mask,
        pocket=PocketBatch(
            coordinates=pocket_coordinates.unsqueeze(0),
            features=pocket_features.unsqueeze(0),
        ),
        pocket_mask=pocket_mask,
        pocket_atom_metadata=pocket_atom_metadata,
    )
    request.validate()
    return request


__all__ = ["Stage3InferenceRequest", "load_stage3_inference_request"]
