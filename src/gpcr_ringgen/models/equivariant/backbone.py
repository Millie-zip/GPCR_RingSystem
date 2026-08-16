"""Pocket-independent E(n)-equivariant Stage 3 denoiser backbone."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from gpcr_ringgen.core.types import Stage3DiffusionState
from gpcr_ringgen.models.embeddings import SinusoidalTimestepEmbedding
from gpcr_ringgen.models.equivariant.layers import EGNNMessageLayer
from gpcr_ringgen.utils.io import load_yaml_config


class EquivariantBackbone(ABC):
    """Shared extension point for equivariant model backbones."""

    @abstractmethod
    def forward(self, inputs: Any) -> Any:
        """Return transformed features."""


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _positive_finite(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return converted


@dataclass(frozen=True)
class EquivariantDenoiserConfig:
    """Dimensions and numerical constants for the base denoiser."""

    atom_feature_dim: int
    edge_feature_dim: int
    hidden_dim: int
    num_layers: int
    timestep_embedding_dim: int
    coordinate_scale: float
    epsilon: float
    max_time_period: float
    descriptor_dim: int | None = None
    num_atom_types: int | None = None
    num_bond_types: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "atom_feature_dim",
            "edge_feature_dim",
            "hidden_dim",
            "num_layers",
            "timestep_embedding_dim",
        ):
            object.__setattr__(self, name, _positive_int(name, getattr(self, name)))
        for name in ("coordinate_scale", "epsilon", "max_time_period"):
            object.__setattr__(
                self,
                name,
                _positive_finite(name, getattr(self, name)),
            )
        if self.timestep_embedding_dim < 2:
            raise ValueError("timestep_embedding_dim must be at least 2")
        if self.max_time_period <= 1.0:
            raise ValueError("max_time_period must be greater than 1")
        if self.descriptor_dim is not None:
            object.__setattr__(
                self,
                "descriptor_dim",
                _positive_int("descriptor_dim", self.descriptor_dim),
            )
        if (self.num_atom_types is None) != (self.num_bond_types is None):
            raise ValueError(
                "num_atom_types and num_bond_types must either both be set or absent"
            )
        for name in ("num_atom_types", "num_bond_types"):
            value = getattr(self, name)
            if value is not None:
                normalized = _positive_int(name, value)
                if normalized < 2:
                    raise ValueError(f"{name} must be at least 2")
                object.__setattr__(self, name, normalized)

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "EquivariantDenoiserConfig":
        model_config = config.get("model", config)
        if not isinstance(model_config, Mapping):
            raise TypeError("model config must be a mapping")
        model_type = model_config.get("type", "egnn_denoiser")
        if model_type != "egnn_denoiser":
            raise ValueError(
                f"unsupported model type {model_type!r}; expected 'egnn_denoiser'"
            )
        required = (
            "atom_feature_dim",
            "edge_feature_dim",
            "hidden_dim",
            "num_layers",
            "timestep_embedding_dim",
            "coordinate_scale",
            "epsilon",
            "max_time_period",
            "descriptor_dim",
        )
        missing = [name for name in required if name not in model_config]
        if missing:
            raise KeyError(f"missing denoiser config keys: {', '.join(missing)}")
        values = {name: model_config[name] for name in required}
        for name in ("num_atom_types", "num_bond_types"):
            if name in model_config:
                values[name] = model_config[name]
        return cls(**values)


def load_equivariant_denoiser_config(
    path: str | Path,
) -> EquivariantDenoiserConfig:
    """Load the base denoiser configuration through the project YAML loader."""

    return EquivariantDenoiserConfig.from_mapping(load_yaml_config(path))


@dataclass(frozen=True)
class DenoiserOutput:
    """Coordinate and optional fixed-topology chemical predictions.

    Tensor fields:
        coordinate_noise: Shape ``[B, N, 3]``; floating-point; same dtype and
            device as input coordinates. Rows with ``atom_mask=False`` are
            exactly zero and must be ignored.
    """

    coordinate_noise: Tensor
    atom_type_logits: Tensor | None = None
    bond_type_logits: Tensor | None = None

    def validate(self, state: Stage3DiffusionState) -> None:
        if not isinstance(self.coordinate_noise, Tensor):
            raise TypeError("coordinate_noise must be a torch.Tensor")
        if self.coordinate_noise.shape != state.coordinates.shape:
            raise ValueError("coordinate_noise must have shape [B, N, 3]")
        if self.coordinate_noise.dtype != state.coordinates.dtype:
            raise TypeError("coordinate_noise dtype must match state coordinates")
        if self.coordinate_noise.device != state.coordinates.device:
            raise ValueError("coordinate_noise device must match state coordinates")
        if not bool(torch.isfinite(self.coordinate_noise).all()):
            raise ValueError("coordinate_noise contains NaN or Inf")
        if bool(
            (
                self.coordinate_noise.masked_select(
                    ~state.mask.atom_mask.unsqueeze(-1)
                )
                != 0
            ).any()
        ):
            raise ValueError("coordinate_noise must be zero for padded atoms")
        if (self.atom_type_logits is None) != (self.bond_type_logits is None):
            raise ValueError(
                "atom_type_logits and bond_type_logits must both be present or absent"
            )
        if self.atom_type_logits is not None and self.bond_type_logits is not None:
            atom_logits = self.atom_type_logits
            bond_logits = self.bond_type_logits
            if atom_logits.ndim != 3 or tuple(atom_logits.shape[:2]) != tuple(
                state.atom_features.shape[:2]
            ):
                raise ValueError("atom_type_logits must have shape [B, N, C_atom]")
            if bond_logits.ndim != 3 or tuple(bond_logits.shape[:2]) != (
                state.batch_size,
                state.edge_features.shape[1],
            ):
                raise ValueError("bond_type_logits must have shape [B, E, C_bond]")
            if atom_logits.shape[-1] < 2 or bond_logits.shape[-1] < 2:
                raise ValueError("chemical logits must contain at least two classes")
            for name, values in (
                ("atom_type_logits", atom_logits),
                ("bond_type_logits", bond_logits),
            ):
                if not torch.is_floating_point(values):
                    raise TypeError(f"{name} must use a floating-point dtype")
                if values.dtype != state.coordinates.dtype:
                    raise TypeError(f"{name} dtype must match state coordinates")
                if values.device != state.coordinates.device:
                    raise ValueError(f"{name} device must match state coordinates")
                if not bool(torch.isfinite(values).all()):
                    raise ValueError(f"{name} contains NaN or Inf")
            if bool(
                (atom_logits.masked_select(~state.mask.atom_mask.unsqueeze(-1)) != 0).any()
            ):
                raise ValueError("atom_type_logits must be zero for padded atoms")
            if bool(
                (bond_logits.masked_select(~state.mask.edge_mask.unsqueeze(-1)) != 0).any()
            ):
                raise ValueError("bond_type_logits must be zero for padded edges")


class EquivariantDenoiserBackbone(nn.Module, EquivariantBackbone):
    """EGNN-style coordinate denoiser without pocket or diffusion logic.

    Atom, bond, timestep, and optional descriptor inputs are treated as scalar
    invariants. Absolute xyz values are never passed to an MLP. Coordinate
    updates are linear combinations of relative unit displacements with scalar
    invariant coefficients.
    """

    def __init__(self, config: EquivariantDenoiserConfig) -> None:
        super().__init__()
        self.config = config
        self.atom_projection = nn.Linear(config.atom_feature_dim, config.hidden_dim)
        self.timestep_embedding = SinusoidalTimestepEmbedding(
            config.timestep_embedding_dim,
            max_period=config.max_time_period,
        )
        self.timestep_projection = nn.Sequential(
            nn.Linear(config.timestep_embedding_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        self.descriptor_projection = (
            None
            if config.descriptor_dim is None
            else nn.Linear(config.descriptor_dim, config.hidden_dim)
        )
        if config.num_atom_types is None:
            self.atom_type_embedding = None
            self.bond_type_embedding = None
            self.atom_type_head = None
            self.bond_type_head = None
        else:
            if config.num_bond_types is None:
                raise ValueError("num_bond_types is required with num_atom_types")
            self.atom_type_embedding = nn.Embedding(
                config.num_atom_types,
                config.hidden_dim,
            )
            self.bond_type_embedding = nn.Embedding(
                config.num_bond_types,
                config.edge_feature_dim,
            )
            self.atom_type_head = nn.Linear(
                config.hidden_dim,
                config.num_atom_types,
            )
            self.bond_type_head = nn.Sequential(
                nn.Linear(
                    2 * config.hidden_dim + config.edge_feature_dim + 1,
                    config.hidden_dim,
                ),
                nn.SiLU(),
                nn.Linear(config.hidden_dim, config.num_bond_types),
            )
        self.layers = nn.ModuleList(
            EGNNMessageLayer(
                hidden_dim=config.hidden_dim,
                edge_feature_dim=config.edge_feature_dim,
                coordinate_scale=config.coordinate_scale,
                epsilon=config.epsilon,
            )
            for _ in range(config.num_layers)
        )

    def _normalize_timestep(
        self,
        state: Stage3DiffusionState,
        timestep: Tensor | int | None,
    ) -> Tensor:
        if timestep is None:
            normalized = state.timestep
        elif isinstance(timestep, bool):
            raise TypeError("timestep must be an integer or a torch.Tensor")
        elif isinstance(timestep, int):
            if timestep < 0:
                raise ValueError("timestep must be non-negative")
            normalized = torch.full(
                (state.batch_size,),
                timestep,
                dtype=torch.long,
                device=state.coordinates.device,
            )
        elif isinstance(timestep, Tensor):
            normalized = timestep
        else:
            raise TypeError("timestep must be an integer or a torch.Tensor")

        if normalized.ndim != 1 or normalized.shape[0] != state.batch_size:
            raise ValueError("timestep must have shape [B]")
        if normalized.device != state.coordinates.device:
            raise ValueError("timestep must be on the state device")
        if not bool(torch.isfinite(normalized).all()):
            raise ValueError("timestep contains NaN or Inf")
        if bool((normalized < 0).any()):
            raise ValueError("timestep must be non-negative")
        return normalized

    def _validate_state_dimensions(self, state: Stage3DiffusionState) -> None:
        if state.atom_features.shape[-1] != self.config.atom_feature_dim:
            raise ValueError("state atom feature dimension does not match model config")
        if state.edge_features.shape[-1] != self.config.edge_feature_dim:
            raise ValueError("state edge feature dimension does not match model config")
        if bool((state.mask.atom_mask.sum(dim=1) == 0).any()):
            raise ValueError("Every sample must contain at least one valid atom")
        parameter = next(self.parameters())
        if parameter.device != state.coordinates.device:
            raise ValueError("model and state must be on the same device")
        if parameter.dtype != state.coordinates.dtype:
            raise TypeError("model and state floating tensors must share a dtype")
        if self.config.descriptor_dim is not None:
            if state.descriptor.shape[-1] != self.config.descriptor_dim:
                raise ValueError(
                    "state descriptor dimension does not match model config"
                )
        if self.config.num_atom_types is not None:
            if state.atom_types is None or state.bond_types is None:
                raise ValueError(
                    "joint chemical denoiser requires state atom_types and bond_types"
                )
            valid_atom_types = state.atom_types.masked_select(state.mask.atom_mask)
            if bool((valid_atom_types >= self.config.num_atom_types).any()):
                raise ValueError("state atom_types exceed the configured vocabulary")
            if self.config.num_bond_types is None:
                raise ValueError("num_bond_types is required for chemical denoising")
            valid_bond_types = state.bond_types.masked_select(state.mask.edge_mask)
            if bool((valid_bond_types >= self.config.num_bond_types).any()):
                raise ValueError("state bond_types exceed the configured vocabulary")

    def forward(
        self,
        state: Stage3DiffusionState,
        timestep: Tensor | int | None = None,
        *,
        node_condition: Tensor | None = None,
    ) -> DenoiserOutput:
        """Predict centered coordinate noise from graph and scalar conditions.

        ``node_condition`` is an optional generic scalar-invariant extension
        point with shape ``[B, N, hidden_dim]``. It must already be encoded on
        the state dtype/device; rows where ``atom_mask=False`` are ignored. The
        base backbone does not construct or depend on Pocket Field data.
        """

        if not isinstance(state, Stage3DiffusionState):
            raise TypeError("state must be a Stage3DiffusionState")
        state.validate()
        self._validate_state_dimensions(state)
        normalized_timestep = self._normalize_timestep(state, timestep)

        atom_mask = state.mask.atom_mask
        node_features = self.atom_projection(state.atom_features)
        edge_features = state.edge_features
        if self.atom_type_embedding is not None:
            if state.atom_types is None or state.bond_types is None:
                raise ValueError("joint chemical state labels are required")
            if self.bond_type_embedding is None:
                raise RuntimeError("bond type embedding is unavailable")
            node_features = node_features + self.atom_type_embedding(state.atom_types)
            edge_features = edge_features + self.bond_type_embedding(state.bond_types)
        time_condition = self.timestep_projection(
            self.timestep_embedding(normalized_timestep)
        )
        node_features = node_features + time_condition.unsqueeze(1)
        if node_condition is not None:
            if not isinstance(node_condition, Tensor):
                raise TypeError("node_condition must be a torch.Tensor")
            expected_shape = (
                state.batch_size,
                state.atom_features.shape[1],
                self.config.hidden_dim,
            )
            if tuple(node_condition.shape) != expected_shape:
                raise ValueError(
                    "node_condition must have shape [B, N, hidden_dim]"
                )
            if node_condition.dtype != state.coordinates.dtype:
                raise TypeError("node_condition dtype must match state coordinates")
            if node_condition.device != state.coordinates.device:
                raise ValueError("node_condition device must match state coordinates")
            if not bool(torch.isfinite(node_condition).all()):
                raise ValueError("node_condition contains NaN or Inf")
            node_features = node_features + node_condition
        if self.descriptor_projection is not None:
            descriptor_mask = state.mask.descriptor_mask
            if descriptor_mask is None:
                raise ValueError("descriptor_mask is required for descriptor conditioning")
            masked_descriptor = state.descriptor * descriptor_mask.to(
                state.descriptor.dtype
            )
            node_features = node_features + self.descriptor_projection(
                masked_descriptor
            ).unsqueeze(1)
        node_features = node_features * atom_mask.unsqueeze(-1).to(
            node_features.dtype
        )

        initial_coordinates = state.coordinates
        updated_coordinates = initial_coordinates
        for layer in self.layers:
            node_features, updated_coordinates = layer(
                node_features,
                updated_coordinates,
                state.edge_index,
                edge_features,
                atom_mask,
                state.mask.edge_mask,
            )

        coordinate_mask = atom_mask.unsqueeze(-1).to(initial_coordinates.dtype)
        coordinate_noise = (updated_coordinates - initial_coordinates) * coordinate_mask
        valid_counts = coordinate_mask.sum(dim=1, keepdim=True)
        prediction_center = (
            coordinate_noise.sum(dim=1, keepdim=True) / valid_counts
        )
        coordinate_noise = (coordinate_noise - prediction_center) * coordinate_mask
        atom_type_logits: Tensor | None = None
        bond_type_logits: Tensor | None = None
        if self.atom_type_head is not None:
            if self.bond_type_head is None:
                raise RuntimeError("bond type head is unavailable")
            atom_type_logits = self.atom_type_head(node_features)
            atom_type_logits = atom_type_logits * atom_mask.unsqueeze(-1).to(
                atom_type_logits.dtype
            )
            batch_size, num_atoms, _ = node_features.shape
            num_edges = state.edge_index.shape[-1]
            batch_index = torch.arange(
                batch_size,
                device=state.edge_index.device,
            ).unsqueeze(1).expand(batch_size, num_edges)
            source = state.edge_index[:, 0, :].clamp(min=0, max=num_atoms - 1)
            target = state.edge_index[:, 1, :].clamp(min=0, max=num_atoms - 1)
            source_hidden = node_features[batch_index, source]
            target_hidden = node_features[batch_index, target]
            relative = updated_coordinates[batch_index, target] - updated_coordinates[
                batch_index, source
            ]
            squared_distance = torch.sum(relative.square(), dim=-1, keepdim=True)
            bond_type_logits = self.bond_type_head(
                torch.cat(
                    (
                        source_hidden + target_hidden,
                        torch.abs(source_hidden - target_hidden),
                        edge_features,
                        squared_distance,
                    ),
                    dim=-1,
                )
            )
            bond_type_logits = bond_type_logits * state.mask.edge_mask.unsqueeze(
                -1
            ).to(bond_type_logits.dtype)
        output = DenoiserOutput(
            coordinate_noise=coordinate_noise,
            atom_type_logits=atom_type_logits,
            bond_type_logits=bond_type_logits,
        )
        output.validate(state)
        return output


__all__ = [
    "DenoiserOutput",
    "EquivariantBackbone",
    "EquivariantDenoiserBackbone",
    "EquivariantDenoiserConfig",
    "load_equivariant_denoiser_config",
]
