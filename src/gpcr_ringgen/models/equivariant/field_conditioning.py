"""Field-query boundary and scalar conditioning for the Stage 3 denoiser."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from gpcr_ringgen.core.types import Stage3DiffusionState
from gpcr_ringgen.models.equivariant.backbone import (
    DenoiserOutput,
    EquivariantDenoiserBackbone,
)
from gpcr_ringgen.pocket.fields import FieldQueryResult, QueryablePocketField
from gpcr_ringgen.utils.io import load_yaml_config


@dataclass(frozen=True)
class FieldConditioningConfig:
    """Configuration for optional invariant scalar Field conditioning.

    Attributes:
        mode: ``"none"`` bypasses Field Query and exactly uses the base
            denoiser. ``"additive"`` queries the field at every valid ligand
            coordinate, encodes the configured scalar channels, and adds the
            encoding to the initial node hidden state.
        channels: Ordered Field channel names expected from the stable query
            API. Their order defines the input columns of the learned encoder.
            At least one channel is required for ``"additive"`` mode.
    """

    mode: str
    channels: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.mode, str):
            raise TypeError("field conditioning mode must be a string")
        if self.mode not in {"none", "additive"}:
            raise ValueError("field conditioning mode must be 'none' or 'additive'")
        if isinstance(self.channels, str) or not isinstance(self.channels, Sequence):
            raise TypeError("field conditioning channels must be a sequence")
        normalized_channels = tuple(self.channels)
        if any(not isinstance(name, str) or not name.strip() for name in normalized_channels):
            raise ValueError("field conditioning channel names must be non-empty strings")
        if len(set(normalized_channels)) != len(normalized_channels):
            raise ValueError("field conditioning channels must be unique")
        if self.mode == "additive" and not normalized_channels:
            raise ValueError("additive field conditioning requires at least one channel")
        object.__setattr__(self, "channels", normalized_channels)

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "FieldConditioningConfig":
        """Construct from a root mapping or ``field_conditioning`` section."""

        conditioning_config = config.get("field_conditioning", config)
        if not isinstance(conditioning_config, Mapping):
            raise TypeError("field_conditioning config must be a mapping")
        missing = [
            name for name in ("mode", "channels") if name not in conditioning_config
        ]
        if missing:
            raise KeyError(
                "missing field conditioning config keys: " + ", ".join(missing)
            )
        return cls(
            mode=conditioning_config["mode"],
            channels=conditioning_config["channels"],
        )


def load_field_conditioning_config(path: str | Path) -> FieldConditioningConfig:
    """Load a validated Field conditioning configuration from YAML."""

    return FieldConditioningConfig.from_mapping(load_yaml_config(path))


class FieldConditionedEquivariantDenoiser(nn.Module):
    """Wrap the base denoiser with optional Pocket Field scalar conditioning.

    In ``additive`` mode, :meth:`forward` evaluates ``F(X_t)`` through the
    existing :class:`QueryablePocketField` API. The result has shape
    ``[B, N, C_field]`` and is encoded to ``[B, N, hidden_dim]`` before being
    added to the backbone's scalar node state. Field coordinates and ligand
    coordinates must use the same local frame.

    Geometry-field channels depend only on relative pocket/query distances and
    are therefore invariant under joint rotations and translations. The
    backbone still generates vector updates exclusively from relative ligand
    displacements, preserving E(3) equivariance. Descriptor values remain a
    separate global scalar condition handled by the existing backbone. They
    are masked but not normalized here; upstream preprocessing owns descriptor
    normalization.
    """

    def __init__(
        self,
        backbone: EquivariantDenoiserBackbone,
        config: FieldConditioningConfig,
    ) -> None:
        super().__init__()
        if not isinstance(backbone, EquivariantDenoiserBackbone):
            raise TypeError("backbone must be an EquivariantDenoiserBackbone")
        if not isinstance(config, FieldConditioningConfig):
            raise TypeError("config must be a FieldConditioningConfig")
        self.backbone = backbone
        self.config = config
        if config.mode == "none":
            self.field_encoder = None
        else:
            backbone_parameter = next(backbone.parameters())
            self.field_encoder = nn.Sequential(
                nn.Linear(len(config.channels), backbone.config.hidden_dim),
                nn.SiLU(),
                nn.Linear(
                    backbone.config.hidden_dim,
                    backbone.config.hidden_dim,
                ),
            ).to(
                device=backbone_parameter.device,
                dtype=backbone_parameter.dtype,
            )

    def query_field(
        self,
        state: Stage3DiffusionState,
        pocket_field: QueryablePocketField,
    ) -> FieldQueryResult:
        """Query configured channels at the current ligand coordinates.

        Args:
            state: Stage 3 state whose coordinates have shape ``[B, N, 3]``;
                floating-point on the model device. ``atom_mask=False`` marks
                padding and is forwarded as the query mask.
            pocket_field: Existing queryable field with the same batch size,
                dtype, device, coordinate frame, and ordered configured channels.

        Returns:
            A validated result with values shaped ``[B, N, C_field]`` and
            boolean query mask shaped ``[B, N]``. Padded rows are zero.
        """

        if self.config.mode != "additive":
            raise RuntimeError("Field Query is disabled when conditioning mode is none")
        if not isinstance(state, Stage3DiffusionState):
            raise TypeError("state must be a Stage3DiffusionState")
        state.validate()
        if not isinstance(pocket_field, QueryablePocketField):
            raise TypeError("pocket_field must implement QueryablePocketField")
        pocket_field.validate(
            batch_size=state.batch_size,
            device=state.coordinates.device,
        )
        if tuple(pocket_field.channels) != self.config.channels:
            raise ValueError(
                "pocket field channels do not match configured order: "
                f"{tuple(pocket_field.channels)} versus {self.config.channels}"
            )
        result = pocket_field.query(
            state.coordinates,
            query_mask=state.mask.atom_mask,
        )
        result.validate()
        expected_shape = (
            state.batch_size,
            state.coordinates.shape[1],
            len(self.config.channels),
        )
        if tuple(result.values.shape) != expected_shape:
            raise ValueError("field query values must have shape [B, N, C_field]")
        if result.values.dtype != state.coordinates.dtype:
            raise TypeError("field query dtype must match state coordinates")
        if result.values.device != state.coordinates.device:
            raise ValueError("field query device must match state coordinates")
        if not torch.equal(result.query_mask, state.mask.atom_mask):
            raise ValueError("field query mask must match state atom_mask")
        return result

    def _encode_field(self, result: FieldQueryResult, atom_mask: Tensor) -> Tensor:
        if self.field_encoder is None:
            raise RuntimeError("field encoder is unavailable in none mode")
        encoded = self.field_encoder(result.values)
        return encoded * atom_mask.unsqueeze(-1).to(encoded.dtype)

    def forward(
        self,
        state: Stage3DiffusionState,
        timestep: Tensor | int | None = None,
        *,
        pocket_field: QueryablePocketField | None = None,
    ) -> DenoiserOutput:
        """Predict coordinate noise with optional Field conditioning.

        ``mode="none"`` does not require or query ``pocket_field`` and is
        exactly the base denoiser path. ``mode="additive"`` requires a field,
        queries ``F(X_t)``, applies the mask, and injects encoded scalar node
        conditions before equivariant message passing.
        """

        if self.config.mode == "none":
            return self.backbone(state, timestep)
        if pocket_field is None:
            raise ValueError("pocket_field is required for additive conditioning")
        field_result = self.query_field(state, pocket_field)
        node_condition = self._encode_field(field_result, state.mask.atom_mask)
        return self.backbone(
            state,
            timestep,
            node_condition=node_condition,
        )


__all__ = [
    "FieldConditionedEquivariantDenoiser",
    "FieldConditioningConfig",
    "load_field_conditioning_config",
]
