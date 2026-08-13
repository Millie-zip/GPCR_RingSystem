"""Stable Pocket Field API and a basic vectorized geometry implementation.

The public query protocol is intentionally independent of the Gaussian kernel.
Future Stage 3 components should depend on :class:`QueryablePocketField`, not on
the concrete implementation in this module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable

import torch
from torch import Tensor

from gpcr_ringgen.core.interfaces import PocketFieldProvider
from gpcr_ringgen.core.types import PocketField
from gpcr_ringgen.pocket.representation import ProcessedPocketBatch
from gpcr_ringgen.utils.io import load_yaml_config


def _require_positive_finite(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0.0:
        raise ValueError(f"{name} must be finite and greater than zero, got {value!r}")
    return converted


def _require_bool(name: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool, got {type(value).__name__}")
    return value


@dataclass(frozen=True)
class GeometryFieldConfig:
    """Configuration for the basic pocket geometry field.

    Attributes:
        sigma: Gaussian width in the same length unit as the pocket coordinates.
        cutoff: Maximum atom-query distance contributing to occupancy. The
            nearest-distance channel is not clipped by this value.
        epsilon: Positive lower bound used for numerical denominators.
        occupancy_enabled: Whether to emit the Gaussian occupancy channel.
        nearest_distance_enabled: Whether to emit the nearest-pocket-distance
            channel.
    """

    sigma: float
    cutoff: float
    epsilon: float
    occupancy_enabled: bool
    nearest_distance_enabled: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "sigma", _require_positive_finite("sigma", self.sigma))
        object.__setattr__(self, "cutoff", _require_positive_finite("cutoff", self.cutoff))
        object.__setattr__(
            self, "epsilon", _require_positive_finite("epsilon", self.epsilon)
        )
        object.__setattr__(
            self,
            "occupancy_enabled",
            _require_bool("occupancy_enabled", self.occupancy_enabled),
        )
        object.__setattr__(
            self,
            "nearest_distance_enabled",
            _require_bool("nearest_distance_enabled", self.nearest_distance_enabled),
        )
        if not (self.occupancy_enabled or self.nearest_distance_enabled):
            raise ValueError("at least one geometry field channel must be enabled")

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "GeometryFieldConfig":
        """Construct from the ``field`` section of a YAML-style mapping."""

        field_config: Mapping[str, Any]
        if "field" in config:
            nested = config["field"]
            if not isinstance(nested, Mapping):
                raise TypeError("field config must be a mapping")
            field_config = nested
        else:
            field_config = config

        field_type = field_config.get("type", "gaussian_geometry")
        if field_type != "gaussian_geometry":
            raise ValueError(
                "unsupported geometry field type "
                f"{field_type!r}; expected 'gaussian_geometry'"
            )

        channels = field_config.get("channels")
        if not isinstance(channels, Mapping):
            raise TypeError("field.channels must be a mapping")

        missing = [
            key
            for key in ("sigma", "cutoff", "epsilon")
            if key not in field_config
        ]
        if missing:
            raise KeyError(f"missing geometry field config keys: {', '.join(missing)}")

        missing_channels = [
            key
            for key in ("occupancy", "nearest_distance")
            if key not in channels
        ]
        if missing_channels:
            raise KeyError(
                "missing geometry field channel config keys: "
                + ", ".join(missing_channels)
            )

        return cls(
            sigma=field_config["sigma"],
            cutoff=field_config["cutoff"],
            epsilon=field_config["epsilon"],
            occupancy_enabled=channels["occupancy"],
            nearest_distance_enabled=channels["nearest_distance"],
        )


def load_geometry_field_config(path: str | Path) -> GeometryFieldConfig:
    """Load and validate a geometry field configuration from YAML."""

    raw_config = load_yaml_config(path)
    if not isinstance(raw_config, Mapping):
        raise TypeError("geometry field config root must be a mapping")
    return GeometryFieldConfig.from_mapping(raw_config)


@dataclass(frozen=True)
class FieldChannel:
    """Stable, serializable definition of one field feature channel.

    Attributes:
        name: Unique machine-readable channel name. Its position in the channel
            tuple is the corresponding index in the last output dimension.
        description: Human-readable mathematical/semantic definition.
        unit: Output unit without assuming a specific pocket coordinate unit.
    """

    name: str
    description: str
    unit: str

    def __post_init__(self) -> None:
        for attribute_name in ("name", "description", "unit"):
            value = getattr(self, attribute_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"field channel {attribute_name} must be non-empty")

    def to_metadata(self) -> dict[str, str]:
        """Return a JSON/YAML-compatible channel metadata record."""

        return {
            "name": self.name,
            "description": self.description,
            "unit": self.unit,
        }


@dataclass(frozen=True)
class FieldQueryResult:
    """Field features returned by the stable single/batched query API.

    Attributes:
        values: Shape ``[Q, C]`` for an unbatched single-pocket query or
            ``[B, Q, C]`` for a padded batch. Floating-point on the field
            device, with dtype matching the pocket coordinates. ``C`` follows
            ``channels``. Values at invalid query positions are zero.
        query_mask: Shape ``[Q]`` for a single-pocket query or ``[B, Q]`` for a
            padded batch; ``torch.bool`` on the field device. ``True`` marks a
            valid query point and ``False`` marks padding whose values must be
            ignored. Empty ``Q=0`` tensors are valid.
        channels: Ordered channel names corresponding to ``values[..., c]``.
        channel_metadata: Ordered definitions corresponding one-to-one with
            ``channels`` and the final output dimension.
    """

    values: Tensor
    query_mask: Tensor
    channels: tuple[str, ...]
    channel_metadata: tuple[FieldChannel, ...]

    @property
    def is_batched(self) -> bool:
        """Whether this result uses the padded ``[B, Q, C]`` representation."""

        return self.values.ndim == 3

    @property
    def num_channels(self) -> int:
        """Return ``C``, including for an empty query point dimension."""

        return len(self.channels)

    def channel_index(self, name: str) -> int:
        """Return the output index for ``name`` without downstream guessing."""

        try:
            return self.channels.index(name)
        except ValueError as exc:
            raise KeyError(
                f"unknown field channel {name!r}; available channels: {self.channels}"
            ) from exc

    def select_channel(self, name: str) -> Tensor:
        """Select one channel as shape ``[Q]`` or ``[B, Q]``."""

        return self.values[..., self.channel_index(name)]

    def validate(self) -> None:
        """Validate query result shape, dtype, device, mask, and finiteness."""

        if self.values.ndim not in (2, 3):
            raise ValueError(
                "values must have shape [Q, C] or [B, Q, C], "
                f"got {tuple(self.values.shape)}"
            )
        if not self.values.is_floating_point():
            raise TypeError(f"values must have floating dtype, got {self.values.dtype}")
        expected_mask_rank = self.values.ndim - 1
        if self.query_mask.ndim != expected_mask_rank:
            raise ValueError(
                "query_mask must have shape [Q] or [B, Q] matching values, "
                f"got {tuple(self.query_mask.shape)}"
            )
        if self.query_mask.dtype != torch.bool:
            raise TypeError(
                f"query_mask must have dtype torch.bool, got {self.query_mask.dtype}"
            )
        if self.query_mask.shape != self.values.shape[:-1]:
            raise ValueError(
                "query_mask shape must match the query dimensions of values, got "
                f"{tuple(self.query_mask.shape)} versus {tuple(self.values.shape[:-1])}"
            )
        if self.query_mask.device != self.values.device:
            raise ValueError("query_mask and values must be on the same device")
        if self.values.shape[-1] != len(self.channels):
            raise ValueError(
                "values channel dimension must match channels, got "
                f"{self.values.shape[-1]} versus {len(self.channels)}"
            )
        if len(set(self.channels)) != len(self.channels):
            raise ValueError("channels must contain unique names")
        if len(self.channel_metadata) != len(self.channels):
            raise ValueError("channel_metadata length must match channels")
        metadata_names = tuple(channel.name for channel in self.channel_metadata)
        if metadata_names != self.channels:
            raise ValueError(
                "channel_metadata names and order must exactly match channels"
            )
        if not torch.isfinite(self.values).all():
            raise ValueError("field values contain NaN or Inf")
        if torch.any(self.values.masked_select(~self.query_mask.unsqueeze(-1)) != 0):
            raise ValueError("field values at masked query positions must be zero")


@runtime_checkable
class QueryablePocketField(PocketField, Protocol):
    """Stable Stage 3-facing API for querying a prebuilt pocket field."""

    @property
    def channels(self) -> tuple[str, ...]:
        """Ordered names of the emitted field channels."""

        ...

    @property
    def channel_metadata(self) -> tuple[FieldChannel, ...]:
        """Ordered, serializable semantic definitions for the channels."""

        ...

    def query(
        self,
        query_points: Tensor,
        query_mask: Tensor | None = None,
    ) -> FieldQueryResult:
        """Query local coordinates.

        Args:
            query_points: Shape ``[Q, 3]`` for a field with ``B=1``, or padded
                shape ``[B, Q, 3]``. Floating tensor using the same dtype,
                device, length unit, and local coordinate frame as the pocket.
                ``Q=0`` is valid.
            query_mask: Optional shape ``[Q]`` or ``[B, Q]`` boolean tensor on
                the same device, matching ``query_points``. ``True`` marks
                valid query points; ``False`` marks padding. If omitted, every
                query point is valid.
        """

        ...


class GaussianGeometryField:
    """Vectorized Gaussian occupancy and nearest-distance pocket field.

    The implementation consumes the Stage 2 pocket preprocessing result
    directly. It performs no parsing, coordinate transformation, voxelization,
    or learned field construction.
    """

    def __init__(
        self,
        processed_pocket: ProcessedPocketBatch,
        config: GeometryFieldConfig,
    ) -> None:
        processed_pocket.validate()
        self._processed_pocket = processed_pocket
        self._config = config
        self.validate()

    @property
    def batch_size(self) -> int:
        return self._processed_pocket.batch_size

    @property
    def device(self) -> torch.device:
        return self._processed_pocket.pocket.coordinates.device

    @property
    def dtype(self) -> torch.dtype:
        return self._processed_pocket.pocket.coordinates.dtype

    @property
    def channel_metadata(self) -> tuple[FieldChannel, ...]:
        channel_definitions: list[FieldChannel] = []
        if self._config.occupancy_enabled:
            channel_definitions.append(
                FieldChannel(
                    name="occupancy",
                    description=(
                        "Cutoff-truncated Gaussian density summed over valid "
                        "pocket atoms."
                    ),
                    unit="dimensionless",
                )
            )
        if self._config.nearest_distance_enabled:
            channel_definitions.append(
                FieldChannel(
                    name="nearest_distance",
                    description="Euclidean distance to the nearest valid pocket atom.",
                    unit="pocket_coordinate_unit",
                )
            )
        return tuple(channel_definitions)

    @property
    def channels(self) -> tuple[str, ...]:
        return tuple(channel.name for channel in self.channel_metadata)

    def validate(
        self,
        *,
        batch_size: int | None = None,
        device: torch.device | None = None,
    ) -> None:
        """Validate field data and optional Stage 3 batch compatibility."""

        self._processed_pocket.validate()
        if not self.channels:
            raise ValueError("at least one geometry field channel must be enabled")
        if batch_size is not None and batch_size != self.batch_size:
            raise ValueError(
                "field batch size is incompatible with the requested batch, got "
                f"{self.batch_size} versus {batch_size}"
            )
        if device is not None and device != self.device:
            raise ValueError(
                "field device is incompatible with the requested device, got "
                f"{self.device} versus {device}"
            )

    def _normalize_query(
        self,
        query_points: Tensor,
        query_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor, bool]:
        if not isinstance(query_points, Tensor):
            raise TypeError("query_points must be a torch.Tensor")
        if query_points.ndim not in (2, 3) or query_points.shape[-1] != 3:
            raise ValueError(
                "query_points must have shape [Q, 3] or [B, Q, 3], "
                f"got {tuple(query_points.shape)}"
            )
        is_single = query_points.ndim == 2
        if is_single and self.batch_size != 1:
            raise ValueError(
                "unbatched query_points [Q, 3] require a field with batch size 1"
            )
        query_batch_size = 1 if is_single else query_points.shape[0]
        if query_batch_size != self.batch_size:
            raise ValueError(
                "query_points batch size must match pocket batch size, got "
                f"{query_batch_size} versus {self.batch_size}"
            )
        if not query_points.is_floating_point():
            raise TypeError(
                f"query_points must have floating dtype, got {query_points.dtype}"
            )
        if query_points.dtype != self.dtype:
            raise TypeError(
                "query_points dtype must match pocket coordinates, got "
                f"{query_points.dtype} versus {self.dtype}"
            )
        if query_points.device != self.device:
            raise ValueError("query_points and pocket coordinates must be on the same device")
        if not torch.isfinite(query_points).all():
            raise ValueError("query_points contain NaN or Inf")

        expected_mask_shape = query_points.shape[:-1]
        if query_mask is None:
            validated_query_mask = torch.ones(
                expected_mask_shape,
                dtype=torch.bool,
                device=query_points.device,
            )
        else:
            if not isinstance(query_mask, Tensor):
                raise TypeError("query_mask must be a torch.Tensor")
            if query_mask.shape != expected_mask_shape:
                raise ValueError(
                    "query_mask shape must match query_points without the coordinate "
                    f"dimension, got {tuple(query_mask.shape)} versus "
                    f"{tuple(expected_mask_shape)}"
                )
            if query_mask.dtype != torch.bool:
                raise TypeError(
                    f"query_mask must have dtype torch.bool, got {query_mask.dtype}"
                )
            if query_mask.device != query_points.device:
                raise ValueError("query_mask and query_points must be on the same device")
            validated_query_mask = query_mask

        if is_single:
            return (
                query_points.unsqueeze(0),
                validated_query_mask.unsqueeze(0),
                True,
            )
        return query_points, validated_query_mask, False

    def _query_batched(self, query_points: Tensor, query_mask: Tensor) -> Tensor:
        """Evaluate padded ``[B, Q, 3]`` points without Python atom loops.

        Keeping the pairwise ``O(B * Q * P)`` allocation behind this internal
        boundary permits future chunking without changing the public API.
        """

        coordinates = self._processed_pocket.pocket.coordinates
        pocket_mask = self._processed_pocket.pocket_mask

        # Clamping the differences before squaring prevents overflow for finite
        # but extremely distant float32/float64 query points.
        max_component = math.sqrt(torch.finfo(self.dtype).max / 3.0)
        delta = torch.clamp(
            query_points.unsqueeze(2) - coordinates.unsqueeze(1),
            min=-max_component,
            max=max_component,
        )
        squared_distance = torch.sum(delta * delta, dim=-1)
        squared_distance = torch.clamp(
            squared_distance, max=torch.finfo(self.dtype).max
        )
        valid_atoms = pocket_mask.unsqueeze(1)

        output_channels: list[Tensor] = []
        if self._config.occupancy_enabled:
            variance = max(self._config.sigma**2, self._config.epsilon)
            cutoff_squared = self._config.cutoff**2
            within_cutoff = squared_distance <= cutoff_squared
            occupancy = torch.exp(-squared_distance / (2.0 * variance))
            occupancy = torch.where(valid_atoms & within_cutoff, occupancy, 0.0)
            output_channels.append(torch.sum(occupancy, dim=-1))

        if self._config.nearest_distance_enabled:
            masked_squared_distance = torch.where(
                valid_atoms,
                squared_distance,
                torch.full_like(squared_distance, torch.inf),
            )
            nearest_distance = torch.sqrt(torch.amin(masked_squared_distance, dim=-1))
            output_channels.append(nearest_distance)

        values = torch.stack(output_channels, dim=-1)
        return torch.where(query_mask.unsqueeze(-1), values, 0.0)

    def query(
        self,
        query_points: Tensor,
        query_mask: Tensor | None = None,
    ) -> FieldQueryResult:
        """Evaluate enabled channels for one pocket or a padded pocket batch.

        The pairwise computation is fully vectorized over batch, queries, and
        pocket atoms. Invalid pocket atoms never contribute; invalid query
        positions are returned as zeros. Empty query tensors return an empty
        feature tensor with the correct channel dimension.
        """

        batched_points, batched_mask, is_single = self._normalize_query(
            query_points,
            query_mask,
        )
        values = self._query_batched(batched_points, batched_mask)
        result_values = values[0] if is_single else values
        result_mask = batched_mask[0] if is_single else batched_mask
        result = FieldQueryResult(
            values=result_values,
            query_mask=result_mask,
            channels=self.channels,
            channel_metadata=self.channel_metadata,
        )
        result.validate()
        return result


def build_geometry_field(
    processed_pocket: ProcessedPocketBatch,
    config: GeometryFieldConfig,
) -> QueryablePocketField:
    """Build the configured geometry field from preprocessed pocket data."""

    return GaussianGeometryField(processed_pocket=processed_pocket, config=config)


__all__ = [
    "FieldChannel",
    "FieldQueryResult",
    "GaussianGeometryField",
    "GeometryFieldConfig",
    "PocketField",
    "PocketFieldProvider",
    "QueryablePocketField",
    "build_geometry_field",
    "load_geometry_field_config",
]
