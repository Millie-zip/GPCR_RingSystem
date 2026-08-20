"""Validated noise schedules for Stage 3 forward diffusion."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from gpcr_ringgen.utils.io import load_yaml_config


class NoiseSchedule(ABC):
    """Return a scalar schedule value at a discrete timestep."""

    @abstractmethod
    def value(self, timestep: int) -> float:
        """Return the beta value for one timestep."""


def _require_positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _require_probability(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    converted = float(value)
    if not math.isfinite(converted) or not 0.0 < converted < 1.0:
        raise ValueError(f"{name} must be finite and strictly between 0 and 1")
    return converted


@dataclass(frozen=True)
class DDPMConfig:
    """Configuration for forward DDPM epsilon prediction and its loss.

    ``linear`` is the only schedule currently implemented. Timesteps use
    zero-based indices ``0, ..., num_timesteps - 1``. Consequently ``t=0`` is
    the weakest configured noising step, with beta equal to ``beta_start``;
    it is not an artificial clean-state sentinel.
    """

    num_timesteps: int
    beta_schedule: str
    beta_start: float
    beta_end: float
    prediction_type: str
    loss_reduction: str
    mode: str = "coordinate_only"
    num_atom_types: int | None = None
    num_bond_types: int | None = None
    coordinate_loss_weight: float = 1.0
    atom_loss_weight: float = 1.0
    bond_loss_weight: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "num_timesteps",
            _require_positive_int("num_timesteps", self.num_timesteps),
        )
        if self.beta_schedule != "linear":
            raise ValueError("beta_schedule must be 'linear'")
        object.__setattr__(
            self,
            "beta_start",
            _require_probability("beta_start", self.beta_start),
        )
        object.__setattr__(
            self,
            "beta_end",
            _require_probability("beta_end", self.beta_end),
        )
        if self.beta_end < self.beta_start:
            raise ValueError("beta_end must be greater than or equal to beta_start")
        if self.prediction_type != "epsilon":
            raise ValueError("prediction_type must be 'epsilon'")
        if self.loss_reduction not in {"mean", "sum"}:
            raise ValueError("loss_reduction must be 'mean' or 'sum'")
        if self.mode not in {"coordinate_only", "joint_chemical"}:
            raise ValueError(
                "diffusion mode must be 'coordinate_only' or 'joint_chemical'"
            )
        if self.mode == "joint_chemical":
            for name in ("num_atom_types", "num_bond_types"):
                value = getattr(self, name)
                if value is None:
                    raise ValueError(f"{name} is required in joint_chemical mode")
                object.__setattr__(self, name, _require_positive_int(name, value))
                if getattr(self, name) < 2:
                    raise ValueError(f"{name} must be at least 2")
        for name in (
            "coordinate_loss_weight",
            "atom_loss_weight",
            "bond_loss_weight",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a real number")
            converted = float(value)
            if not math.isfinite(converted) or converted < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, converted)
        if self.mode == "joint_chemical" and (
            self.coordinate_loss_weight
            + self.atom_loss_weight
            + self.bond_loss_weight
            <= 0.0
        ):
            raise ValueError("at least one joint diffusion loss weight must be positive")

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "DDPMConfig":
        """Construct from a top-level mapping or its ``diffusion`` section."""

        diffusion_config = config.get("diffusion", config)
        if not isinstance(diffusion_config, Mapping):
            raise TypeError("diffusion config must be a mapping")
        required = (
            "num_timesteps",
            "beta_schedule",
            "beta_start",
            "beta_end",
            "prediction_type",
            "loss_reduction",
        )
        missing = [name for name in required if name not in diffusion_config]
        if missing:
            raise KeyError(f"missing diffusion config keys: {', '.join(missing)}")
        optional = (
            "mode",
            "num_atom_types",
            "num_bond_types",
            "coordinate_loss_weight",
            "atom_loss_weight",
            "bond_loss_weight",
        )
        values = {name: diffusion_config[name] for name in required}
        values.update(
            {
                name: diffusion_config[name]
                for name in optional
                if name in diffusion_config
            }
        )
        return cls(**values)


def load_ddpm_config(path: str | Path) -> DDPMConfig:
    """Load and validate a DDPM configuration through the project YAML loader."""

    return DDPMConfig.from_mapping(load_yaml_config(path))


class DDPMNoiseSchedule(nn.Module, NoiseSchedule):
    """Linear beta schedule with precomputed DDPM cumulative coefficients.

    Schedule buffers are stored in float64 for stable cumulative products.
    :meth:`coefficients` casts the selected values to the reference tensor's
    dtype and device, so no caller-side schedule transfer is required.
    """

    def __init__(self, config: DDPMConfig) -> None:
        super().__init__()
        if not isinstance(config, DDPMConfig):
            raise TypeError("config must be a DDPMConfig")
        self.config = config
        betas = torch.linspace(
            config.beta_start,
            config.beta_end,
            config.num_timesteps,
            dtype=torch.float64,
        )
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)

    @property
    def num_timesteps(self) -> int:
        """Return the number of valid discrete timestep indices."""

        return self.config.num_timesteps

    def value(self, timestep: int) -> float:
        """Return beta at one zero-based timestep as a Python float."""

        if isinstance(timestep, bool) or not isinstance(timestep, int):
            raise TypeError("timestep must be an integer")
        if not 0 <= timestep < self.num_timesteps:
            raise ValueError(
                f"timestep must be in [0, {self.num_timesteps - 1}]"
            )
        return float(self.betas[timestep].item())

    def coefficients(
        self,
        timestep: Tensor,
        reference: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return broadcastable signal/noise coefficients for a batch.

        Args:
            timestep: Shape ``[B]``; ``torch.long`` on the same device as
                ``reference``. Every value must be in the schedule range.
            reference: Floating tensor shaped ``[B, ...]`` whose dtype/device
                determine the returned coefficient tensors.

        Returns:
            ``(sqrt(alpha_bar_t), sqrt(1-alpha_bar_t))``, each shaped
            ``[B, 1, ...]`` with one singleton for every non-batch axis of
            ``reference``. No mask is applied by the schedule.
        """

        if not isinstance(timestep, Tensor):
            raise TypeError("timestep must be a torch.Tensor")
        if timestep.ndim != 1:
            raise ValueError("timestep must have shape [B]")
        if timestep.dtype != torch.long:
            raise TypeError("timestep must have dtype torch.long")
        if not isinstance(reference, Tensor) or reference.ndim < 1:
            raise ValueError("reference must be a torch.Tensor shaped [B, ...]")
        if not torch.is_floating_point(reference):
            raise TypeError("reference must use a floating-point dtype")
        if timestep.shape[0] != reference.shape[0]:
            raise ValueError("timestep and reference batch sizes must match")
        if timestep.device != reference.device:
            raise ValueError("timestep and reference must be on the same device")
        if bool(((timestep < 0) | (timestep >= self.num_timesteps)).any()):
            raise ValueError(
                f"timestep values must be in [0, {self.num_timesteps - 1}]"
            )

        alpha_bars = self.alpha_bars.to(
            device=reference.device,
            dtype=reference.dtype,
        ).index_select(0, timestep)
        output_shape = (reference.shape[0],) + (1,) * (reference.ndim - 1)
        alpha_bars = alpha_bars.reshape(output_shape)
        return torch.sqrt(alpha_bars), torch.sqrt(1.0 - alpha_bars)


__all__ = [
    "DDPMConfig",
    "DDPMNoiseSchedule",
    "NoiseSchedule",
    "load_ddpm_config",
]
