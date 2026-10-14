"""Differentiable RAscore reward guidance for fixed-topology sampling."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping

import torch
from torch import Tensor

from gpcr_ringgen.core.exceptions import ConfigurationError
from gpcr_ringgen.core.types import Stage3DiffusionState
from gpcr_ringgen.models.equivariant import DenoiserOutput

if TYPE_CHECKING:
    from gpcr_ringgen.training.ra_reward import RARewardModel


def _finite_nonnegative(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite non-negative number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return result


@dataclass(frozen=True)
class RAScoreRewardGuidanceConfig:
    """Runtime controls for reward-gradient guidance."""

    enabled: bool = True
    guidance_scale: float = 0.8
    start_ratio: float = 0.5
    eval_every: int = 1
    temperature: float = 1.0
    gradient_clip_norm: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("ra reward guidance enabled must be a bool")
        object.__setattr__(
            self,
            "guidance_scale",
            _finite_nonnegative("guidance_scale", self.guidance_scale),
        )
        ratio = float(self.start_ratio)
        if not math.isfinite(ratio) or not 0.0 <= ratio <= 1.0:
            raise ValueError("start_ratio must be finite and in [0,1]")
        object.__setattr__(self, "start_ratio", ratio)
        if isinstance(self.eval_every, bool) or not isinstance(self.eval_every, int) or self.eval_every <= 0:
            raise ValueError("eval_every must be a positive integer")
        object.__setattr__(self, "temperature", _finite_nonnegative("temperature", self.temperature))
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive")
        object.__setattr__(
            self,
            "gradient_clip_norm",
            _finite_nonnegative("gradient_clip_norm", self.gradient_clip_norm),
        )

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "RAScoreRewardGuidanceConfig":
        section = config.get("ra_reward", config.get("guidance", {}))
        if not isinstance(section, Mapping):
            raise ConfigurationError("ra_reward config must be a mapping")
        enabled = section.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ConfigurationError("ra_reward.enabled must be a bool")
        return cls(
            enabled=enabled,
            guidance_scale=section.get("guidance_scale", 0.8),
            start_ratio=section.get("start_ratio", 0.5),
            eval_every=section.get("eval_every", 1),
            temperature=section.get("temperature", 1.0),
            gradient_clip_norm=section.get("gradient_clip_norm", 1.0),
        )


def _normalize_gradient(gradient: Tensor, mask: Tensor, max_norm: float) -> Tensor:
    gradient = gradient * mask.unsqueeze(-1).to(gradient.dtype)
    if max_norm == 0.0:
        return torch.zeros_like(gradient)
    norm = gradient.square().sum(dim=(-1, -2), keepdim=True).sqrt()
    factor = torch.minimum(
        torch.ones_like(norm),
        torch.full_like(norm, max_norm) / norm.clamp_min(torch.finfo(gradient.dtype).eps),
    )
    return gradient * factor


@dataclass(frozen=True)
class RewardGuidanceResult:
    prediction: DenoiserOutput
    reward: Tensor | None


class RAScoreRewardGuidance:
    """Apply a frozen RA reward checkpoint to chemical denoiser logits.

    This is intentionally a reward-gradient method, not classical CFG: the
    generator checkpoint remains unconditional and the external reward model
    supplies the additional direction. Coordinates and fixed topology are not
    modified because the RA reward model consumes chemical graph labels only.
    """

    def __init__(
        self,
        reward_model: "RARewardModel",
        config: RAScoreRewardGuidanceConfig | None = None,
    ) -> None:
        if not isinstance(reward_model, torch.nn.Module) or not hasattr(reward_model, "config"):
            raise TypeError("reward_model must be an RARewardModel")
        self.reward_model = reward_model.eval()
        for parameter in self.reward_model.parameters():
            parameter.requires_grad_(False)
        self.config = config or RAScoreRewardGuidanceConfig()

    def _active(self, timestep_index: int, num_timesteps: int) -> bool:
        if not self.config.enabled or self.config.guidance_scale == 0.0:
            return False
        if isinstance(timestep_index, bool) or not isinstance(timestep_index, int):
            raise TypeError("timestep_index must be an integer")
        if num_timesteps <= 0 or timestep_index < 0 or timestep_index >= num_timesteps:
            raise ValueError("timestep_index is outside the diffusion schedule")
        reverse_index = num_timesteps - 1 - timestep_index
        if reverse_index % self.config.eval_every != 0:
            return False
        progress = reverse_index / max(num_timesteps - 1, 1)
        return progress >= self.config.start_ratio

    def apply(
        self,
        state: Stage3DiffusionState,
        prediction: DenoiserOutput,
        *,
        timestep_index: int,
        num_timesteps: int,
    ) -> RewardGuidanceResult:
        if not isinstance(state, Stage3DiffusionState):
            raise TypeError("state must be a Stage3DiffusionState")
        state.validate()
        if not isinstance(prediction, DenoiserOutput):
            raise TypeError("prediction must be a DenoiserOutput")
        prediction.validate(state)
        if not self._active(timestep_index, num_timesteps):
            return RewardGuidanceResult(prediction, None)
        if prediction.atom_type_logits is None or prediction.bond_type_logits is None:
            raise ValueError("RA reward guidance requires atom and bond logits")
        parameter = next(self.reward_model.parameters())
        if parameter.device != state.coordinates.device or parameter.dtype != state.coordinates.dtype:
            raise TypeError("RA reward model and diffusion state must share device and dtype")

        # Detach from the generator graph. Guidance changes the reverse-step
        # logits but never accumulates gradients into the generator checkpoint.
        atom_logits = prediction.atom_type_logits.detach().requires_grad_(True)
        bond_logits = prediction.bond_type_logits.detach().requires_grad_(True)
        with torch.enable_grad():
            atom_probabilities = torch.softmax(atom_logits / self.config.temperature, dim=-1)
            bond_probabilities = torch.softmax(bond_logits / self.config.temperature, dim=-1)
            reward = self.reward_model(
                state,
                atom_type_probabilities=atom_probabilities,
                bond_type_probabilities=bond_probabilities,
            )
            atom_gradient, bond_gradient = torch.autograd.grad(
                reward.sum(),
                (atom_logits, bond_logits),
                allow_unused=False,
            )
        scale = self.config.guidance_scale
        guided_atom_logits = atom_logits.detach() + scale * _normalize_gradient(
            atom_gradient.detach(),
            state.mask.atom_mask,
            self.config.gradient_clip_norm,
        )
        guided_bond_logits = bond_logits.detach() + scale * _normalize_gradient(
            bond_gradient.detach(),
            state.mask.edge_mask,
            self.config.gradient_clip_norm,
        )
        return RewardGuidanceResult(
            DenoiserOutput(
                coordinate_noise=prediction.coordinate_noise,
                atom_type_logits=guided_atom_logits,
                bond_type_logits=guided_bond_logits,
                formal_charge_logits=prediction.formal_charge_logits,
                chirality_logits=prediction.chirality_logits,
            ),
            reward.detach(),
        )


def load_rascore_reward_guidance(
    config: Mapping[str, Any],
    *,
    device: torch.device,
    dtype: torch.dtype,
    atom_feature_dim: int,
    edge_feature_dim: int,
    num_atom_types: int,
    num_bond_types: int,
) -> RAScoreRewardGuidance | None:
    """Load and validate the optional reward checkpoint from an inference config."""
    section = config.get("ra_reward")
    if section is None:
        return None
    if not isinstance(section, Mapping):
        raise ConfigurationError("ra_reward config must be a mapping")
    enabled = section.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ConfigurationError("ra_reward.enabled must be a bool")
    if not enabled:
        return None
    checkpoint = section.get("checkpoint")
    if not isinstance(checkpoint, str) or not checkpoint.strip():
        raise ConfigurationError("ra_reward.checkpoint is required when enabled")
    from gpcr_ringgen.training.ra_reward import load_ra_reward_checkpoint

    reward_model = load_ra_reward_checkpoint(
        checkpoint,
        device=device,
        dtype=dtype,
    )
    from gpcr_ringgen.checkpoint_semantics import validate_reward_guidance_metadata

    validate_reward_guidance_metadata(
        reward_model.checkpoint_semantics, reward_model.training_provenance, config
    )
    reward_config = reward_model.config
    expected = {
        "atom_feature_dim": atom_feature_dim,
        "edge_feature_dim": edge_feature_dim,
        "num_atom_types": num_atom_types,
        "num_bond_types": num_bond_types,
    }
    for name, value in expected.items():
        if getattr(reward_config, name) != value:
            raise ConfigurationError(
                f"RA reward checkpoint {name} does not match inference model: "
                f"{getattr(reward_config, name)} versus {value}"
            )
    return RAScoreRewardGuidance(
        reward_model,
        RAScoreRewardGuidanceConfig.from_mapping(config),
    )


__all__ = [
    "RAScoreRewardGuidance",
    "RAScoreRewardGuidanceConfig",
    "load_rascore_reward_guidance",
    "RewardGuidanceResult",
]
