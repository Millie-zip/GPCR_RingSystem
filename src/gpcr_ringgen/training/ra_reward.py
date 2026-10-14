"""Independent differentiable surrogate for an external RAscore teacher."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from gpcr_ringgen.core.types import Stage3DiffusionState
from gpcr_ringgen.core.exceptions import ConfigurationError
from gpcr_ringgen.checkpoint_semantics import plain_data, validate_reward_metadata
from gpcr_ringgen.utils.io import ensure_directory


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


@dataclass(frozen=True)
class RARewardModelConfig:
    """Shape and vocabulary contract for the independent reward model."""

    atom_feature_dim: int
    edge_feature_dim: int
    hidden_dim: int
    num_layers: int
    num_atom_types: int
    num_bond_types: int
    dropout: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "atom_feature_dim",
            "edge_feature_dim",
            "hidden_dim",
            "num_layers",
            "num_atom_types",
            "num_bond_types",
        ):
            object.__setattr__(self, name, _positive_int(name, getattr(self, name)))
        if self.num_atom_types < 2 or self.num_bond_types < 2:
            raise ValueError("chemical vocabularies must contain at least two classes")
        value = float(self.dropout)
        if not math.isfinite(value) or not 0.0 <= value < 1.0:
            raise ValueError("dropout must be finite and in [0, 1)")
        object.__setattr__(self, "dropout", value)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RARewardModelConfig":
        section = value.get("ra_reward_model", value)
        if not isinstance(section, Mapping):
            raise TypeError("ra_reward_model config must be a mapping")
        required = (
            "atom_feature_dim",
            "edge_feature_dim",
            "hidden_dim",
            "num_layers",
            "num_atom_types",
            "num_bond_types",
        )
        missing = [name for name in required if name not in section]
        if missing:
            raise KeyError("missing RA reward model keys: " + ", ".join(missing))
        return cls(
            **{name: section[name] for name in required},
            dropout=section.get("dropout", 0.0),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "atom_feature_dim": self.atom_feature_dim,
            "edge_feature_dim": self.edge_feature_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "num_atom_types": self.num_atom_types,
            "num_bond_types": self.num_bond_types,
            "dropout": self.dropout,
        }


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    weights = mask.unsqueeze(-1).to(values.dtype)
    denominator = weights.sum(dim=1).clamp_min(1.0)
    return (values * weights).sum(dim=1) / denominator


class RARewardModel(nn.Module):
    """Graph reward model that accepts hard labels or differentiable probabilities.

    The model deliberately ignores coordinates. During inference, the denoiser's
    atom/bond logits are converted to probabilities and passed here, allowing
    gradients with respect to those logits without calling the black-box teacher.
    """

    def __init__(self, config: RARewardModelConfig) -> None:
        super().__init__()
        if not isinstance(config, RARewardModelConfig):
            raise TypeError("config must be an RARewardModelConfig")
        self.config = config
        hidden = config.hidden_dim
        self.atom_projection = nn.Sequential(
            nn.Linear(config.atom_feature_dim + config.num_atom_types, hidden),
            nn.SiLU(),
            nn.Dropout(config.dropout),
        )
        self.edge_projection = nn.Sequential(
            nn.Linear(config.edge_feature_dim + config.num_bond_types, hidden),
            nn.SiLU(),
            nn.Dropout(config.dropout),
        )
        self.message_layers = nn.ModuleList(
            nn.Sequential(
                nn.Linear(3 * hidden, hidden),
                nn.SiLU(),
                nn.Dropout(config.dropout),
            )
            for _ in range(config.num_layers)
        )
        self.node_layers = nn.ModuleList(
            nn.Sequential(
                nn.Linear(2 * hidden, hidden),
                nn.SiLU(),
                nn.Dropout(config.dropout),
            )
            for _ in range(config.num_layers)
        )
        self.edge_layers = nn.ModuleList(
            nn.Sequential(
                nn.Linear(3 * hidden, hidden),
                nn.SiLU(),
                nn.Dropout(config.dropout),
            )
            for _ in range(config.num_layers)
        )
        self.head = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def _probabilities(
        self,
        labels: Tensor | None,
        probabilities: Tensor | None,
        mask: Tensor,
        num_classes: int,
        name: str,
        dtype: torch.dtype,
    ) -> Tensor:
        if probabilities is None:
            if labels is None:
                raise ValueError(f"{name} labels are required")
            probabilities = torch.nn.functional.one_hot(
                labels,
                num_classes=num_classes,
            ).to(dtype)
        else:
            if probabilities.ndim != 3 or tuple(probabilities.shape[:2]) != tuple(mask.shape):
                raise ValueError(f"{name} probabilities must have shape [B, L, K]")
            if probabilities.shape[-1] != num_classes:
                raise ValueError(f"{name} probabilities have the wrong class count")
            if probabilities.device != mask.device or probabilities.dtype != dtype:
                raise TypeError(f"{name} probabilities must match state device and dtype")
            if not bool(torch.isfinite(probabilities).all()):
                raise ValueError(f"{name} probabilities contain NaN or Inf")
            probabilities = probabilities.clamp_min(0.0)
            probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
        return probabilities * mask.unsqueeze(-1).to(dtype)

    def forward(
        self,
        state: Stage3DiffusionState,
        *,
        atom_type_probabilities: Tensor | None = None,
        bond_type_probabilities: Tensor | None = None,
    ) -> Tensor:
        if not isinstance(state, Stage3DiffusionState):
            raise TypeError("state must be a Stage3DiffusionState")
        state.validate()
        if state.atom_types is None or state.bond_types is None:
            raise ValueError("RA reward model requires atom_types and bond_types")
        if state.atom_features.shape[-1] != self.config.atom_feature_dim:
            raise ValueError("state atom features do not match RA reward model")
        if state.edge_features.shape[-1] != self.config.edge_feature_dim:
            raise ValueError("state edge features do not match RA reward model")
        atom_probabilities = self._probabilities(
            state.atom_types,
            atom_type_probabilities,
            state.mask.atom_mask,
            self.config.num_atom_types,
            "atom",
            state.atom_features.dtype,
        )
        bond_probabilities = self._probabilities(
            state.bond_types,
            bond_type_probabilities,
            state.mask.edge_mask,
            self.config.num_bond_types,
            "bond",
            state.edge_features.dtype,
        )
        node_hidden = self.atom_projection(
            torch.cat((state.atom_features, atom_probabilities), dim=-1)
        )
        edge_hidden = self.edge_projection(
            torch.cat((state.edge_features, bond_probabilities), dim=-1)
        )
        node_hidden = node_hidden * state.mask.atom_mask.unsqueeze(-1).to(node_hidden.dtype)
        edge_hidden = edge_hidden * state.mask.edge_mask.unsqueeze(-1).to(edge_hidden.dtype)
        batch_size, num_atoms, _ = node_hidden.shape
        num_edges = edge_hidden.shape[1]
        batch_index = torch.arange(batch_size, device=state.edge_index.device).unsqueeze(1)
        batch_index = batch_index.expand(batch_size, num_edges)
        source = state.edge_index[:, 0].clamp(min=0, max=num_atoms - 1)
        target = state.edge_index[:, 1].clamp(min=0, max=num_atoms - 1)
        for message_layer, node_layer, edge_layer in zip(
            self.message_layers,
            self.node_layers,
            self.edge_layers,
        ):
            source_hidden = node_hidden[batch_index, source]
            target_hidden = node_hidden[batch_index, target]
            message = message_layer(
                torch.cat((source_hidden + target_hidden, torch.abs(source_hidden - target_hidden), edge_hidden), dim=-1)
            )
            message = message * state.mask.edge_mask.unsqueeze(-1).to(message.dtype)
            aggregate = torch.zeros_like(node_hidden)
            for batch in range(batch_size):
                valid_edges = state.mask.edge_mask[batch]
                aggregate[batch].index_add_(0, source[batch, valid_edges], message[batch, valid_edges])
                aggregate[batch].index_add_(0, target[batch, valid_edges], message[batch, valid_edges])
            node_hidden = node_hidden + node_layer(torch.cat((node_hidden, aggregate), dim=-1))
            node_hidden = node_hidden * state.mask.atom_mask.unsqueeze(-1).to(node_hidden.dtype)
            source_hidden = node_hidden[batch_index, source]
            target_hidden = node_hidden[batch_index, target]
            edge_hidden = edge_hidden + edge_layer(
                torch.cat((source_hidden + target_hidden, torch.abs(source_hidden - target_hidden), edge_hidden), dim=-1)
            )
            edge_hidden = edge_hidden * state.mask.edge_mask.unsqueeze(-1).to(edge_hidden.dtype)
        pooled = torch.cat(
            (
                _masked_mean(node_hidden, state.mask.atom_mask),
                _masked_mean(edge_hidden, state.mask.edge_mask),
            ),
            dim=-1,
        )
        return self.head(pooled).squeeze(-1)


def save_ra_reward_checkpoint(
    model: RARewardModel,
    path: str | Path,
    *,
    epoch: int,
    semantics: Mapping[str, Any],
    provenance: Mapping[str, Any],
    optimizer: torch.optim.Optimizer | None = None,
    metrics: Mapping[str, Any] | None = None,
) -> Path:
    """Save an inspectable, self-describing RA reward checkpoint."""
    if not isinstance(model, RARewardModel):
        raise TypeError("model must be an RARewardModel")
    validate_reward_metadata(semantics, provenance, model.config.to_mapping())
    output = Path(path)
    ensure_directory(output.parent)
    payload: dict[str, Any] = {
        "checkpoint_version": 2,
        "semantics": plain_data(semantics),
        "training_provenance": plain_data(provenance),
        "model_config": model.config.to_mapping(),
        "model_state_dict": model.state_dict(),
        "epoch": int(epoch),
        "metrics": dict(metrics or {}),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(payload, output)
    return output


def load_ra_reward_checkpoint(
    path: str | Path,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> RARewardModel:
    """Load and validate a reward checkpoint for inference."""
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise ConfigurationError(f"RA reward checkpoint not found: {checkpoint_path}")
    try:
        payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if not isinstance(payload, Mapping):
            raise ValueError("checkpoint must be a mapping")
        config_value = payload.get("model_config")
        state_dict = payload.get("model_state_dict", payload.get("state_dict"))
        if not isinstance(config_value, Mapping) or not isinstance(state_dict, Mapping):
            raise ValueError("checkpoint must contain model_config and model_state_dict")
        model = RARewardModel(RARewardModelConfig.from_mapping(config_value)).to(device=device, dtype=dtype)
        if payload.get("checkpoint_version") != 2:
            raise ConfigurationError("RA reward checkpoint lacks supported semantic metadata")
        validate_reward_metadata(payload.get("semantics"), payload.get("training_provenance"), model.config.to_mapping())
        model.checkpoint_semantics = plain_data(payload["semantics"])
        model.training_provenance = plain_data(payload["training_provenance"])
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return model
    except (EOFError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ConfigurationError(f"invalid RA reward checkpoint: {exc}") from exc


__all__ = [
    "RARewardModel",
    "RARewardModelConfig",
    "load_ra_reward_checkpoint",
    "save_ra_reward_checkpoint",
]
