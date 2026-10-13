"""Mask-aware coordinate objectives for Stage 3 diffusion training."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from gpcr_ringgen.models.diffusion.chemical import JointForwardDiffusionOutput
from gpcr_ringgen.models.diffusion.ring_sidechain import RingSidechainForwardOutput
from gpcr_ringgen.models.equivariant.backbone import DenoiserOutput


class Loss(ABC):
    """Minimal objective contract retained for future training components."""

    @abstractmethod
    def compute(
        self,
        prediction: Any,
        target: Any,
        mask: Any | None = None,
    ) -> Any:
        """Compute an objective from aligned prediction and target values."""


def masked_coordinate_mse(
    prediction: Tensor,
    target: Tensor,
    diffusion_mask: Tensor,
    *,
    reduction: str = "mean",
) -> Tensor:
    """Compute DDPM coordinate-noise MSE over diffused non-ring atoms only.

    Args:
        prediction: Shape ``[B, N, 3]``; floating-point on the model device.
        target: Shape ``[B, N, 3]``; same dtype/device as ``prediction``.
        diffusion_mask: Shape ``[B, N]``; ``torch.bool`` on the same device.
            ``True`` non-ring atoms contribute all coordinate components;
            fixed ring atoms and padding do not contribute.
        reduction: ``"mean"`` divides by ``3 * number_of_valid_atoms``;
            ``"sum"`` returns the masked squared-error sum. Both are scalars.
    """

    if not isinstance(prediction, Tensor) or not isinstance(target, Tensor):
        raise TypeError("prediction and target must be torch.Tensor objects")
    if prediction.ndim != 3 or prediction.shape[-1] != 3:
        raise ValueError("prediction must have shape [B, N, 3]")
    if target.shape != prediction.shape:
        raise ValueError("target shape must match prediction [B, N, 3]")
    if not torch.is_floating_point(prediction) or not torch.is_floating_point(target):
        raise TypeError("prediction and target must use floating-point dtypes")
    if target.dtype != prediction.dtype:
        raise TypeError("target dtype must match prediction")
    if target.device != prediction.device:
        raise ValueError("target device must match prediction")
    if not isinstance(diffusion_mask, Tensor) or diffusion_mask.ndim != 2:
        raise ValueError("diffusion_mask must have shape [B, N]")
    if tuple(diffusion_mask.shape) != tuple(prediction.shape[:2]):
        raise ValueError("diffusion_mask shape must match prediction [B, N]")
    if diffusion_mask.dtype != torch.bool:
        raise TypeError("diffusion_mask must have dtype torch.bool")
    if diffusion_mask.device != prediction.device:
        raise ValueError("diffusion_mask device must match prediction")
    if not bool(torch.isfinite(prediction).all()):
        raise ValueError("prediction contains NaN or Inf")
    if not bool(torch.isfinite(target).all()):
        raise ValueError("target contains NaN or Inf")
    if reduction not in {"mean", "sum"}:
        raise ValueError("reduction must be 'mean' or 'sum'")

    weights = diffusion_mask.unsqueeze(-1).to(prediction.dtype)
    valid_atom_count = diffusion_mask.sum()
    if int(valid_atom_count.item()) == 0:
        raise ValueError("diffusion_mask must contain at least one diffused atom")
    squared_error = (prediction - target).square() * weights
    total = squared_error.sum()
    if reduction == "sum":
        return total
    denominator = valid_atom_count.to(prediction.dtype) * prediction.shape[-1]
    return total / denominator


class NoisePredictionLoss(nn.Module, Loss):
    """Callable epsilon-prediction MSE configured with a fixed reduction."""

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        if reduction not in {"mean", "sum"}:
            raise ValueError("reduction must be 'mean' or 'sum'")
        self.reduction = reduction

    def compute(
        self,
        prediction: Tensor,
        target: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        """Return scalar mask-aware epsilon-prediction MSE."""

        if mask is None:
            raise ValueError("diffusion mask is required for diffusion loss")
        return masked_coordinate_mse(
            prediction,
            target,
            mask,
            reduction=self.reduction,
        )

    def forward(
        self,
        prediction: Tensor,
        target: Tensor,
        diffusion_mask: Tensor,
    ) -> Tensor:
        """Delegate module calls to :meth:`compute`."""

        return self.compute(prediction, target, diffusion_mask)


def masked_categorical_cross_entropy(
    logits: Tensor,
    target: Tensor,
    mask: Tensor,
    *,
    reduction: str = "mean",
) -> Tensor:
    """Cross entropy for categorical labels on frozen topology positions.

    Args:
        logits: Shape ``[B, L, K]``; floating-point on the model device.
        target: Shape ``[B, L]``; ``torch.long`` category IDs. For atoms,
            ``L=N``; for bonds, ``L=E`` and only existing fixed edges are valid.
        mask: Shape ``[B, L]``; ``torch.bool``. ``False`` padding contributes
            neither loss values nor the mean denominator.
    """

    if not isinstance(logits, Tensor) or logits.ndim != 3:
        raise ValueError("logits must be a torch.Tensor with shape [B, L, K]")
    if not torch.is_floating_point(logits):
        raise TypeError("logits must use a floating-point dtype")
    if not isinstance(target, Tensor) or target.ndim != 2:
        raise ValueError("target must be a torch.Tensor with shape [B, L]")
    if target.dtype != torch.long:
        raise TypeError("target must have dtype torch.long")
    if not isinstance(mask, Tensor) or mask.ndim != 2:
        raise ValueError("mask must be a torch.Tensor with shape [B, L]")
    if mask.dtype != torch.bool:
        raise TypeError("mask must have dtype torch.bool")
    if tuple(logits.shape[:2]) != tuple(target.shape) or target.shape != mask.shape:
        raise ValueError("logits, target, and mask batch/length dimensions must match")
    if logits.device != target.device or target.device != mask.device:
        raise ValueError("logits, target, and mask must share a device")
    if logits.shape[-1] < 2:
        raise ValueError("categorical logits must contain at least two classes")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("categorical logits contain NaN or Inf")
    valid_target = target.masked_select(mask)
    if valid_target.numel() == 0:
        raise ValueError("categorical mask must contain at least one valid entry")
    if bool(((valid_target < 0) | (valid_target >= logits.shape[-1])).any()):
        raise ValueError("categorical target is outside the logits vocabulary")
    if reduction not in {"mean", "sum"}:
        raise ValueError("reduction must be 'mean' or 'sum'")
    return F.cross_entropy(logits[mask], valid_target, reduction=reduction)


@dataclass(frozen=True)
class JointDiffusionLossOutput:
    """Scalar coordinate, atom, bond, and weighted total objectives."""

    total: Tensor
    coordinate: Tensor
    atom: Tensor
    bond: Tensor


class JointDiffusionLoss(nn.Module):
    """Joint objective with frozen topology and generated chemical labels."""

    def __init__(
        self,
        *,
        coordinate_weight: float = 1.0,
        atom_weight: float = 1.0,
        bond_weight: float = 1.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        for name, value in (
            ("coordinate_weight", coordinate_weight),
            ("atom_weight", atom_weight),
            ("bond_weight", bond_weight),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a real number")
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if coordinate_weight + atom_weight + bond_weight <= 0.0:
            raise ValueError("at least one joint loss weight must be positive")
        if reduction not in {"mean", "sum"}:
            raise ValueError("reduction must be 'mean' or 'sum'")
        self.coordinate_weight = float(coordinate_weight)
        self.atom_weight = float(atom_weight)
        self.bond_weight = float(bond_weight)
        self.reduction = reduction

    def forward(
        self,
        prediction: DenoiserOutput,
        target: JointForwardDiffusionOutput,
    ) -> JointDiffusionLossOutput:
        if not isinstance(prediction, DenoiserOutput):
            raise TypeError("prediction must be a DenoiserOutput")
        if not isinstance(target, JointForwardDiffusionOutput):
            raise TypeError("target must be a JointForwardDiffusionOutput")
        state = target.noisy_state
        prediction.validate(state)
        if prediction.atom_type_logits is None or prediction.bond_type_logits is None:
            raise ValueError("joint diffusion requires atom and bond prediction logits")
        coordinate_loss = masked_coordinate_mse(
            prediction.coordinate_noise,
            target.coordinate_noise,
            state.mask.atom_mask,
            reduction=self.reduction,
        )
        atom_loss = masked_categorical_cross_entropy(
            prediction.atom_type_logits,
            target.clean_atom_types,
            state.mask.atom_mask,
            reduction=self.reduction,
        )
        bond_loss = masked_categorical_cross_entropy(
            prediction.bond_type_logits,
            target.clean_bond_types,
            state.mask.edge_mask,
            reduction=self.reduction,
        )
        total = (
            self.coordinate_weight * coordinate_loss
            + self.atom_weight * atom_loss
            + self.bond_weight * bond_loss
        )
        if not bool(torch.isfinite(total)):
            raise ValueError("joint diffusion loss is not finite")
        return JointDiffusionLossOutput(
            total=total,
            coordinate=coordinate_loss,
            atom=atom_loss,
            bond=bond_loss,
        )


@dataclass(frozen=True)
class RingSidechainDiffusionLossOutput:
    """Route-A coordinate and categorical loss components."""

    total: Tensor
    coordinate: Tensor
    atom: Tensor
    bond: Tensor
    formal_charge: Tensor
    chirality: Tensor


def _masked_ce_or_zero(
    logits: Tensor,
    target: Tensor,
    mask: Tensor,
    *,
    reduction: str,
) -> Tensor:
    if bool(mask.any()):
        return masked_categorical_cross_entropy(
            logits,
            target,
            mask,
            reduction=reduction,
        )
    return logits.sum() * 0.0


class RingSidechainDiffusionLoss(nn.Module):
    """Route-A objective with fixed ring existence and generated chemistry."""

    def __init__(
        self,
        *,
        coordinate_weight: float = 1.0,
        atom_weight: float = 1.0,
        bond_weight: float = 1.0,
        formal_charge_weight: float = 1.0,
        chirality_weight: float = 1.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        weights = (
            coordinate_weight,
            atom_weight,
            bond_weight,
            formal_charge_weight,
            chirality_weight,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or value < 0
            for value in weights
        ):
            raise ValueError("Route-A loss weights must be non-negative real numbers")
        if sum(weights) <= 0:
            raise ValueError("at least one Route-A loss weight must be positive")
        if reduction not in {"mean", "sum"}:
            raise ValueError("reduction must be 'mean' or 'sum'")
        self.coordinate_weight = float(coordinate_weight)
        self.atom_weight = float(atom_weight)
        self.bond_weight = float(bond_weight)
        self.formal_charge_weight = float(formal_charge_weight)
        self.chirality_weight = float(chirality_weight)
        self.reduction = reduction

    def forward(
        self,
        prediction: DenoiserOutput,
        target: RingSidechainForwardOutput,
    ) -> RingSidechainDiffusionLossOutput:
        if not isinstance(target, RingSidechainForwardOutput):
            raise TypeError("target must be a RingSidechainForwardOutput")
        state = target.noisy_state
        prediction.validate(state)
        if prediction.atom_type_logits is None or prediction.bond_type_logits is None:
            raise ValueError("Route-A prediction requires atom and bond logits")
        if prediction.formal_charge_logits is None or prediction.chirality_logits is None:
            raise ValueError("Route-A prediction requires charge and chirality logits")
        coordinate = masked_coordinate_mse(
            prediction.coordinate_noise,
            target.coordinate_noise,
            state.mask.atom_mask,
            reduction=self.reduction,
        )
        atom = masked_categorical_cross_entropy(
            prediction.atom_type_logits,
            target.clean_atom_types,
            state.mask.atom_mask,
            reduction=self.reduction,
        )
        bond = _masked_ce_or_zero(
            prediction.bond_type_logits,
            target.clean_bond_types,
            target.bond_loss_mask,
            reduction=self.reduction,
        )
        formal_charge = _masked_ce_or_zero(
            prediction.formal_charge_logits,
            target.clean_formal_charge_types,
            target.outer_atom_mask,
            reduction=self.reduction,
        )
        chirality = _masked_ce_or_zero(
            prediction.chirality_logits,
            target.clean_chiralities,
            target.outer_atom_mask,
            reduction=self.reduction,
        )
        total = (
            self.coordinate_weight * coordinate
            + self.atom_weight * atom
            + self.bond_weight * bond
            + self.formal_charge_weight * formal_charge
            + self.chirality_weight * chirality
        )
        if not bool(torch.isfinite(total)):
            raise ValueError("Route-A diffusion loss is not finite")
        return RingSidechainDiffusionLossOutput(
            total=total,
            coordinate=coordinate,
            atom=atom,
            bond=bond,
            formal_charge=formal_charge,
            chirality=chirality,
        )


__all__ = [
    "JointDiffusionLoss",
    "JointDiffusionLossOutput",
    "Loss",
    "NoisePredictionLoss",
    "RingSidechainDiffusionLoss",
    "RingSidechainDiffusionLossOutput",
    "masked_categorical_cross_entropy",
    "masked_coordinate_mse",
]
