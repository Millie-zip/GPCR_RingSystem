"""Mask-aware coordinate objectives for Stage 3 diffusion training."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import Tensor, nn


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
    atom_mask: Tensor,
    *,
    reduction: str = "mean",
) -> Tensor:
    """Compute DDPM coordinate-noise MSE over valid atoms only.

    Args:
        prediction: Shape ``[B, N, 3]``; floating-point on the model device.
        target: Shape ``[B, N, 3]``; same dtype/device as ``prediction``.
        atom_mask: Shape ``[B, N]``; ``torch.bool`` on the same device.
            ``True`` atoms contribute all three coordinate components;
            ``False`` padding contributes neither values nor denominator count.
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
    if not isinstance(atom_mask, Tensor) or atom_mask.ndim != 2:
        raise ValueError("atom_mask must be a torch.Tensor with shape [B, N]")
    if tuple(atom_mask.shape) != tuple(prediction.shape[:2]):
        raise ValueError("atom_mask shape must match prediction [B, N]")
    if atom_mask.dtype != torch.bool:
        raise TypeError("atom_mask must have dtype torch.bool")
    if atom_mask.device != prediction.device:
        raise ValueError("atom_mask device must match prediction")
    if not bool(torch.isfinite(prediction).all()):
        raise ValueError("prediction contains NaN or Inf")
    if not bool(torch.isfinite(target).all()):
        raise ValueError("target contains NaN or Inf")
    if reduction not in {"mean", "sum"}:
        raise ValueError("reduction must be 'mean' or 'sum'")

    weights = atom_mask.unsqueeze(-1).to(prediction.dtype)
    valid_atom_count = atom_mask.sum()
    if int(valid_atom_count.item()) == 0:
        raise ValueError("atom_mask must contain at least one valid atom")
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
            raise ValueError("atom mask is required for diffusion loss")
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
        atom_mask: Tensor,
    ) -> Tensor:
        """Delegate module calls to :meth:`compute`."""

        return self.compute(prediction, target, atom_mask)


__all__ = ["Loss", "NoisePredictionLoss", "masked_coordinate_mse"]
