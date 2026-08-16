"""Training extension points and the Stage 3 coordinate-noise objective."""

from gpcr_ringgen.training.losses import (
    JointDiffusionLoss,
    JointDiffusionLossOutput,
    NoisePredictionLoss,
    masked_categorical_cross_entropy,
    masked_coordinate_mse,
)
from gpcr_ringgen.training.trainer import Trainer

__all__ = [
    "JointDiffusionLoss",
    "JointDiffusionLossOutput",
    "NoisePredictionLoss",
    "Trainer",
    "masked_categorical_cross_entropy",
    "masked_coordinate_mse",
]
