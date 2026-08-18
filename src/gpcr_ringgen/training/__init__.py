"""Training extension points and the Stage 3 coordinate-noise objective."""

from gpcr_ringgen.training.losses import (
    JointDiffusionLoss,
    JointDiffusionLossOutput,
    NoisePredictionLoss,
    RingSidechainDiffusionLoss,
    RingSidechainDiffusionLossOutput,
    masked_categorical_cross_entropy,
    masked_coordinate_mse,
)
from gpcr_ringgen.training.trainer import Trainer, TrainingResult, run_training

__all__ = [
    "JointDiffusionLoss",
    "JointDiffusionLossOutput",
    "NoisePredictionLoss",
    "RingSidechainDiffusionLoss",
    "RingSidechainDiffusionLossOutput",
    "Trainer",
    "TrainingResult",
    "masked_categorical_cross_entropy",
    "masked_coordinate_mse",
    "run_training",
]
