"""Training extension points and the Stage 3 coordinate-noise objective."""

from gpcr_ringgen.training.losses import NoisePredictionLoss, masked_coordinate_mse
from gpcr_ringgen.training.trainer import Trainer

__all__ = ["NoisePredictionLoss", "Trainer", "masked_coordinate_mse"]
