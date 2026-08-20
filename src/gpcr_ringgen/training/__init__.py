"""Training extension points and the Stage 3 coordinate-noise objective."""

from gpcr_ringgen.training.losses import NoisePredictionLoss, masked_coordinate_mse
from gpcr_ringgen.training.trainer import Trainer, TrainingResult, run_training
from gpcr_ringgen.training.inference import run_record_inference

__all__ = ["NoisePredictionLoss", "Trainer", "TrainingResult", "run_training", "run_record_inference", "masked_coordinate_mse"]
