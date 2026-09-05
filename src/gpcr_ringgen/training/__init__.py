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
from gpcr_ringgen.training.inference import run_record_inference
from gpcr_ringgen.training.ra_reward import (
    RARewardModel,
    RARewardModelConfig,
    load_ra_reward_checkpoint,
    save_ra_reward_checkpoint,
)
from gpcr_ringgen.training.ra_reward_training import (
    RARewardTrainingResult,
    run_ra_reward_training,
)
from gpcr_ringgen.training.ra_reward_data import (
    RAScoreLabelStore,
    RAScoreLabeledDataset,
    build_rascore_label_cache,
    collate_ra_records,
)

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
    "run_record_inference",
    "RARewardModel",
    "RARewardModelConfig",
    "RARewardTrainingResult",
    "load_ra_reward_checkpoint",
    "run_ra_reward_training",
    "save_ra_reward_checkpoint",
    "RAScoreLabelStore",
    "RAScoreLabeledDataset",
    "build_rascore_label_cache",
    "collate_ra_records",
]
