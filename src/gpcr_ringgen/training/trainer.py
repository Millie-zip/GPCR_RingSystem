"""Plain PyTorch training loop for the real Stage 3 JSONL asset library."""

from __future__ import annotations

import hashlib
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from gpcr_ringgen.core.exceptions import ConfigurationError
from gpcr_ringgen.checkpoint_semantics import (
    generator_semantics,
    training_provenance,
    validate_generator_checkpoint,
)
from gpcr_ringgen.data import FeatureManifest, JsonlMolecularDataset, collate_molecules
from gpcr_ringgen.models.diffusion import (
    DDPMConfig,
    DDPMNoiseSchedule,
    ChemicalVocabulary,
    clean_stage3_state,
    q_sample,
    q_sample_joint,
    sample_timesteps,
)
from gpcr_ringgen.models.equivariant import (
    EquivariantDenoiserBackbone,
    EquivariantDenoiserConfig,
    FieldConditionedEquivariantDenoiser,
    FieldConditioningConfig,
)
from gpcr_ringgen.training.losses import JointDiffusionLoss, NoisePredictionLoss
from gpcr_ringgen.utils.device import resolve_device
from gpcr_ringgen.utils.io import save_json
from gpcr_ringgen.utils.seed import make_torch_generator, set_seed


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingResult:
    output_dir: Path
    last_checkpoint: Path
    best_checkpoint: Path
    metrics_path: Path
    epochs_completed: int


class Trainer(nn.Module):
    """Compatibility wrapper exposing the concrete training entry point."""

    def fit(self, config: dict[str, Any]) -> TrainingResult:
        return run_training(config)


def _section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = config.get(name)
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{name} must be a mapping")
    return value


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigurationError(f"{name} must be a positive integer")
    return value


def _non_negative(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{name} must be a non-negative number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ConfigurationError(f"{name} must be finite and non-negative")
    return result


def _manifest_path(config: Mapping[str, Any]) -> Path:
    value = _section(config, "data").get("feature_manifest")
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError("data.feature_manifest is required")
    path = Path(value)
    if not path.is_file():
        raise ConfigurationError(f"feature manifest not found: {path}")
    return path


def _validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    data = _section(config, "data")
    training = _section(config, "training")
    data_path = data.get("path")
    if not isinstance(data_path, str) or not Path(data_path).is_file():
        raise ConfigurationError(f"data.path not found: {data_path}")
    manifest_path = _manifest_path(config)
    manifest = FeatureManifest.from_file(manifest_path)
    if manifest.descriptor_dim != 0:
        raise ConfigurationError("current training path requires descriptor_features: []")
    device_config = _section(config, "device")
    if device_config.get("dtype") not in {"float32", "float64"}:
        raise ConfigurationError("device.dtype must be float32 or float64")
    if not isinstance(config.get("seed"), int) or isinstance(config.get("seed"), bool):
        raise ConfigurationError("top-level seed must be an integer")
    field = config.get("field_conditioning", {"mode": "none", "channels": []})
    if not isinstance(field, Mapping) or field.get("mode", "none") != "none":
        raise ConfigurationError("training requires field_conditioning.mode=none")
    model_config = EquivariantDenoiserConfig.from_mapping(config)
    if model_config.atom_feature_dim != manifest.atom_feature_dim or model_config.edge_feature_dim != manifest.edge_feature_dim:
        raise ConfigurationError("model feature dimensions do not match the feature manifest")
    if model_config.descriptor_dim is not None:
        raise ConfigurationError("model.descriptor_dim must be null for pocket-disabled training")
    ddpm = DDPMConfig.from_mapping(config)
    if ddpm.mode == "joint_chemical":
        if manifest.schema_version != "stage3_joint_feature_manifest_v1":
            raise ConfigurationError(
                "joint_chemical training requires "
                "stage3_joint_feature_manifest_v1; clean chemical labels "
                "must not be present in atom or edge features"
            )
        if model_config.num_atom_types != ddpm.num_atom_types:
            raise ConfigurationError(
                "model.num_atom_types must match diffusion.num_atom_types"
            )
        if model_config.num_bond_types != ddpm.num_bond_types:
            raise ConfigurationError(
                "model.num_bond_types must match diffusion.num_bond_types"
            )
        try:
            vocabulary = ChemicalVocabulary.from_mapping(config)
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigurationError(
                f"joint_chemical training requires a valid chemical_vocabulary: {exc}"
            ) from exc
        if vocabulary.num_atom_types != ddpm.num_atom_types:
            raise ConfigurationError(
                "chemical_vocabulary.atom_types must match diffusion.num_atom_types"
            )
        if vocabulary.num_bond_types != ddpm.num_bond_types:
            raise ConfigurationError(
                "chemical_vocabulary.bond_types must match diffusion.num_bond_types"
            )
    if not (0.0 < float(training.get("train_fraction", 0.9)) < 1.0):
        raise ConfigurationError("training.train_fraction must be between 0 and 1")
    return {
        "data_path": Path(data_path),
        "manifest_path": manifest_path,
        "manifest_hash": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "epochs": _positive_int("training.epochs", training.get("epochs", 1)),
        "batch_size": _positive_int("training.batch_size", training.get("batch_size", 1)),
        "num_workers": int(training.get("num_workers", 0)),
        "train_fraction": float(training.get("train_fraction", 0.9)),
        "learning_rate": _non_negative("training.learning_rate", training.get("learning_rate", 1e-3)),
        "weight_decay": _non_negative("training.weight_decay", training.get("weight_decay", 0.0)),
        "grad_clip_norm": _non_negative("training.grad_clip_norm", training.get("grad_clip_norm", 1.0)),
        "output_dir": Path(str(training.get("output_dir", "runs/stage3"))),
        "resume_from": None if training.get("resume_from") is None else Path(str(training["resume_from"])),
        "max_records": data.get("max_records"),
        "device": resolve_device(str(device_config.get("type", "auto"))),
        "dtype": torch.float32 if device_config["dtype"] == "float32" else torch.float64,
        "schedule": DDPMNoiseSchedule(ddpm),
        "model_config": model_config,
        "joint_loss": (
            None
            if ddpm.mode != "joint_chemical"
            else JointDiffusionLoss(
                coordinate_weight=ddpm.coordinate_loss_weight,
                atom_weight=ddpm.atom_loss_weight,
                bond_weight=ddpm.bond_loss_weight,
                reduction=ddpm.loss_reduction,
            )
        ),
    }


def _make_model(options: Mapping[str, Any]) -> nn.Module:
    backbone = EquivariantDenoiserBackbone(options["model_config"])
    return FieldConditionedEquivariantDenoiser(
        backbone,
        FieldConditioningConfig(mode="none", channels=()),
    ).to(device=options["device"], dtype=options["dtype"])


def _batch_loss(
    model: nn.Module,
    stage3_input,
    schedule: DDPMNoiseSchedule,
    generator: torch.Generator,
    joint_loss: JointDiffusionLoss | None = None,
) -> Tensor:
    clean = clean_stage3_state(stage3_input)
    timestep = sample_timesteps(clean.batch_size, schedule, device=clean.coordinates.device, generator=generator)
    if schedule.config.mode == "joint_chemical":
        if joint_loss is None:
            raise RuntimeError("joint_chemical training loss is not initialized")
        target = q_sample_joint(clean, timestep, schedule, generator=generator)
        prediction = model(target.noisy_state, timestep)
        return joint_loss(prediction, target).total
    target = q_sample(clean, timestep, schedule, generator=generator)
    prediction = model(target.noisy_state, timestep)
    return NoisePredictionLoss(reduction=schedule.config.loss_reduction)(
        prediction.coordinate_noise,
        target.noise,
        target.noisy_state.mask.atom_mask,
    )


def _epoch(model: nn.Module, loader: DataLoader, options: Mapping[str, Any], epoch: int, training: bool, optimizer: torch.optim.Optimizer | None) -> float:
    phase = "train" if training else "validation"
    total_batches = len(loader)
    log_interval = max(1, total_batches // 10)
    LOGGER.info(
        "Epoch %d/%d %s phase started (batches=%d)",
        epoch + 1,
        options["epochs"],
        phase,
        total_batches,
    )
    model.train(training)
    generator = make_torch_generator(int(options["seed"]) + epoch * 1000003 + int(not training), device=options["device"])
    total = 0.0
    batches = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for records in loader:
            stage3_input = collate_molecules(records, dtype=options["dtype"], device=options["device"])
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            loss = _batch_loss(
                model,
                stage3_input,
                options["schedule"],
                generator,
                options["joint_loss"],
            )
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("non-finite training loss")
            if optimizer is not None:
                loss.backward()
                if options["grad_clip_norm"] > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), options["grad_clip_norm"])
                optimizer.step()
            total += float(loss.detach().cpu())
            batches += 1
            if batches == 1 or batches % log_interval == 0 or batches == total_batches:
                LOGGER.info(
                    "Epoch %d/%d %s progress: batch %d/%d, loss=%.6f",
                    epoch + 1,
                    options["epochs"],
                    phase,
                    batches,
                    total_batches,
                    float(loss.detach().cpu()),
                )
    if batches == 0:
        raise ConfigurationError("split produced no batches")
    mean_loss = total / batches
    LOGGER.info(
        "Epoch %d/%d %s phase finished (mean_loss=%.6f)",
        epoch + 1,
        options["epochs"],
        phase,
        mean_loss,
    )
    return mean_loss


def run_training(config: dict[str, Any]) -> TrainingResult:
    """Train directly from materialized real JSONL records.

    Ring/noise masks are retained in records for audit only. They are not
    passed into the collate mask or loss, so training has no weak constraint.
    """
    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")
    options = _validate_config(config)
    options = {**options, "seed": config["seed"]}
    semantics = generator_semantics(config)
    provenance = training_provenance(config)
    set_seed(int(config["seed"]))
    LOGGER.info(
        "Training setup: data=%s, device=%s, dtype=%s, epochs=%d, batch_size=%d",
        options["data_path"].resolve(),
        options["device"],
        options["dtype"],
        options["epochs"],
        options["batch_size"],
    )
    train_dataset = JsonlMolecularDataset(options["data_path"], max_records=options["max_records"], split="train", train_fraction=options["train_fraction"])
    validation_dataset = JsonlMolecularDataset(options["data_path"], max_records=options["max_records"], split="validation", train_fraction=options["train_fraction"])
    train_loader = DataLoader(train_dataset, batch_size=options["batch_size"], shuffle=False, num_workers=options["num_workers"], collate_fn=list)
    validation_loader = DataLoader(validation_dataset, batch_size=options["batch_size"], shuffle=False, num_workers=options["num_workers"], collate_fn=list)
    LOGGER.info(
        "Datasets ready: train_records=%d, validation_records=%d, train_batches=%d, validation_batches=%d",
        len(train_dataset),
        len(validation_dataset),
        len(train_loader),
        len(validation_loader),
    )
    model = _make_model(options)
    LOGGER.info(
        "Model ready: parameters=%d, diffusion_mode=%s",
        sum(parameter.numel() for parameter in model.parameters()),
        options["schedule"].config.mode,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=options["learning_rate"], weight_decay=options["weight_decay"])
    start_epoch = 0
    if options["resume_from"] is not None:
        LOGGER.info("Loading checkpoint: %s", options["resume_from"].resolve())
        checkpoint = torch.load(options["resume_from"], map_location=options["device"], weights_only=True)
        if not isinstance(checkpoint, Mapping) or not isinstance(checkpoint.get("model_state_dict"), Mapping):
            raise ConfigurationError("resume checkpoint lacks model_state_dict")
        if checkpoint.get("feature_manifest_hash") != options["manifest_hash"]:
            raise ConfigurationError("resume checkpoint feature manifest does not match current data")
        validate_generator_checkpoint(checkpoint, config, resume_provenance=provenance)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        if isinstance(checkpoint.get("optimizer_state_dict"), Mapping):
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        LOGGER.info("Checkpoint loaded; resuming from epoch %d", start_epoch + 1)
    output_dir = options["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    last_checkpoint = output_dir / "last.ckpt"
    best_checkpoint = output_dir / "best.ckpt"
    metrics_path = output_dir / "metrics.jsonl"
    best_validation = float("inf")
    LOGGER.info("Training started: output_dir=%s, start_epoch=%d", output_dir.resolve(), start_epoch + 1)
    with metrics_path.open("a", encoding="utf-8") as metrics_file:
        for epoch in range(start_epoch, options["epochs"]):
            LOGGER.info("Epoch %d/%d started", epoch + 1, options["epochs"])
            train_loss = _epoch(model, train_loader, options, epoch, True, optimizer)
            validation_loss = _epoch(model, validation_loader, options, epoch, False, None)
            metrics = {"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss}
            metrics_file.write(json.dumps(metrics) + "\n")
            metrics_file.flush()
            LOGGER.info(
                "Epoch %d/%d metrics: train_loss=%.6f, validation_loss=%.6f",
                epoch + 1,
                options["epochs"],
                train_loss,
                validation_loss,
            )
            payload = {
                "checkpoint_version": 2,
                "semantics": semantics,
                "training_provenance": provenance,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "seed": config["seed"],
                "feature_manifest_hash": options["manifest_hash"],
                "config": json.loads(json.dumps(config, default=str)),
            }
            torch.save(payload, last_checkpoint)
            LOGGER.info("Epoch %d/%d checkpoint saved: %s", epoch + 1, options["epochs"], last_checkpoint.resolve())
            if validation_loss < best_validation:
                best_validation = validation_loss
                torch.save(payload, best_checkpoint)
                LOGGER.info(
                    "Epoch %d/%d new best checkpoint saved: validation_loss=%.6f, path=%s",
                    epoch + 1,
                    options["epochs"],
                    validation_loss,
                    best_checkpoint.resolve(),
                )
    save_json({"epochs_completed": options["epochs"], "train_records": len(train_dataset), "validation_records": len(validation_dataset), "last_checkpoint": str(last_checkpoint), "best_checkpoint": str(best_checkpoint), "training_provenance": provenance}, output_dir / "summary.json")
    LOGGER.info(
        "Training finished: epochs_completed=%d, best_validation_loss=%.6f, summary=%s",
        options["epochs"],
        best_validation,
        (output_dir / "summary.json").resolve(),
    )
    return TrainingResult(output_dir, last_checkpoint, best_checkpoint, metrics_path, options["epochs"])


__all__ = ["Trainer", "TrainingResult", "run_training"]
