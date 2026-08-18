"""Plain PyTorch training orchestration for the real Stage 3 asset library."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Subset

from gpcr_ringgen.core.exceptions import ConfigurationError
from gpcr_ringgen.data.collate import collate_molecules
from gpcr_ringgen.data.datasets import JsonlMolecularDataset
from gpcr_ringgen.data.manifest import FeatureManifest
from gpcr_ringgen.models.diffusion.chemical import q_sample_joint
from gpcr_ringgen.models.diffusion.schedules import DDPMConfig, DDPMNoiseSchedule
from gpcr_ringgen.models.diffusion.state import clean_stage3_state
from gpcr_ringgen.models.diffusion.transitions import q_sample, sample_timesteps
from gpcr_ringgen.models.diffusion.vocabulary import ChemicalVocabulary
from gpcr_ringgen.models.equivariant.backbone import (
    DenoiserOutput,
    EquivariantDenoiserBackbone,
    EquivariantDenoiserConfig,
)
from gpcr_ringgen.models.equivariant.field_conditioning import (
    FieldConditionedEquivariantDenoiser,
    FieldConditioningConfig,
)
from gpcr_ringgen.training.losses import JointDiffusionLoss, NoisePredictionLoss
from gpcr_ringgen.utils.device import resolve_device
from gpcr_ringgen.utils.io import save_json
from gpcr_ringgen.utils.seed import make_torch_generator, set_seed


class Trainer(nn.Module):
    """Compatibility base class for future specialized trainers."""

    def fit(self, config: dict[str, Any]) -> Any:
        raise NotImplementedError


@dataclass(frozen=True)
class TrainingResult:
    """Paths and final metrics from one training run."""

    output_dir: Path
    last_checkpoint: Path
    best_checkpoint: Path
    metrics_path: Path
    epochs_completed: int


def _section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = config.get(name)
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{name} must be a mapping")
    return value


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigurationError(f"{name} must be a positive integer")
    return value


def _non_negative_float(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{name} must be a non-negative number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ConfigurationError(f"{name} must be a finite non-negative number")
    return result


def _feature_manifest_path(config: Mapping[str, Any]) -> Path:
    data = _section(config, "data")
    value = data.get("feature_manifest")
    if value is None:
        raise ConfigurationError("data.feature_manifest is required")
    path = Path(str(value))
    if not path.is_file():
        raise ConfigurationError(f"feature manifest not found: {path}")
    return path


def _feature_manifest_hash(config: Mapping[str, Any]) -> str:
    path = _feature_manifest_path(config)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_training_config(config: Mapping[str, Any]) -> dict[str, Any]:
    data = _section(config, "data")
    training = _section(config, "training")
    if not data.get("path"):
        raise ConfigurationError("data.path is required")
    epochs = _positive_int("training.epochs", training.get("epochs", 1))
    batch_size = _positive_int("training.batch_size", training.get("batch_size", 1))
    num_workers = training.get("num_workers", 0)
    if isinstance(num_workers, bool) or not isinstance(num_workers, int) or num_workers < 0:
        raise ConfigurationError("training.num_workers must be a non-negative integer")
    train_fraction = float(training.get("train_fraction", 0.9))
    val_fraction = float(training.get("validation_fraction", 0.1))
    if not (0.0 < train_fraction <= 1.0 and 0.0 <= val_fraction < 1.0):
        raise ConfigurationError("train/validation fractions are invalid")
    if train_fraction + val_fraction > 1.0:
        raise ConfigurationError("train_fraction + validation_fraction must be <= 1")
    if config.get("seed") is None or isinstance(config["seed"], bool) or not isinstance(config["seed"], int):
        raise ConfigurationError("top-level seed must be an integer")
    device = _section(config, "device")
    if device.get("type") is None or device.get("dtype") not in {"float32", "float64"}:
        raise ConfigurationError("device.type and device.dtype must be configured")
    ddpm = DDPMConfig.from_mapping(config)
    model = EquivariantDenoiserConfig.from_mapping(config)
    manifest_path = _feature_manifest_path(config)
    manifest = FeatureManifest.from_file(manifest_path)
    if model.atom_feature_dim != manifest.atom_feature_dim:
        raise ConfigurationError("model.atom_feature_dim does not match feature manifest")
    if model.edge_feature_dim != manifest.edge_feature_dim:
        raise ConfigurationError("model.edge_feature_dim does not match feature manifest")
    if (model.descriptor_dim or 0) != manifest.descriptor_dim:
        raise ConfigurationError("model.descriptor_dim does not match feature manifest")
    if manifest.pocket_feature_dim:
        raise ConfigurationError(
            "training requires an empty pocket feature list; pocket features are disabled"
        )
    if ddpm.mode == "joint_chemical":
        vocabulary = ChemicalVocabulary.from_mapping(config)
        if model.num_atom_types != ddpm.num_atom_types or model.num_bond_types != ddpm.num_bond_types:
            raise ConfigurationError("model and diffusion categorical dimensions must match")
        if vocabulary.num_atom_types != ddpm.num_atom_types or vocabulary.num_bond_types != ddpm.num_bond_types:
            raise ConfigurationError("chemical vocabulary and diffusion dimensions must match")
    field_conditioning = config.get("field_conditioning", {"mode": "none", "channels": []})
    if not isinstance(field_conditioning, Mapping) or field_conditioning.get("mode", "none") != "none":
        raise ConfigurationError("training currently requires field_conditioning.mode=none")
    return {
        "epochs": epochs,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "train_fraction": train_fraction,
        "validation_fraction": val_fraction,
        "learning_rate": _non_negative_float("training.learning_rate", training.get("learning_rate", 1e-3)),
        "weight_decay": _non_negative_float("training.weight_decay", training.get("weight_decay", 0.0)),
        "grad_clip_norm": _non_negative_float("training.grad_clip_norm", training.get("grad_clip_norm", 1.0)),
        "output_dir": Path(str(training.get("output_dir", "runs/stage3"))),
        "resume_from": None if training.get("resume_from") is None else Path(str(training["resume_from"])),
        "max_records": None if data.get("max_records") is None else _positive_int("data.max_records", data["max_records"]),
        "dtype": torch.float32 if device["dtype"] == "float32" else torch.float64,
        "device": resolve_device(str(device["type"])),
        "schedule": DDPMNoiseSchedule(ddpm),
        "mode": ddpm.mode,
        "model_config": model,
        "feature_manifest": manifest,
        "feature_manifest_path": manifest_path,
    }


def _make_model(model_config: EquivariantDenoiserConfig, device: torch.device, dtype: torch.dtype) -> nn.Module:
    backbone = EquivariantDenoiserBackbone(model_config).to(device=device, dtype=dtype)
    # Keep the checkpoint namespace identical to the inference pipeline.  The
    # ``none`` wrapper is an exact pass-through and does not add a weak
    # constraint or pocket signal to the training path.
    return FieldConditionedEquivariantDenoiser(
        backbone,
        FieldConditioningConfig(mode="none", channels=()),
    )


def _batch_loss(
    model: nn.Module,
    stage3_input,
    schedule: DDPMNoiseSchedule,
    mode: str,
    *,
    generator: torch.Generator,
) -> tuple[Tensor, dict[str, float]]:
    clean_state = clean_stage3_state(stage3_input)
    timestep = sample_timesteps(
        clean_state.batch_size,
        schedule,
        device=clean_state.coordinates.device,
        generator=generator,
    )
    if mode == "joint_chemical":
        target = q_sample_joint(clean_state, timestep, schedule, generator=generator)
        prediction: DenoiserOutput = model(target.noisy_state, timestep)
        output = JointDiffusionLoss(
            coordinate_weight=schedule.config.coordinate_loss_weight,
            atom_weight=schedule.config.atom_loss_weight,
            bond_weight=schedule.config.bond_loss_weight,
            reduction=schedule.config.loss_reduction,
        )(prediction, target)
        return output.total, {
            "total": float(output.total.detach().cpu()),
            "coordinate": float(output.coordinate.detach().cpu()),
            "atom": float(output.atom.detach().cpu()),
            "bond": float(output.bond.detach().cpu()),
        }
    target = q_sample(clean_state, timestep, schedule, generator=generator)
    prediction = model(target.noisy_state, timestep)
    coordinate = NoisePredictionLoss(reduction=schedule.config.loss_reduction)(
        prediction.coordinate_noise,
        target.noise,
        target.noisy_state.mask.atom_mask,
    )
    return coordinate, {
        "total": float(coordinate.detach().cpu()),
        "coordinate": float(coordinate.detach().cpu()),
    }


def _checkpoint_payload(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    epoch: int,
    global_step: int,
    metrics: Mapping[str, float],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "checkpoint_version": 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": None if scheduler is None else scheduler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "metrics": dict(metrics),
        "seed": config["seed"],
        "feature_manifest_hash": _feature_manifest_hash(config),
        "config": json.loads(json.dumps(config, default=str)),
    }


def _run_epoch(
    *,
    model: nn.Module,
    loader: DataLoader,
    schedule: DDPMNoiseSchedule,
    mode: str,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    optimizer: torch.optim.Optimizer | None,
    grad_clip_norm: float,
    epoch: int,
    training: bool,
) -> dict[str, float]:
    model.train(training)
    totals: dict[str, float] = {}
    count = 0
    generator = make_torch_generator(seed + epoch * 1_000_003 + (0 if training else 1), device=device)
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for records in loader:
            stage3_input = collate_molecules(records, dtype=dtype, device=device)
            if training and optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            loss, metrics = _batch_loss(
                model,
                stage3_input,
                schedule,
                mode,
                generator=generator,
            )
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("training produced a non-finite loss")
            if training and optimizer is not None:
                loss.backward()
                if grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value
            count += 1
    if count == 0:
        raise RuntimeError("training loader produced no batches")
    return {key: value / count for key, value in totals.items()}


def run_training(config: dict[str, Any]) -> TrainingResult:
    """Train Stage 3 from a converted JSONL asset library.

    No ring, geometry, chemistry, or pocket weak constraint is applied here.
    Such checks belong to post-training evaluation and test-time validation.
    """

    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")
    options = _validate_training_config(config)
    set_seed(int(config["seed"]))
    dataset = JsonlMolecularDataset(
        _section(config, "data")["path"],
        max_records=options["max_records"],
    )
    if len(dataset) < 2:
        raise ConfigurationError("at least two records are required for train/validation")
    train_count = max(1, int(len(dataset) * options["train_fraction"]))
    val_count = int(len(dataset) * options["validation_fraction"])
    if val_count == 0:
        val_count = 1
    if train_count + val_count > len(dataset):
        train_count = len(dataset) - val_count
    if train_count < 1:
        raise ConfigurationError("training split is empty")
    generator = torch.Generator(device="cpu").manual_seed(int(config["seed"]))
    permutation = torch.randperm(len(dataset), generator=generator).tolist()
    train_indices = permutation[:train_count]
    val_indices = permutation[train_count : train_count + val_count]
    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=options["batch_size"],
        shuffle=False,
        num_workers=options["num_workers"],
        collate_fn=list,
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices),
        batch_size=options["batch_size"],
        shuffle=False,
        num_workers=options["num_workers"],
        collate_fn=list,
    )
    model = _make_model(options["model_config"], options["device"], options["dtype"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=options["learning_rate"],
        weight_decay=options["weight_decay"],
    )
    scheduler = None
    start_epoch = 0
    global_step = 0
    if options["resume_from"] is not None:
        checkpoint = torch.load(options["resume_from"], map_location=options["device"], weights_only=True)
        if not isinstance(checkpoint, Mapping) or not isinstance(checkpoint.get("model_state_dict"), Mapping):
            raise ConfigurationError("resume checkpoint does not contain model_state_dict")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        if isinstance(checkpoint.get("optimizer_state_dict"), Mapping):
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        global_step = int(checkpoint.get("global_step", 0))
    output_dir = options["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    last_checkpoint = output_dir / "last.ckpt"
    best_checkpoint = output_dir / "best.ckpt"
    best_value = float("inf")
    with metrics_path.open("a", encoding="utf-8") as metrics_file:
        for epoch in range(start_epoch, options["epochs"]):
            train_metrics = _run_epoch(
                model=model,
                loader=train_loader,
                schedule=options["schedule"],
                mode=options["mode"],
                device=options["device"],
                dtype=options["dtype"],
                seed=int(config["seed"]),
                optimizer=optimizer,
                grad_clip_norm=options["grad_clip_norm"],
                epoch=epoch,
                training=True,
            )
            val_metrics = _run_epoch(
                model=model,
                loader=val_loader,
                schedule=options["schedule"],
                mode=options["mode"],
                device=options["device"],
                dtype=options["dtype"],
                seed=int(config["seed"]),
                optimizer=None,
                grad_clip_norm=0.0,
                epoch=epoch,
                training=False,
            )
            global_step += len(train_loader)
            payload = {
                "epoch": epoch,
                "global_step": global_step,
                "train": train_metrics,
                "validation": val_metrics,
            }
            metrics_file.write(json.dumps(payload, sort_keys=True) + "\n")
            metrics_file.flush()
            checkpoint_payload = _checkpoint_payload(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                global_step=global_step,
                metrics={**{f"train_{key}": value for key, value in train_metrics.items()}, **{f"validation_{key}": value for key, value in val_metrics.items()}},
                config=config,
            )
            torch.save(checkpoint_payload, last_checkpoint)
            if val_metrics["total"] < best_value:
                best_value = val_metrics["total"]
                torch.save(checkpoint_payload, best_checkpoint)
    save_json(
        {
            "epochs_completed": options["epochs"],
            "train_records": train_count,
            "validation_records": val_count,
            "mode": options["mode"],
            "last_checkpoint": str(last_checkpoint),
            "best_checkpoint": str(best_checkpoint),
        },
        output_dir / "summary.json",
    )
    return TrainingResult(
        output_dir=output_dir,
        last_checkpoint=last_checkpoint,
        best_checkpoint=best_checkpoint,
        metrics_path=metrics_path,
        epochs_completed=options["epochs"],
    )
