"""Training loop for the independent differentiable RA reward surrogate."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn
from torch.utils.data import DataLoader

from gpcr_ringgen.core.exceptions import ConfigurationError
from gpcr_ringgen.checkpoint_semantics import (
    label_cache_provenance,
    reward_semantics,
    training_provenance,
    validate_reward_metadata,
)
from gpcr_ringgen.data import collate_molecules
from gpcr_ringgen.models.diffusion import clean_stage3_state
from gpcr_ringgen.training.ra_reward import (
    RARewardModel,
    RARewardModelConfig,
    save_ra_reward_checkpoint,
)
from gpcr_ringgen.training.ra_reward_data import (
    RAScoreLabeledDataset,
    collate_ra_records,
)
from gpcr_ringgen.utils.device import resolve_device
from gpcr_ringgen.utils.io import save_json
from gpcr_ringgen.utils.seed import set_seed


@dataclass(frozen=True)
class RARewardTrainingResult:
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
    return int(value)


def _nonnegative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ConfigurationError(f"{name} must be a non-negative integer")
    return int(value)


def _finite_nonnegative(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{name} must be a finite non-negative number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ConfigurationError(f"{name} must be a finite non-negative number")
    return result


def _validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    data = _section(config, "data")
    training = _section(config, "training")
    device_config = _section(config, "device")
    records_path = data.get("path")
    labels_path = data.get("labels_path")
    if not isinstance(records_path, str) or not Path(records_path).is_file():
        raise ConfigurationError(f"data.path not found: {records_path}")
    if not isinstance(labels_path, str) or not Path(labels_path).is_file():
        raise ConfigurationError(f"data.labels_path not found: {labels_path}")
    if device_config.get("dtype") not in {"float32", "float64"}:
        raise ConfigurationError("device.dtype must be float32 or float64")
    if not isinstance(config.get("seed"), int) or isinstance(config.get("seed"), bool):
        raise ConfigurationError("top-level seed must be an integer")
    train_fraction = float(training.get("train_fraction", 0.9))
    if not 0.0 < train_fraction < 1.0:
        raise ConfigurationError("training.train_fraction must be between 0 and 1")
    return {
        "records_path": Path(records_path),
        "labels_path": Path(labels_path),
        "epochs": _positive_int("training.epochs", training.get("epochs", 1)),
        "batch_size": _positive_int("training.batch_size", training.get("batch_size", 1)),
        "num_workers": _nonnegative_int("training.num_workers", training.get("num_workers", 0)),
        "train_fraction": train_fraction,
        "learning_rate": _finite_nonnegative("training.learning_rate", training.get("learning_rate", 1.0e-3)),
        "weight_decay": _finite_nonnegative("training.weight_decay", training.get("weight_decay", 0.0)),
        "output_dir": Path(str(training.get("output_dir", "runs/ra_reward"))),
        "max_records": data.get("max_records"),
        "device": resolve_device(str(device_config.get("type", "auto"))),
        "dtype": torch.float32 if device_config["dtype"] == "float32" else torch.float64,
        "model_config": RARewardModelConfig.from_mapping(config),
    }


def _epoch(
    model: RARewardModel,
    loader: DataLoader,
    options: Mapping[str, Any],
    optimizer: torch.optim.Optimizer | None,
) -> float:
    training = optimizer is not None
    model.train(training)
    total = 0.0
    batches = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for records, targets in loader:
            stage3_input = collate_molecules(
                records,
                dtype=options["dtype"],
                device=options["device"],
            )
            state = clean_stage3_state(stage3_input)
            targets = targets.to(device=options["device"], dtype=options["dtype"])
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            prediction = model(state)
            loss = nn.functional.smooth_l1_loss(prediction, targets)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("non-finite RA reward loss")
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            total += float(loss.detach().cpu())
            batches += 1
    if batches == 0:
        raise ConfigurationError("RA reward split produced no batches")
    return total / batches


def run_ra_reward_training(config: Mapping[str, Any]) -> RARewardTrainingResult:
    """Train the reward surrogate from materialized records and a label cache."""
    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")
    options = _validate_config(config)
    semantics = reward_semantics(config, options["model_config"].to_mapping())
    provenance = {**training_provenance(config), **label_cache_provenance(options["labels_path"])}
    validate_reward_metadata(semantics, provenance, options["model_config"].to_mapping())
    set_seed(int(config["seed"]))
    train_dataset = RAScoreLabeledDataset(
        options["records_path"],
        options["labels_path"],
        split="train",
        train_fraction=options["train_fraction"],
        max_records=options["max_records"],
    )
    validation_dataset = RAScoreLabeledDataset(
        options["records_path"],
        options["labels_path"],
        split="validation",
        train_fraction=options["train_fraction"],
        max_records=options["max_records"],
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=options["batch_size"],
        shuffle=True,
        num_workers=options["num_workers"],
        collate_fn=collate_ra_records,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=options["batch_size"],
        shuffle=False,
        num_workers=options["num_workers"],
        collate_fn=collate_ra_records,
    )
    model = RARewardModel(options["model_config"]).to(
        device=options["device"],
        dtype=options["dtype"],
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=options["learning_rate"],
        weight_decay=options["weight_decay"],
    )
    output_dir = options["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    last_checkpoint = output_dir / "last.ckpt"
    best_checkpoint = output_dir / "best.ckpt"
    metrics_path = output_dir / "metrics.jsonl"
    best_validation = float("inf")
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        for epoch in range(options["epochs"]):
            train_loss = _epoch(model, train_loader, options, optimizer)
            validation_loss = _epoch(model, validation_loader, options, None)
            metrics = {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
            }
            metrics_file.write(json.dumps(metrics) + "\n")
            metrics_file.flush()
            save_ra_reward_checkpoint(
                model,
                last_checkpoint,
                epoch=epoch,
                semantics=semantics,
                provenance=provenance,
                optimizer=optimizer,
                metrics=metrics,
            )
            if validation_loss < best_validation:
                best_validation = validation_loss
                save_ra_reward_checkpoint(
                    model,
                    best_checkpoint,
                    epoch=epoch,
                    semantics=semantics,
                    provenance=provenance,
                    optimizer=optimizer,
                    metrics=metrics,
                )
    save_json(
        {
            "epochs_completed": options["epochs"],
            "train_records": len(train_dataset),
            "validation_records": len(validation_dataset),
            "training_provenance": provenance,
            "last_checkpoint": str(last_checkpoint),
            "best_checkpoint": str(best_checkpoint),
        },
        output_dir / "summary.json",
    )
    return RARewardTrainingResult(
        output_dir,
        last_checkpoint,
        best_checkpoint,
        metrics_path,
        options["epochs"],
    )


__all__ = ["RARewardTrainingResult", "run_ra_reward_training"]
