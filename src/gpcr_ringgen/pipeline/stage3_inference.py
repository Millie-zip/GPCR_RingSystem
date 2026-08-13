"""Orchestrate the existing modules into independent Stage 3 inference."""

from __future__ import annotations

import pickle
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from gpcr_ringgen.core.exceptions import ConfigurationError
from gpcr_ringgen.data import (
    load_stage3_inference_request,
    stage2_output_to_stage3_input,
)
from gpcr_ringgen.models.diffusion import DDPMConfig, DDPMNoiseSchedule
from gpcr_ringgen.models.equivariant import (
    EquivariantDenoiserBackbone,
    EquivariantDenoiserConfig,
    FieldConditionedEquivariantDenoiser,
    FieldConditioningConfig,
)
from gpcr_ringgen.pocket import (
    GeometryFieldConfig,
    ProcessedPocketBatch,
    QueryablePocketField,
    build_geometry_field,
    preprocess_pocket,
)
from gpcr_ringgen.sampling import (
    DDPMStage3Sampler,
    Stage3SamplingOutput,
    restore_global_coordinates,
)
from gpcr_ringgen.utils.device import resolve_device
from gpcr_ringgen.utils.io import load_yaml_config, save_json
from gpcr_ringgen.utils.seed import set_seed


def _config_mapping(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = config.get(name)
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{name} config must be a mapping")
    return value


def _path_value(value: object, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{name} must be a non-empty path string")
    return Path(value)


@dataclass(frozen=True)
class Stage3InferencePipelineConfig:
    """Runtime and path settings for the independent Stage 3 pipeline."""

    input_path: Path
    checkpoint_path: Path | None
    output_path: Path
    allow_random_weights: bool
    seed: int
    device: str
    dtype: str
    return_trajectory: bool

    def __post_init__(self) -> None:
        if not isinstance(self.input_path, Path):
            raise TypeError("input_path must be a pathlib.Path")
        if self.checkpoint_path is not None and not isinstance(
            self.checkpoint_path,
            Path,
        ):
            raise TypeError("checkpoint_path must be a pathlib.Path or None")
        if not isinstance(self.output_path, Path):
            raise TypeError("output_path must be a pathlib.Path")
        if not isinstance(self.allow_random_weights, bool):
            raise TypeError("allow_random_weights must be a bool")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("device must be a non-empty string")
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("dtype must be 'float32' or 'float64'")
        if not isinstance(self.return_trajectory, bool):
            raise TypeError("return_trajectory must be a bool")
        if self.checkpoint_path is None and not self.allow_random_weights:
            raise ConfigurationError(
                "checkpoint_path is null; set allow_random_weights=true only for "
                "an explicit structural smoke test"
            )

    @property
    def torch_dtype(self) -> torch.dtype:
        """Return the configured PyTorch floating dtype."""

        return torch.float32 if self.dtype == "float32" else torch.float64

    @classmethod
    def from_mapping(
        cls,
        config: Mapping[str, Any],
    ) -> "Stage3InferencePipelineConfig":
        """Parse paths, runtime, seed, and sampling controls."""

        pipeline = _config_mapping(config, "pipeline")
        device = _config_mapping(config, "device")
        sampling = _config_mapping(config, "sampling")
        required_pipeline = (
            "input_path",
            "checkpoint_path",
            "output_path",
            "allow_random_weights",
        )
        missing = [name for name in required_pipeline if name not in pipeline]
        if missing:
            raise ConfigurationError(
                "missing pipeline config keys: " + ", ".join(missing)
            )
        if "seed" not in config:
            raise ConfigurationError("missing top-level seed")
        if "type" not in device or "dtype" not in device:
            raise ConfigurationError("device config requires type and dtype")
        if "return_trajectory" not in sampling:
            raise ConfigurationError("sampling config requires return_trajectory")
        checkpoint_value = pipeline["checkpoint_path"]
        checkpoint_path = (
            None
            if checkpoint_value is None
            else _path_value(checkpoint_value, "pipeline.checkpoint_path")
        )
        return cls(
            input_path=_path_value(pipeline["input_path"], "pipeline.input_path"),
            checkpoint_path=checkpoint_path,
            output_path=_path_value(
                pipeline["output_path"],
                "pipeline.output_path",
            ),
            allow_random_weights=pipeline["allow_random_weights"],
            seed=config["seed"],
            device=device["type"],
            dtype=device["dtype"],
            return_trajectory=sampling["return_trajectory"],
        )


@dataclass(frozen=True)
class Stage3InferenceRun:
    """In-memory artifacts and saved path from one pipeline execution.

    Tensor fields:
        global_coordinates: Shape ``[B, N, 3]``; same floating dtype/device as
            sampled local coordinates. Valid atoms have the saved pocket center
            added; padded atoms are zero.
    """

    output_path: Path
    sampling_output: Stage3SamplingOutput
    processed_pocket: ProcessedPocketBatch
    pocket_field: QueryablePocketField
    global_coordinates: Tensor
    checkpoint_loaded: bool
    random_weights: bool


def _checkpoint_state_dict(
    checkpoint: object,
) -> Mapping[str, Tensor]:
    if not isinstance(checkpoint, Mapping):
        raise ConfigurationError("checkpoint must contain a model state mapping")
    candidate: object = checkpoint
    for key in ("model_state_dict", "state_dict"):
        if key in checkpoint:
            candidate = checkpoint[key]
            break
    if not isinstance(candidate, Mapping):
        raise ConfigurationError("checkpoint state_dict must be a mapping")
    if any(not isinstance(key, str) for key in candidate):
        raise ConfigurationError("checkpoint state_dict keys must be strings")
    if any(not isinstance(value, Tensor) for value in candidate.values()):
        raise ConfigurationError("checkpoint state_dict values must be tensors")
    return candidate


def _load_model_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    *,
    device: torch.device,
) -> None:
    if not checkpoint_path.is_file():
        raise ConfigurationError(f"checkpoint not found: {checkpoint_path}")
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )
        model.load_state_dict(_checkpoint_state_dict(checkpoint), strict=True)
    except (EOFError, OSError, pickle.UnpicklingError, RuntimeError, TypeError, ValueError) as exc:
        raise ConfigurationError(
            f"checkpoint is incompatible with the configured Stage 3 model: {exc}"
        ) from exc


def _serialize_output(
    *,
    config: Mapping[str, Any],
    config_reference: str | Path | None,
    pipeline_config: Stage3InferencePipelineConfig,
    sampling_output: Stage3SamplingOutput,
    processed_pocket: ProcessedPocketBatch,
    pocket_field: QueryablePocketField,
    global_coordinates: Tensor,
    checkpoint_loaded: bool,
) -> dict[str, Any]:
    samples: list[dict[str, Any]] = []
    state = sampling_output.final_state
    for batch_index, sample_id in enumerate(state.metadata.sample_ids):
        sample: dict[str, Any] = {
            "sample_id": sample_id,
            "coordinate_frame": "pocket_local_and_restored_global",
            "local_coordinates": state.coordinates[batch_index].detach().cpu().tolist(),
            "global_coordinates": global_coordinates[batch_index]
            .detach()
            .cpu()
            .tolist(),
            "atom_mask": state.mask.atom_mask[batch_index].detach().cpu().tolist(),
            "descriptor": state.descriptor[batch_index].detach().cpu().tolist(),
            "descriptor_mask": state.mask.descriptor_mask[batch_index]
            .detach()
            .cpu()
            .tolist(),
            "topology": {
                "atom_features": state.atom_features[batch_index]
                .detach()
                .cpu()
                .tolist(),
                "edge_index": state.edge_index[batch_index]
                .detach()
                .cpu()
                .tolist(),
                "edge_features": state.edge_features[batch_index]
                .detach()
                .cpu()
                .tolist(),
                "edge_mask": state.mask.edge_mask[batch_index]
                .detach()
                .cpu()
                .tolist(),
            },
            "pocket_transform": {
                "kind": processed_pocket.transform.kind,
                "center": processed_pocket.transform.center[batch_index]
                .detach()
                .cpu()
                .tolist(),
            },
            "metadata": dict(state.metadata.records[batch_index]),
        }
        if sampling_output.trajectory is not None:
            sample["trajectory_local_coordinates"] = [
                coordinates[batch_index].detach().cpu().tolist()
                for coordinates in sampling_output.trajectory
            ]
        samples.append(sample)

    random_weights = not checkpoint_loaded
    return {
        "pipeline": "stage3_independent_inference",
        "model_weights": "random_untrained" if random_weights else "checkpoint",
        "scientific_status": (
            "pipeline_smoke_test_only"
            if random_weights
            else "model_output_not_scientifically_evaluated"
        ),
        "checkpoint_path": (
            None
            if pipeline_config.checkpoint_path is None
            else str(pipeline_config.checkpoint_path)
        ),
        "seed": pipeline_config.seed,
        "device": str(state.coordinates.device),
        "dtype": str(state.coordinates.dtype),
        "field_channels": list(pocket_field.channels),
        "num_timesteps": len(sampling_output.trajectory) - 1
        if sampling_output.trajectory is not None
        else DDPMConfig.from_mapping(config).num_timesteps,
        "trajectory_saved": sampling_output.trajectory is not None,
        "config_reference": None if config_reference is None else str(config_reference),
        "config_snapshot": dict(config),
        "samples": samples,
    }


def run_stage3_inference(
    config: Mapping[str, Any],
    *,
    config_reference: str | Path | None = None,
) -> Stage3InferenceRun:
    """Run the independent Stage 3 orchestration without Stage 1/Stage 2 models.

    This function delegates pocket centering, Field construction, Stage 2
    adaptation, Gaussian initialization, reverse diffusion, coordinate inverse
    transformation, and JSON serialization to their existing public APIs.
    """

    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")
    pipeline_config = Stage3InferencePipelineConfig.from_mapping(config)
    device = resolve_device(pipeline_config.device)
    request = load_stage3_inference_request(
        pipeline_config.input_path,
        dtype=pipeline_config.torch_dtype,
        device=device,
    )

    processed_pocket = preprocess_pocket(
        request.pocket,
        pocket_mask=request.pocket_mask,
        atom_metadata=request.pocket_atom_metadata,
        metadata=request.stage2_output.metadata,
    )
    field_config = GeometryFieldConfig.from_mapping(config)
    pocket_field = build_geometry_field(processed_pocket, field_config)
    conditioning_config = FieldConditioningConfig.from_mapping(config)
    stage3_input = stage2_output_to_stage3_input(
        request.stage2_output,
        descriptor=request.descriptor,
        pocket=processed_pocket.pocket,
        descriptor_mask=request.descriptor_mask,
        pocket_mask=processed_pocket.pocket_mask,
        pocket_field=(
            pocket_field if conditioning_config.mode == "additive" else None
        ),
    )

    set_seed(pipeline_config.seed)
    backbone = EquivariantDenoiserBackbone(
        EquivariantDenoiserConfig.from_mapping(config)
    )
    model = FieldConditionedEquivariantDenoiser(
        backbone,
        conditioning_config,
    ).to(device=device, dtype=pipeline_config.torch_dtype)
    checkpoint_loaded = pipeline_config.checkpoint_path is not None
    if pipeline_config.checkpoint_path is not None:
        _load_model_checkpoint(
            model,
            pipeline_config.checkpoint_path,
            device=device,
        )

    schedule = DDPMNoiseSchedule(DDPMConfig.from_mapping(config))
    sampling_output = DDPMStage3Sampler(schedule).sample(
        stage3_input,
        model,
        seed=pipeline_config.seed,
        pocket_field=(
            pocket_field if conditioning_config.mode == "additive" else None
        ),
        return_trajectory=pipeline_config.return_trajectory,
    )
    global_coordinates = restore_global_coordinates(
        sampling_output,
        processed_pocket.transform,
    )
    payload = _serialize_output(
        config=config,
        config_reference=config_reference,
        pipeline_config=pipeline_config,
        sampling_output=sampling_output,
        processed_pocket=processed_pocket,
        pocket_field=pocket_field,
        global_coordinates=global_coordinates,
        checkpoint_loaded=checkpoint_loaded,
    )
    output_path = save_json(payload, pipeline_config.output_path)
    return Stage3InferenceRun(
        output_path=output_path,
        sampling_output=sampling_output,
        processed_pocket=processed_pocket,
        pocket_field=pocket_field,
        global_coordinates=global_coordinates,
        checkpoint_loaded=checkpoint_loaded,
        random_weights=not checkpoint_loaded,
    )


def run_stage3_inference_from_config(path: str | Path) -> Stage3InferenceRun:
    """Load one YAML configuration and execute independent Stage 3 inference."""

    return run_stage3_inference(load_yaml_config(path), config_reference=path)


__all__ = [
    "Stage3InferencePipelineConfig",
    "Stage3InferenceRun",
    "run_stage3_inference",
    "run_stage3_inference_from_config",
]
