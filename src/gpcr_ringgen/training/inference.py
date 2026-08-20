"""Pocket-disabled inference directly from a materialized training record."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from gpcr_ringgen.core.exceptions import ConfigurationError
from gpcr_ringgen.data import FeatureManifest, JsonlMolecularDataset, collate_molecules
from gpcr_ringgen.models.diffusion import DDPMConfig, DDPMNoiseSchedule
from gpcr_ringgen.models.equivariant import (
    EquivariantDenoiserBackbone,
    EquivariantDenoiserConfig,
    FieldConditionedEquivariantDenoiser,
    FieldConditioningConfig,
)
from gpcr_ringgen.sampling import DDPMStage3Sampler
from gpcr_ringgen.utils.device import resolve_device
from gpcr_ringgen.utils.io import save_json
from gpcr_ringgen.utils.seed import set_seed


def _state_dict(checkpoint: object) -> Mapping[str, torch.Tensor]:
    if not isinstance(checkpoint, Mapping):
        raise ConfigurationError("checkpoint must be a mapping")
    candidate = checkpoint.get("model_state_dict", checkpoint.get("state_dict"))
    if not isinstance(candidate, Mapping):
        raise ConfigurationError("checkpoint has no model_state_dict")
    return candidate


def run_record_inference(config: Mapping[str, Any]) -> Path:
    """Load one real JSONL record, restore a checkpoint, and sample coordinates."""
    data = config.get("data")
    inference = config.get("inference")
    device_config = config.get("device")
    if not isinstance(data, Mapping) or not isinstance(inference, Mapping) or not isinstance(device_config, Mapping):
        raise ConfigurationError("data, inference, and device sections are required")
    manifest_path = Path(str(data["feature_manifest"]))
    manifest = FeatureManifest.from_file(manifest_path)
    model_config = EquivariantDenoiserConfig.from_mapping(config)
    if model_config.atom_feature_dim != manifest.atom_feature_dim or model_config.edge_feature_dim != manifest.edge_feature_dim:
        raise ConfigurationError("model dimensions do not match feature manifest")
    if model_config.descriptor_dim is not None or manifest.descriptor_dim != 0:
        raise ConfigurationError("pocket-disabled inference requires descriptor_dim=null and no descriptors")
    device = resolve_device(str(device_config.get("type", "auto")))
    dtype = torch.float32 if device_config.get("dtype") == "float32" else torch.float64
    set_seed(int(config["seed"]))
    dataset = JsonlMolecularDataset(data["path"])
    index = int(inference.get("record_index", 0))
    record = dataset[index]
    stage3_input = collate_molecules([record], dtype=dtype, device=device)
    model = FieldConditionedEquivariantDenoiser(
        EquivariantDenoiserBackbone(model_config),
        FieldConditioningConfig(mode="none", channels=()),
    ).to(device=device, dtype=dtype)
    checkpoint_path = Path(str(inference["checkpoint"]))
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    expected_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if isinstance(checkpoint, Mapping) and checkpoint.get("feature_manifest_hash") != expected_hash:
        raise ConfigurationError("checkpoint feature manifest does not match inference data")
    model.load_state_dict(_state_dict(checkpoint), strict=True)
    schedule = DDPMNoiseSchedule(DDPMConfig.from_mapping(config))
    output = DDPMStage3Sampler(schedule).sample(stage3_input, model, seed=int(config["seed"]), return_trajectory=False)
    state = output.final_state
    payload = {
        "pipeline": "stage3_pocket_disabled_record_inference",
        "scientific_status": "model_output_not_scientifically_evaluated",
        "checkpoint": str(checkpoint_path),
        "sample_id": record.sample_id,
        "coordinates": state.coordinates[0].detach().cpu().tolist(),
        "atom_mask": state.mask.atom_mask[0].detach().cpu().tolist(),
        "edge_index": state.edge_index[0].detach().cpu().tolist(),
        "edge_mask": state.mask.edge_mask[0].detach().cpu().tolist(),
        "metadata": record.metadata,
    }
    output_path = Path(str(inference["output"]))
    save_json(payload, output_path)
    return output_path


__all__ = ["run_record_inference"]
