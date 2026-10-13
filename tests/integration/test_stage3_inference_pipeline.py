"""End-to-end tests for independent Stage 3 inference orchestration."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch
import yaml

from gpcr_ringgen.core.exceptions import ConfigurationError
from gpcr_ringgen.pipeline import (
    Stage3InferencePipelineConfig,
    run_stage3_inference,
    run_stage3_inference_from_config,
)
from gpcr_ringgen.pocket import QueryablePocketField
from gpcr_ringgen.sampling import JointDDPMStage3Sampler
from gpcr_ringgen.utils.io import load_yaml_config

REPOSITORY_ROOT = Path(__file__).parents[2]
SMOKE_CONFIG = REPOSITORY_ROOT / "configs" / "pipeline" / "stage3_smoke.yaml"
SMOKE_INPUT = REPOSITORY_ROOT / "examples" / "inputs" / "stage3_smoke.json"


def _config(output_path: Path) -> dict:
    config = copy.deepcopy(load_yaml_config(SMOKE_CONFIG))
    config["pipeline"]["input_path"] = str(SMOKE_INPUT)
    config["pipeline"]["output_path"] = str(output_path)
    return config


def test_minimal_pocket_scaffold_pipeline_builds_field_and_calls_sampler(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    original_sample = JointDDPMStage3Sampler.sample

    def tracked_sample(self, stage3_input, denoiser, **kwargs):
        calls.append(
            {
                "sample_ids": stage3_input.metadata.sample_ids,
                "has_field": kwargs.get("pocket_field") is not None,
                "seed": kwargs.get("seed"),
            }
        )
        return original_sample(self, stage3_input, denoiser, **kwargs)

    monkeypatch.setattr(JointDDPMStage3Sampler, "sample", tracked_sample)
    output_path = tmp_path / "stage3_output.json"

    run = run_stage3_inference(
        _config(output_path),
        config_reference=SMOKE_CONFIG,
    )

    assert calls == [
        {
            "sample_ids": ("stage3-smoke-001",),
            "has_field": True,
            "seed": 42,
        }
    ]
    assert isinstance(run.pocket_field, QueryablePocketField)
    assert run.pocket_field.channels == ("occupancy", "nearest_distance")
    assert run.sampling_output.coordinates.shape == (1, 3, 3)
    assert run.global_coordinates.shape == (1, 3, 3)
    assert run.sampling_output.coordinates.device.type == "cpu"
    assert torch.isfinite(run.sampling_output.coordinates).all()
    assert run.random_weights
    assert not run.checkpoint_loaded
    assert run.output_path == output_path
    assert output_path.is_file()

    valid = run.sampling_output.final_state.mask.atom_mask
    expected_global = (
        run.sampling_output.coordinates
        + run.processed_pocket.transform.center[:, None, :]
    )
    assert torch.allclose(run.global_coordinates[valid], expected_global[valid])
    fixed_ring = ~run.sampling_output.final_state.mask.diffusion_mask & valid
    expected_ring = torch.tensor(
        [[10.0, -5.0, 2.0], [11.0, -5.0, 2.0]]
    )
    assert torch.equal(run.global_coordinates[fixed_ring], expected_ring)

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["pipeline"] == "stage3_independent_inference"
    assert payload["model_weights"] == "random_untrained"
    assert payload["scientific_status"] == "pipeline_smoke_test_only"
    assert payload["diffusion_mode"] == "joint_chemical"
    assert payload["trajectory_saved"] is False
    assert payload["config_reference"] == str(SMOKE_CONFIG)
    assert payload["samples"][0]["sample_id"] == "stage3-smoke-001"
    assert payload["samples"][0]["diffusion_mask"] == [False, False, True]
    assert len(payload["samples"][0]["local_coordinates"]) == 3
    assert len(payload["samples"][0]["global_coordinates"]) == 3
    assert payload["samples"][0]["topology"]["edge_index"] == [
        [0, 1, 2],
        [1, 2, 0],
    ]
    assert len(payload["samples"][0]["chemical_labels"]["atom_type_ids"]) == 3
    assert len(payload["samples"][0]["chemical_labels"]["atom_types"]) == 3
    assert len(payload["samples"][0]["chemical_labels"]["bond_type_ids"]) == 3
    assert len(payload["samples"][0]["chemical_labels"]["bond_types"]) == 3
    assert payload["chemical_vocabulary"]["atom_types"][0] == "C"
    assert payload["chemical_vocabulary"]["bond_types"][-1] == "AROMATIC"
    assert (
        payload["samples"][0]["metadata"]["scaffold_source"]
        == "fixed-topology-mock"
    )
    assert payload["config_snapshot"]["diffusion"]["num_timesteps"] == 4


def test_pipeline_fixed_seed_is_reproducible(tmp_path: Path) -> None:
    first = run_stage3_inference(_config(tmp_path / "first.json"))
    second = run_stage3_inference(_config(tmp_path / "second.json"))

    assert torch.equal(
        first.sampling_output.coordinates,
        second.sampling_output.coordinates,
    )
    assert torch.equal(first.global_coordinates, second.global_coordinates)
    assert torch.equal(
        first.sampling_output.final_state.atom_types,
        second.sampling_output.final_state.atom_types,
    )
    assert torch.equal(
        first.sampling_output.final_state.bond_types,
        second.sampling_output.final_state.bond_types,
    )


def test_standard_input_rejects_unknown_atom_category(tmp_path: Path) -> None:
    input_payload = json.loads(SMOKE_INPUT.read_text(encoding="utf-8"))
    input_payload["scaffold"]["atom_types"][0] = "Xe"
    input_path = tmp_path / "unknown_atom.json"
    input_path.write_text(json.dumps(input_payload), encoding="utf-8")
    config = _config(tmp_path / "unused.json")
    config["pipeline"]["input_path"] = str(input_path)

    with pytest.raises(ConfigurationError, match="unsupported categories: Xe"):
        run_stage3_inference(config)


def test_yaml_entrypoint_runs_without_stage1_or_stage2_models(tmp_path: Path) -> None:
    config = _config(tmp_path / "from_yaml.json")
    config_path = tmp_path / "pipeline.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    run = run_stage3_inference_from_config(config_path)
    payload = json.loads(run.output_path.read_text(encoding="utf-8"))

    assert run.output_path == tmp_path / "from_yaml.json"
    assert payload["config_reference"] == str(config_path)
    assert (
        payload["samples"][0]["metadata"]["scaffold_source"]
        == "fixed-topology-mock"
    )
    assert "stage1" not in payload["pipeline"]
    assert "stage2" not in payload["pipeline"]


def test_random_weights_require_explicit_smoke_opt_in(tmp_path: Path) -> None:
    config = _config(tmp_path / "rejected.json")
    config["pipeline"]["allow_random_weights"] = False

    with pytest.raises(ConfigurationError, match="allow_random_weights=true"):
        Stage3InferencePipelineConfig.from_mapping(config)


def test_joint_pipeline_rejects_vocabulary_size_mismatch(tmp_path: Path) -> None:
    config = _config(tmp_path / "rejected_vocabulary.json")
    config["chemical_vocabulary"]["atom_types"] = ["C", "N"]

    with pytest.raises(ConfigurationError, match="num_atom_types must match"):
        run_stage3_inference(config)
