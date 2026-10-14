"""End-to-end tests for independent Stage 3 inference orchestration."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch
import yaml

from gpcr_ringgen.core.exceptions import ConfigurationError
from gpcr_ringgen.models.equivariant import (
    EquivariantDenoiserBackbone,
    EquivariantDenoiserConfig,
    FieldConditionedEquivariantDenoiser,
    FieldConditioningConfig,
)
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


@pytest.mark.parametrize("joint", [False, True], ids=["coordinates", "joint"])
@pytest.mark.parametrize("numeric_labels", [False, True], ids=["names", "ids"])
def test_pipeline_accepts_H_B_Si_and_respects_configured_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, joint: bool, numeric_labels: bool
) -> None:
    config = _config(tmp_path / "extended_output.json")
    atom_names = ["H", "B", "Si"]
    if joint:
        # The input must follow this order rather than the canonical data order.
        vocabulary = ["Si", "H", "B", "C", "N", "O", "S", "P", "F", "Cl", "Br", "I"]
        bonds = ["AROMATIC", "TRIPLE", "DOUBLE", "SINGLE"]
        config["chemical_vocabulary"] = {"atom_types": vocabulary, "bond_types": bonds}
        config["model"]["num_atom_types"] = 12
        config["diffusion"]["num_atom_types"] = 12
        expected_atom_ids = [1, 2, 0]
        expected_bond_ids = [3, 2, 0]
    else:
        config.pop("chemical_vocabulary")
        config["diffusion"]["mode"] = "coordinate_only"
        for key in ("num_atom_types", "num_bond_types", "atom_type_vocab_size", "bond_type_vocab_size"):
            config["model"].pop(key, None)
        for key in ("num_atom_types", "num_bond_types"):
            config["diffusion"].pop(key, None)
        expected_atom_ids = [9, 10, 11]
        expected_bond_ids = [0, 1, 3]
    payload = json.loads(SMOKE_INPUT.read_text(encoding="utf-8"))
    payload["scaffold"]["atom_types"] = expected_atom_ids if numeric_labels else atom_names
    payload["scaffold"]["bond_types"] = expected_bond_ids if numeric_labels else ["SINGLE", "DOUBLE", "AROMATIC"]
    input_path = tmp_path / "extended_input.json"
    input_path.write_text(json.dumps(payload), encoding="utf-8")
    config["pipeline"]["input_path"] = str(input_path)

    if joint:
        original_sample = JointDDPMStage3Sampler.sample

        def check_labels_then_sample(self, stage3_input, denoiser, **kwargs):
            assert stage3_input.scaffold.atom_types[0].tolist() == expected_atom_ids
            assert stage3_input.scaffold.bond_types[0].tolist() == expected_bond_ids
            return original_sample(self, stage3_input, denoiser, **kwargs)

        monkeypatch.setattr(JointDDPMStage3Sampler, "sample", check_labels_then_sample)

    run = run_stage3_inference(config)

    state = run.sampling_output.final_state
    if not joint:
        assert state.atom_types[0].tolist() == expected_atom_ids
        assert state.bond_types[0].tolist() == expected_bond_ids
    else:
        assert run.chemical_vocabulary.atom_types == tuple(vocabulary)
        assert all(0 <= label < 12 for label in state.atom_types[0].tolist())
    assert torch.isfinite(run.global_coordinates).all()
    assert run.output_path.is_file()


def test_legacy_coordinate_pipeline_rejects_extended_element(tmp_path: Path) -> None:
    config = _config(tmp_path / "legacy_output.json")
    config.pop("chemical_vocabulary")
    config["diffusion"]["mode"] = "coordinate_only"
    for key in ("num_atom_types", "num_bond_types"):
        config["model"].pop(key)
        config["diffusion"].pop(key)
    # The smoke model explicitly retains its 9-entry embedding configuration.
    payload = json.loads(SMOKE_INPUT.read_text(encoding="utf-8"))
    payload["scaffold"]["atom_types"][0] = "H"
    input_path = tmp_path / "legacy_input.json"
    input_path.write_text(json.dumps(payload), encoding="utf-8")
    config["pipeline"]["input_path"] = str(input_path)

    with pytest.raises(ConfigurationError, match="unsupported categories: H"):
        run_stage3_inference(config)


@pytest.mark.parametrize("category", ["H", 9], ids=["name", "id"])
def test_joint_pipeline_rejects_labels_outside_its_nine_class_vocabulary(
    tmp_path: Path, category: str | int
) -> None:
    config = _config(tmp_path / "rejected.json")
    payload = json.loads(SMOKE_INPUT.read_text(encoding="utf-8"))
    payload["scaffold"]["atom_types"] = [category, "N", "O"] if isinstance(category, str) else [category, 1, 2]
    input_path = tmp_path / "out_of_vocabulary.json"
    input_path.write_text(json.dumps(payload), encoding="utf-8")
    config["pipeline"]["input_path"] = str(input_path)

    with pytest.raises(ConfigurationError, match="unsupported categories: H|IDs must be in"):
        run_stage3_inference(config)


def test_legacy_coordinate_pipeline_loads_nine_class_checkpoint(tmp_path: Path) -> None:
    config = _config(tmp_path / "legacy_checkpoint_output.json")
    config.pop("chemical_vocabulary")
    config["diffusion"]["mode"] = "coordinate_only"
    for key in ("num_atom_types", "num_bond_types"):
        config["model"].pop(key)
        config["diffusion"].pop(key)
    model = FieldConditionedEquivariantDenoiser(
        EquivariantDenoiserBackbone(EquivariantDenoiserConfig.from_mapping(config)),
        FieldConditioningConfig.from_mapping(config),
    )
    assert model.backbone.atom_type_embedding.weight.shape[0] == 9
    # A legacy file is usable only when its saved configuration establishes
    # its scientific meaning. Bare tensor dictionaries are tested separately.
    config["feature_manifest"] = {
        "schema_version": "test_pipeline_features_v1",
        "atom_features": [f"atom_{index}" for index in range(4)],
        "edge_features": ["edge_0", "edge_1"],
        "descriptor_features": ["descriptor_0", "descriptor_1", "descriptor_2"],
    }
    checkpoint_path = tmp_path / "legacy.ckpt"
    saved_config = copy.deepcopy(config)
    saved_config["model"].pop("atom_type_vocab_size")
    torch.save({"checkpoint_version": 1, "model_state_dict": model.state_dict(), "config": saved_config}, checkpoint_path)
    config["pipeline"].update(checkpoint_path=str(checkpoint_path), allow_random_weights=False)

    run = run_stage3_inference(config)

    assert run.checkpoint_loaded
    assert not run.random_weights
    assert run.sampling_output.final_state.atom_types[0].tolist() == [0, 1, 2]
    assert run.output_path.is_file()
    config["field_conditioning"]["channels"].reverse()
    with pytest.raises(ConfigurationError, match="generator.field_conditioning.channels"):
        run_stage3_inference(config)
    config["field_conditioning"]["channels"].reverse()
    config["model"].pop("atom_type_vocab_size")
    with pytest.raises(ConfigurationError, match="semantics mismatch"):
        run_stage3_inference(config)
