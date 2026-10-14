"""Scientific metadata checks through real checkpoint consumption paths."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sqlite3

import pytest
import torch
import yaml

from gpcr_ringgen.checkpoint_semantics import file_sha256, label_cache_provenance
from gpcr_ringgen.core.exceptions import ConfigurationError
from gpcr_ringgen.pipeline import run_stage3_inference
from gpcr_ringgen.training import run_record_inference, run_training, run_ra_reward_training
from gpcr_ringgen.training.ra_reward import load_ra_reward_checkpoint
from gpcr_ringgen.training.ra_reward_data import build_rascore_label_cache
from gpcr_ringgen.utils.io import load_yaml_config


ROOT = Path(__file__).parents[2]


@pytest.fixture
def trained_generator(tmp_path: Path) -> tuple[dict, Path]:
    config = load_yaml_config(ROOT / "configs/training/stage3_joint_real.yaml")
    path = tmp_path / "records.jsonl"
    records = [{
        "schema_version": "stage3_training_record_v1",
        "sample_id": f"group-{index}",
        "graph": {"atom_features": [[0.0] * 3] * 2, "coordinates": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                  "atom_types": [0, 1], "edge_index": [[0], [1]], "edge_features": [[0.0]], "bond_types": [0]},
        "descriptor": [],
        "metadata": {"split_group": f"group-{index}", "canonical_isomeric_smiles": "CC"},
    } for index in range(12)]
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    config["data"].update(path=str(path), feature_manifest=str(ROOT / config["data"]["feature_manifest"]))
    config["training"].update(epochs=1, batch_size=4, train_fraction=0.5, output_dir=str(tmp_path / "generator"))
    config["model"].update(hidden_dim=8, num_layers=1, timestep_embedding_dim=4)
    config["device"].update(type="cpu", dtype="float32")
    config["diffusion"].update(num_timesteps=2, beta_start=0.01, beta_end=0.05)
    trained = run_training(config)
    return config, trained.last_checkpoint


def _entry_config(config: dict, checkpoint: Path, entry: str, tmp_path: Path) -> dict:
    result = copy.deepcopy(config)
    if entry == "record":
        result["inference"] = {"checkpoint": str(checkpoint), "output": str(tmp_path / "record_output.json")}
    elif entry == "resume":
        result["training"].update(resume_from=str(checkpoint), epochs=2, output_dir=str(tmp_path / "resume"))
    else:
        smoke = load_yaml_config(ROOT / "configs/pipeline/stage3_smoke.yaml")
        result["pipeline"] = smoke["pipeline"]
        result["pipeline"].update(checkpoint_path=str(checkpoint), allow_random_weights=False, output_path=str(tmp_path / "pipeline_output.json"))
        result["field"] = smoke["field"]
        result["sampling"] = smoke["sampling"]
        payload = json.loads((ROOT / "examples/inputs/stage3_smoke.json").read_text(encoding="utf-8"))
        payload["scaffold"]["atom_features"] = [[0.0] * 3 for _ in range(3)]
        payload["scaffold"]["edge_features"] = [[0.0] for _ in range(3)]
        payload["descriptor"] = {"values": [], "mask": []}
        path = tmp_path / "input.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        result["pipeline"]["input_path"] = str(path)
    return result


def _run(entry: str, config: dict):
    if entry == "record":
        return run_record_inference(config)
    if entry == "resume":
        return run_training(config)
    return run_stage3_inference(config)


@pytest.mark.parametrize("entry", ["record", "pipeline", "resume"])
@pytest.mark.parametrize("mutation", ["atom_order", "bond_order", "schedule", "timesteps", "scale", "features"])
def test_same_shape_semantic_changes_are_rejected(trained_generator, tmp_path: Path, entry: str, mutation: str) -> None:
    original, checkpoint = trained_generator
    config = _entry_config(original, checkpoint, entry, tmp_path)
    if mutation in ("atom_order", "bond_order"):
        key = "atom_types" if mutation == "atom_order" else "bond_types"
        config["chemical_vocabulary"][key][:2] = reversed(config["chemical_vocabulary"][key][:2])
    elif mutation == "schedule":
        config["diffusion"]["beta_end"] = 0.2
    elif mutation == "timesteps":
        config["diffusion"]["num_timesteps"] = 3
    elif mutation == "scale":
        config["model"]["coordinate_scale"] = 0.2
    else:
        manifest = load_yaml_config(config["data"]["feature_manifest"])
        manifest["atom_features"][:2] = reversed(manifest["atom_features"][:2])
        path = tmp_path / "changed_manifest.yaml"
        path.write_text(yaml.safe_dump(manifest), encoding="utf-8")
        config["data"]["feature_manifest"] = str(path)
    with pytest.raises(ConfigurationError, match="semantics mismatch|feature manifest"):
        _run(entry, config)
    assert not (tmp_path / "record_output.json").exists()
    assert not (tmp_path / "pipeline_output.json").exists()
    assert not (tmp_path / "resume/last.ckpt").exists()


@pytest.mark.parametrize("entry", ["record", "pipeline", "resume"])
def test_runtime_path_changes_are_allowed(trained_generator, tmp_path: Path, entry: str) -> None:
    original, checkpoint = trained_generator
    config = _entry_config(original, checkpoint, entry, tmp_path)
    relocated = tmp_path / "relocated.jsonl"
    relocated.write_bytes(Path(config["data"]["path"]).read_bytes())
    config["data"]["path"] = str(relocated)
    if entry != "resume":
        config["seed"] += 1
    _run(entry, config)


def test_resume_rejects_changed_data_version(trained_generator, tmp_path: Path) -> None:
    original, checkpoint = trained_generator
    config = _entry_config(original, checkpoint, "resume", tmp_path)
    with Path(config["data"]["path"]).open("a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(ConfigurationError, match="training_provenance.records_sha256"):
        run_training(config)


@pytest.mark.parametrize("entry", ["record", "pipeline"])
def test_legacy_generator_with_saved_config_and_manifest_hash_is_usable(trained_generator, tmp_path: Path, entry: str) -> None:
    original, checkpoint = trained_generator
    payload = torch.load(checkpoint, weights_only=True)
    payload.pop("semantics")
    payload.pop("training_provenance")
    payload["checkpoint_version"] = 1
    payload["config"]["data"]["feature_manifest"] = "missing/historical/path.yaml"
    legacy = tmp_path / "legacy.ckpt"
    torch.save(payload, legacy)
    _run(entry, _entry_config(original, legacy, entry, tmp_path))


def test_bare_weights_and_legacy_resume_without_data_version_are_rejected(trained_generator, tmp_path: Path) -> None:
    original, checkpoint = trained_generator
    payload = torch.load(checkpoint, weights_only=True)
    bare = tmp_path / "bare.ckpt"
    torch.save({"model_state_dict": payload["model_state_dict"]}, bare)
    with pytest.raises(ConfigurationError, match="semantic metadata"):
        run_stage3_inference(_entry_config(original, bare, "pipeline", tmp_path))
    payload.pop("semantics")
    payload.pop("training_provenance")
    payload["checkpoint_version"] = 1
    legacy = tmp_path / "legacy.ckpt"
    torch.save(payload, legacy)
    with pytest.raises(ConfigurationError, match="training_provenance"):
        run_training(_entry_config(original, legacy, "resume", tmp_path))


@pytest.fixture
def trained_reward(trained_generator, tmp_path: Path) -> tuple[dict, Path]:
    config, generator_checkpoint = trained_generator
    labels = tmp_path / "labels.db"
    build_rascore_label_cache(config["data"]["path"], labels, lambda smiles: 0.5, teacher_version="test_teacher_v1")
    reward_config = load_yaml_config(ROOT / "configs/training/ra_reward.yaml")
    reward_config["data"] = {**config["data"], "labels_path": str(labels)}
    reward_config["chemical_vocabulary"] = copy.deepcopy(config["chemical_vocabulary"])
    reward_config["training"].update(epochs=1, batch_size=4, train_fraction=0.5, output_dir=str(tmp_path / "reward"))
    reward_config["device"].update(type="cpu", dtype="float32")
    reward_config["ra_reward_model"].update(hidden_dim=8, num_layers=1, dropout=0.0)
    result = run_ra_reward_training(reward_config)
    payload = torch.load(result.last_checkpoint, weights_only=True)
    provenance = payload["training_provenance"]
    assert provenance["records_sha256"] == file_sha256(config["data"]["path"])
    assert provenance["labels_sha256"] == label_cache_provenance(labels)["labels_sha256"]
    inference = _entry_config(config, generator_checkpoint, "record", tmp_path)
    inference["ra_reward"] = {
        "checkpoint": str(result.last_checkpoint),
        "expected_provenance": {key: provenance[key] for key in ("teacher_version", "labels_sha256", "records_sha256")},
    }
    return inference, result.last_checkpoint


def test_matching_reward_checkpoint_guides_inference(trained_reward) -> None:
    config, checkpoint = trained_reward
    run_record_inference(config)
    loaded = load_ra_reward_checkpoint(checkpoint, device=torch.device("cpu"), dtype=torch.float32)
    assert loaded.training_provenance["teacher_version"] == "test_teacher_v1"
    assert all(not parameter.requires_grad for parameter in loaded.parameters())


@pytest.mark.parametrize("mutation", ["atom_order", "features", "teacher_version", "labels_sha256", "records_sha256", "missing_metadata", "missing_pins"])
def test_reward_semantic_and_source_mismatches_are_rejected(trained_reward, tmp_path: Path, mutation: str) -> None:
    config, checkpoint = trained_reward
    if mutation == "missing_pins":
        config["ra_reward"].pop("expected_provenance")
    elif mutation in ("teacher_version", "labels_sha256", "records_sha256"):
        config["ra_reward"]["expected_provenance"][mutation] = "other_teacher" if mutation == "teacher_version" else "0" * 64
    else:
        payload = torch.load(checkpoint, weights_only=True)
        if mutation == "atom_order":
            payload["semantics"]["chemical_vocabulary"]["atom_types"][:2] = ["N", "C"]
        elif mutation == "features":
            payload["semantics"]["features"]["atom_features"][:2] = reversed(payload["semantics"]["features"]["atom_features"][:2])
        else:
            payload.pop("semantics")
            payload.pop("training_provenance")
            payload["checkpoint_version"] = 1
        changed = tmp_path / "changed_reward.ckpt"
        torch.save(payload, changed)
        config["ra_reward"]["checkpoint"] = str(changed)
    with pytest.raises(ConfigurationError, match="semantics mismatch|semantic metadata|expected_provenance"):
        run_record_inference(config)
    assert not (tmp_path / "record_output.json").exists()


def test_mixed_teacher_cache_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "mixed.db"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE rascore_labels(canonical_smiles TEXT, ra_score REAL, teacher_version TEXT)")
        connection.executemany("INSERT INTO rascore_labels VALUES (?, ?, ?)", [("C", 0.2, "teacher_a"), ("CC", 0.8, "teacher_b")])
    with pytest.raises(ConfigurationError, match="exactly one teacher version"):
        label_cache_provenance(path)


def test_label_fingerprint_observes_WAL_updates(tmp_path: Path) -> None:
    path = tmp_path / "wal.db"
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("CREATE TABLE rascore_labels(canonical_smiles TEXT, ra_score REAL, teacher_version TEXT)")
        connection.execute("INSERT INTO rascore_labels VALUES ('C', 0.2, 'test_teacher_v1')")
        connection.commit()
        before = label_cache_provenance(path)
        connection.execute("UPDATE rascore_labels SET ra_score=0.8")
        connection.commit()
        after = label_cache_provenance(path)
    assert before["teacher_version"] == after["teacher_version"]
    assert before["label_count"] == after["label_count"] == 1
    assert before["labels_sha256"] != after["labels_sha256"]
