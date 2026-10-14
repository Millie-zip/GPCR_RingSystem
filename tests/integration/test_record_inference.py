"""Checkpoint-to-JSON regression tests for pocket-disabled record inference."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from gpcr_ringgen.data.vocabulary import ATOM_TYPES, BOND_TYPES
from gpcr_ringgen.core.exceptions import ConfigurationError
from gpcr_ringgen.training.inference import run_record_inference
from gpcr_ringgen.training.trainer import run_training
from gpcr_ringgen.utils.io import load_yaml_config


REPOSITORY_ROOT = Path(__file__).parents[2]


@pytest.mark.parametrize(
    "joint, atom_ids, legacy",
    [
        (False, [0, 1], False),
        (True, [0, 1], False),
        (False, [9, 10, 11], False),
        (True, [9, 10, 11], False),
        (False, [0, 1], True),
    ],
    ids=["coordinates", "joint_chemical", "coordinates_H_B_Si", "joint_H_B_Si", "legacy_9"],
)
def test_record_inference_saves_chemical_labels(
    tmp_path: Path, joint: bool, atom_ids: list[int], legacy: bool
) -> None:
    name = "stage3_joint_real.yaml" if joint else "stage3_real.yaml"
    config = load_yaml_config(REPOSITORY_ROOT / "configs" / "training" / name)
    if joint:
        # Use a non-default order to verify that joint output uses its config.
        config["chemical_vocabulary"]["atom_types"][:2] = ["N", "C"]
        expected_atoms = config["chemical_vocabulary"]["atom_types"]
        expected_bonds = config["chemical_vocabulary"]["bond_types"]
    else:
        assert "chemical_vocabulary" not in config
        expected_atoms = list(ATOM_TYPES)
        expected_bonds = list(BOND_TYPES)
    if legacy:
        config["model"]["atom_type_vocab_size"] = 9
    num_atoms = len(atom_ids)
    num_edges = num_atoms - 1
    edge_index = [list(range(num_edges)), list(range(1, num_atoms))]

    records = [
        {
            "schema_version": "stage3_training_record_v1",
            "sample_id": f"group-{index}",
            "graph": {
                "atom_features": [[0.0] * config["model"]["atom_feature_dim"] for _ in atom_ids],
                "coordinates": [[float(atom), 0.0, 0.0] for atom in range(num_atoms)],
                "atom_types": atom_ids,
                "edge_index": edge_index,
                "edge_features": [[0.0] * config["model"]["edge_feature_dim"] for _ in range(num_edges)],
                "bond_types": [1] * num_edges,
            },
            "descriptor": [],
            "metadata": {"split_group": f"group-{index}"},
        }
        for index in range(12)
    ]
    data_path = tmp_path / "records.jsonl"
    data_path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    config["data"]["path"] = str(data_path)
    config["data"]["feature_manifest"] = str(REPOSITORY_ROOT / config["data"]["feature_manifest"])
    config["training"].update(
        epochs=1, batch_size=4, train_fraction=0.5, output_dir=str(tmp_path / "training")
    )
    config["device"].update(type="cpu", dtype="float32")
    config["model"].update(hidden_dim=8, num_layers=1, timestep_embedding_dim=4)
    config["diffusion"].update(num_timesteps=2, beta_start=0.01, beta_end=0.05)
    training = run_training(config)
    checkpoint = torch.load(training.last_checkpoint, weights_only=True)
    embedding = checkpoint["model_state_dict"]["backbone.atom_type_embedding.weight"]
    assert embedding.shape[0] == (9 if legacy else 12)
    config["inference"] = {
        "checkpoint": str(training.last_checkpoint),
        "record_index": 0,
        "output": str(tmp_path / "inference.json"),
    }

    output_path = run_record_inference(config)

    assert output_path == tmp_path / "inference.json"
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["sample_id"] == "group-0"
    assert payload["chemical_vocabulary"] == {
        "atom_types": expected_atoms,
        "bond_types": expected_bonds,
    }
    assert payload["atom_mask"] == [True] * num_atoms
    assert payload["edge_mask"] == [True] * num_edges
    assert payload["edge_index"] == edge_index
    assert torch.isfinite(torch.tensor(payload["coordinates"])).all()
    if joint:
        assert len(payload["atom_type_ids"]) == num_atoms
        assert len(payload["bond_type_ids"]) == num_edges
        assert all(0 <= category < len(expected_atoms) for category in payload["atom_type_ids"])
        assert all(0 <= category < len(expected_bonds) for category in payload["bond_type_ids"])
    else:
        assert payload["atom_type_ids"] == records[0]["graph"]["atom_types"]
        assert payload["bond_type_ids"] == records[0]["graph"]["bond_types"]
    if legacy:
        config["model"].pop("atom_type_vocab_size")
        with pytest.raises(ConfigurationError, match="semantics mismatch"):
            run_record_inference(config)
