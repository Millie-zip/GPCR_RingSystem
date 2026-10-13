"""Regression tests for the raw model-input SQLite adapter."""

import json
import sqlite3
import struct
from pathlib import Path

import pytest

from gpcr_ringgen.data import JsonlMolecularDataset, collate_molecules, materialize_training_jsonl
from gpcr_ringgen.data.three_d_assets import ThreeDAssetBuilderConfig, build_3d_assets
from gpcr_ringgen.models.diffusion import clean_stage3_state


def _write_model_input_db(path: Path) -> str:
    sample_id = "coconut:part_0000.db:MOL_TEST:RS_TEST_001"
    atoms = [
        {
            "atom_index": 0,
            "atomic_number": 6,
            "symbol": "C",
            "formal_charge": 0,
            "is_aromatic": False,
            "degree": 1,
            "total_h": 3,
            "chiral_tag": "CHI_UNSPECIFIED",
        },
        {
            "atom_index": 1,
            "atomic_number": 8,
            "symbol": "O",
            "formal_charge": 0,
            "is_aromatic": False,
            "degree": 1,
            "total_h": 1,
            "chiral_tag": "CHI_UNSPECIFIED",
        },
    ]
    bonds = [
        {
            "bond_index": 0,
            "begin_atom_index": 0,
            "end_atom_index": 1,
            "bond_type": "SINGLE",
            "bond_order": 1.0,
            "is_aromatic": False,
            "is_conjugated": False,
            "stereo": "STEREONONE",
        }
    ]
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE molecules (
                test_id INTEGER PRIMARY KEY,
                source_part TEXT NOT NULL,
                molecule_uid TEXT NOT NULL,
                canonical_isomeric_smiles TEXT NOT NULL,
                num_atoms INTEGER NOT NULL,
                num_bonds INTEGER NOT NULL,
                x0 BLOB NOT NULL,
                ring_atom_mask BLOB NOT NULL,
                noise_mask BLOB NOT NULL,
                atom_facts_json TEXT NOT NULL,
                bond_facts_json TEXT NOT NULL,
                sample_id TEXT,
                ring_system_uid TEXT
            )
            """
        )
        connection.execute(
            """
            INSERT INTO molecules VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                1,
                "part_0000.db",
                "MOL_TEST",
                "CO",
                2,
                1,
                struct.pack("<6f", 0.0, 0.0, 0.0, 1.4, 0.0, 0.0),
                bytes((1, 0)),
                bytes((1, 1)),
                json.dumps(atoms),
                json.dumps(bonds),
                sample_id,
                "RS_TEST_001",
            ),
        )
    return sample_id


def _write_normalized_model_input_db(path: Path) -> str:
    """Create the normalized molecules/ring_systems shape used by production."""
    sample_id = "model3d:part_0000.db:MOL_NORMALIZED"
    atoms = [
        {
            "atom_index": 0,
            "atomic_number": 6,
            "symbol": "C",
            "formal_charge": 0,
            "is_aromatic": False,
            "degree": 1,
            "total_h": 3,
            "chiral_tag": "CHI_UNSPECIFIED",
        },
        {
            "atom_index": 1,
            "atomic_number": 8,
            "symbol": "O",
            "formal_charge": 0,
            "is_aromatic": False,
            "degree": 1,
            "total_h": 1,
            "chiral_tag": "CHI_UNSPECIFIED",
        },
    ]
    bonds = [
        {
            "bond_index": 0,
            "begin_atom_index": 0,
            "end_atom_index": 1,
            "bond_type": "SINGLE",
            "bond_order": 1.0,
            "is_aromatic": False,
            "is_conjugated": False,
            "stereo": "STEREONONE",
        }
    ]
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE molecules (
                sample_index INTEGER PRIMARY KEY,
                sample_id TEXT NOT NULL,
                source_part TEXT NOT NULL,
                molecule_uid TEXT NOT NULL,
                canonical_isomeric_smiles TEXT NOT NULL,
                atom_count INTEGER NOT NULL,
                bond_count INTEGER NOT NULL,
                x0 BLOB NOT NULL,
                ring_atom_mask BLOB NOT NULL,
                noise_mask BLOB NOT NULL,
                atom_facts_json TEXT NOT NULL,
                bond_facts_json TEXT NOT NULL
            );
            CREATE TABLE ring_systems (
                source_part TEXT NOT NULL,
                ring_system_uid TEXT PRIMARY KEY,
                molecule_uid TEXT NOT NULL,
                system_index INTEGER NOT NULL,
                ring_atom_indices_json TEXT NOT NULL,
                ring_bond_indices_json TEXT NOT NULL
            );
            """
        )
        connection.execute(
            """
            INSERT INTO molecules VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                7,
                sample_id,
                "part_0000.db",
                "MOL_NORMALIZED",
                "CO",
                2,
                1,
                struct.pack("<6f", 0.0, 0.0, 0.0, 1.4, 0.0, 0.0),
                bytes((1, 0)),
                bytes((1, 1)),
                json.dumps(atoms),
                json.dumps(bonds),
            ),
        )
        connection.execute(
            "INSERT INTO ring_systems VALUES (?, ?, ?, ?, ?, ?)",
            ("part_0000.db", "RS_NORMALIZED_001", "MOL_NORMALIZED", 0, "[0]", "[]"),
        )
    return sample_id


def test_raw_model_input_sqlite_preserves_source_coordinates(tmp_path: Path) -> None:
    source_db = tmp_path / "model_input.db"
    sample_id = _write_model_input_db(source_db)
    asset_db = tmp_path / "assets.db"
    records_path = tmp_path / "records.jsonl"
    manifest = Path(__file__).parents[2] / "configs" / "data" / "stage3_feature_manifest.yaml"

    result = build_3d_assets(
        ThreeDAssetBuilderConfig(source_db=source_db, output_db=asset_db)
    )
    assert result["molecule_pass_count"] == 1
    assert result["ring_system_count"] == 1

    materialized = materialize_training_jsonl(
        asset_db,
        records_path,
        manifest,
        overwrite=True,
    )
    assert materialized["written"] == 1
    record = JsonlMolecularDataset(records_path)[0]
    assert record.sample_id == sample_id
    assert record.metadata["source_sample_id"] == sample_id
    assert record.coordinates[0] == [0.0, 0.0, 0.0]
    assert record.coordinates[1] == pytest.approx([1.4, 0.0, 0.0])
    assert record.atom_types == [0, 2]
    assert record.bond_types == [0]

    stage3_input = collate_molecules([record])
    assert stage3_input.scaffold.atom_types is not None
    assert stage3_input.scaffold.bond_types is not None
    clean_state = clean_stage3_state(stage3_input)
    assert clean_state.atom_types is not None
    assert clean_state.bond_types is not None
    assert clean_state.atom_types.tolist() == [[0, 2]]
    assert clean_state.bond_types.tolist() == [[0]]


def test_normalized_model_input_sqlite_joins_ring_systems(tmp_path: Path) -> None:
    source_db = tmp_path / "normalized_model_input.db"
    sample_id = _write_normalized_model_input_db(source_db)
    asset_db = tmp_path / "assets.db"
    records_path = tmp_path / "records.jsonl"
    manifest = Path(__file__).parents[2] / "configs" / "data" / "stage3_joint_feature_manifest.yaml"

    result = build_3d_assets(
        ThreeDAssetBuilderConfig(source_db=source_db, output_db=asset_db)
    )
    assert result["molecule_count"] == 1
    assert result["molecule_pass_count"] == 1
    assert result["ring_system_count"] == 1

    materialized = materialize_training_jsonl(
        asset_db,
        records_path,
        manifest,
        overwrite=True,
    )
    assert materialized["written"] == 1
    record = JsonlMolecularDataset(records_path)[0]
    assert record.sample_id == f"{sample_id}:RS_NORMALIZED_001"
    assert record.metadata["ring_system_uid"] == "RS_NORMALIZED_001"
    assert record.coordinates[1] == pytest.approx([1.4, 0.0, 0.0])
