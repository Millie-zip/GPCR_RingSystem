"""Tests for exporting the reported 5-HT2C SQLite schema to Stage 3."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from gpcr_ringgen.preprocessing.diffusion_dataset.export_stage3 import (
    export_stage3_database,
)


def _source_database(path: Path) -> None:
    atoms = [
        {"atom_index": 0, "symbol": "C"},
        {"atom_index": 1, "symbol": "N"},
        {"atom_index": 2, "symbol": "O"},
    ]
    bonds = [
        {
            "bond_index": 0,
            "begin_atom_index": 0,
            "end_atom_index": 1,
            "bond_type": "SINGLE",
        },
        {
            "bond_index": 1,
            "begin_atom_index": 1,
            "end_atom_index": 2,
            "bond_type": "DOUBLE",
        },
    ]
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE selected_molecules (
                source_part TEXT,
                molecule_uid TEXT,
                canonical_isomeric_smiles TEXT,
                atom_facts_json TEXT,
                bond_facts_json TEXT
            );
            CREATE TABLE selected_ring_systems (
                source_part TEXT,
                molecule_uid TEXT,
                ring_system_uid TEXT,
                ring_atom_indices_json TEXT
            );
            """
        )
        connection.execute(
            "INSERT INTO selected_molecules VALUES (?, ?, ?, ?, ?)",
            ("part_0000.db", "MOL_1", "C1N1O", json.dumps(atoms), json.dumps(bonds)),
        )
        connection.execute(
            "INSERT INTO selected_ring_systems VALUES (?, ?, ?, ?)",
            ("part_0000.db", "MOL_1", "RS_1", "[0,1]"),
        )


def _poses_database(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE docked_poses (
                source_part TEXT,
                molecule_uid TEXT,
                pose_id TEXT,
                coordinates_json TEXT,
                status TEXT,
                cavity_frac REAL,
                anchor_frac REAL,
                span_excess_A REAL,
                collision_frac REAL
            )
            """
        )
        connection.execute(
            "INSERT INTO docked_poses VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "part_0000.db",
                "MOL_1",
                "pose_0",
                "[[1,2,3],[2,2,3],[3,2,3]]",
                "PASS",
                0.9,
                0.3,
                0.2,
                0.01,
            ),
        )


def _pocket_json(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "pocket_id": "8DPF-test",
                "pocket": {
                    "coordinates": [[0.0, 0.0, 0.0]],
                    "features": [[0.0, 0.0]],
                    "mask": [True],
                    "atom_types": ["CA"],
                    "elements": ["C"],
                    "residue_names": ["ALA"],
                    "residue_ids": ["1"],
                    "chain_ids": ["A"],
                },
            }
        ),
        encoding="utf-8",
    )


def test_export_builds_ring_fixed_sample_without_modifying_source(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.db"
    poses = tmp_path / "poses.db"
    pocket = tmp_path / "pocket.json"
    output = tmp_path / "stage3.db"
    _source_database(source)
    _poses_database(poses)
    _pocket_json(pocket)
    source_size = source.stat().st_size

    counts = export_stage3_database(source, poses, pocket, output)

    assert counts == (1, 0)
    assert source.stat().st_size == source_size
    with sqlite3.connect(output) as connection:
        scaffold = json.loads(
            connection.execute(
                "SELECT scaffold_json FROM stage3_samples"
            ).fetchone()[0]
        )
        metadata = dict(connection.execute("SELECT key, value FROM export_metadata"))
    assert scaffold["atom_types"] == [0, 1, 2]
    assert scaffold["bond_types"] == [0, 1]
    assert scaffold["diffusion_mask"] == [False, False, True]
    assert scaffold["coordinates"][0] == [1.0, 2.0, 3.0]
    assert metadata["source_access"] == "READ_ONLY"
    assert metadata["successful_samples"] == "1"


def test_export_refuses_to_overwrite_existing_output(tmp_path: Path) -> None:
    source = tmp_path / "source.db"
    poses = tmp_path / "poses.db"
    pocket = tmp_path / "pocket.json"
    output = tmp_path / "stage3.db"
    _source_database(source)
    _poses_database(poses)
    _pocket_json(pocket)
    output.write_bytes(b"keep")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        export_stage3_database(source, poses, pocket, output)

    assert output.read_bytes() == b"keep"
