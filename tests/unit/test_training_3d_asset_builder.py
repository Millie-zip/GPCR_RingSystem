"""Tests for the pocket-independent 3D training-asset builder."""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3

import pytest

from gpcr_ringgen.preprocessing.diffusion_dataset.build_training_assets import (
    SUPPORTED_ATOM_TYPES,
    SUPPORTED_BOND_TYPES,
    build_training_assets,
    unpack_coordinates,
)


def _source_database(path: Path, *, unsupported_atom: bool = False) -> None:
    atoms = [
        {
            "atom_index": 0,
            "atomic_number": 6,
            "symbol": "Xe" if unsupported_atom else "C",
            "formal_charge": 0,
            "is_aromatic": False,
        },
        {
            "atom_index": 1,
            "atomic_number": 6,
            "symbol": "C",
            "formal_charge": 0,
            "is_aromatic": False,
        },
        {
            "atom_index": 2,
            "atomic_number": 8,
            "symbol": "O",
            "formal_charge": 0,
            "is_aromatic": False,
        },
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
            "bond_type": "SINGLE",
        },
    ]
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE selected_molecules (
                source_part TEXT,
                molecule_uid TEXT,
                original_smiles TEXT,
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
            "INSERT INTO selected_molecules VALUES (?, ?, ?, ?, ?, ?)",
            (
                "part_0000.db",
                "MOL_1",
                "CCO",
                "CCO",
                json.dumps(atoms),
                json.dumps(bonds),
            ),
        )
        connection.executemany(
            "INSERT INTO selected_ring_systems VALUES (?, ?, ?, ?)",
            [
                ("part_0000.db", "MOL_1", "RS_1", "[0]"),
                ("part_0000.db", "MOL_1", "RS_2", "[0,1]"),
            ],
        )


def _fake_conformer(
    smiles: str,
    atoms: object,
    bonds: object,
    random_seed: int,
    timeout_seconds: float,
) -> dict[str, object]:
    assert smiles == "CCO"
    assert random_seed == 0xF00D
    assert timeout_seconds == 5.0
    return {
        "coordinates": [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [2.8, 0.2, 0.0]],
        "source_to_rdkit": [0, 1, 2],
        "rdkit_to_source": [0, 1, 2],
        "mapping_method": "identity",
        "mapping_match_count": 1,
        "embedding_method": "ETKDGv3",
        "etkdg_failure_counts": [],
        "geometry": {
            "min_bond_length_A": 1.3,
            "max_bond_length_A": 1.5,
            "min_nonbonded_distance_A": 2.8,
        },
    }


def test_offline_vocabulary_matches_frozen_model_contract() -> None:
    from gpcr_ringgen.core.vocab import ATOM_TYPE_TO_ID, BOND_TYPE_TO_ID

    assert SUPPORTED_ATOM_TYPES == tuple(ATOM_TYPE_TO_ID)
    assert SUPPORTED_BOND_TYPES == tuple(BOND_TYPE_TO_ID)


def test_builder_writes_aligned_coordinates_masks_and_reports(tmp_path: Path) -> None:
    source = tmp_path / "source.db"
    output = tmp_path / "training_3d.db"
    progress = tmp_path / "progress.json"
    summary = tmp_path / "summary.json"
    _source_database(source)
    source_bytes = source.read_bytes()

    result = build_training_assets(
        source,
        output,
        progress,
        summary,
        limit=100,
        conformer_timeout_seconds=5.0,
        conformer_runner=_fake_conformer,
    )

    assert source.read_bytes() == source_bytes
    assert result["passed_molecules"] == 1
    assert result["passed_ring_samples"] == 2
    assert json.loads(progress.read_text(encoding="utf-8"))["status"] == "complete"
    assert json.loads(summary.read_text(encoding="utf-8"))["source_access"] == "READ_ONLY"

    with sqlite3.connect(output) as connection:
        conformer = connection.execute(
            """SELECT atom_count, coordinate_dtype, x0_blob,
                      source_to_rdkit_json, status
               FROM molecule_conformers"""
        ).fetchone()
        rings = connection.execute(
            """SELECT ring_system_uid, ring_atom_mask_blob, noise_mask_blob, status
               FROM ring_training_samples ORDER BY ring_system_uid"""
        ).fetchall()
        metadata = dict(connection.execute("SELECT key, value FROM build_metadata"))

    assert conformer[0:2] == (3, "float32_le")
    assert unpack_coordinates(conformer[2], 3)[1] == pytest.approx([1.5, 0.0, 0.0])
    assert json.loads(conformer[3]) == [0, 1, 2]
    assert conformer[4] == "PASS"
    assert rings == [
        ("RS_1", b"\x01\x00\x00", b"\x00\x01\x01", "PASS"),
        ("RS_2", b"\x01\x01\x00", b"\x00\x00\x01", "PASS"),
    ]
    assert metadata["docking"] == "DISABLED"
    assert metadata["pocket_alignment"] == "DISABLED"
    assert metadata["force_field_optimization"] == "DISABLED"


def test_unsupported_vocabulary_is_recorded_without_calling_worker(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.db"
    output = tmp_path / "training_3d.db"
    _source_database(source, unsupported_atom=True)

    def forbidden(*args: object) -> dict[str, object]:
        raise AssertionError("worker must not run for an unsupported atom")

    result = build_training_assets(
        source,
        output,
        tmp_path / "progress.json",
        tmp_path / "summary.json",
        conformer_runner=forbidden,
    )

    assert result["failed_molecules"] == 1
    with sqlite3.connect(output) as connection:
        status, code, text = connection.execute(
            "SELECT status, error_code, error_text FROM molecule_conformers"
        ).fetchone()
        failure_count = connection.execute(
            "SELECT COUNT(*) FROM processing_failures"
        ).fetchone()[0]
    assert status == "FAIL"
    assert code == "ValueError"
    assert "unsupported atom type" in text
    assert failure_count == 1


def test_builder_refuses_to_overwrite_output_without_resume(tmp_path: Path) -> None:
    source = tmp_path / "source.db"
    output = tmp_path / "training_3d.db"
    _source_database(source)
    output.write_bytes(b"keep")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        build_training_assets(
            source,
            output,
            tmp_path / "progress.json",
            tmp_path / "summary.json",
            conformer_runner=_fake_conformer,
        )

    assert output.read_bytes() == b"keep"


def test_reviewed_notebook_core_generates_source_order_coordinates_when_rdkit_available() -> None:
    pytest.importorskip("rdkit")
    from gpcr_ringgen.preprocessing.diffusion_dataset.notebook_conformer import (
        generate_notebook_conformer,
    )

    atoms = [
        {"atom_index": 0, "atomic_number": 6, "symbol": "C", "formal_charge": 0, "is_aromatic": False},
        {"atom_index": 1, "atomic_number": 6, "symbol": "C", "formal_charge": 0, "is_aromatic": False},
        {"atom_index": 2, "atomic_number": 8, "symbol": "O", "formal_charge": 0, "is_aromatic": False},
    ]
    bonds = [
        {"bond_index": 0, "begin_atom_index": 0, "end_atom_index": 1, "bond_type": "SINGLE"},
        {"bond_index": 1, "begin_atom_index": 1, "end_atom_index": 2, "bond_type": "SINGLE"},
    ]

    result = generate_notebook_conformer("CCO", atoms, bonds)

    assert result["source_to_rdkit"] == [0, 1, 2]
    assert len(result["coordinates"]) == 3
    assert all(len(row) == 3 for row in result["coordinates"])
