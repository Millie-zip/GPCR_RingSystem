"""Tests for docking-command and pose-database preparation contracts."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from gpcr_ringgen.preprocessing.diffusion_dataset.build_stage3_assets import (
    _create_pose_schema,
    _format_command,
    _load_command,
)


def test_docking_command_requires_safe_path_placeholders(tmp_path: Path) -> None:
    config = tmp_path / "dock.json"
    config.write_text(
        json.dumps(
            {
                "command": [
                    "dock",
                    "--receptor",
                    "{receptor_pdb}",
                    "--ligand",
                    "{ligand_sdf}",
                    "--out",
                    "{output_sdf}",
                    "--reference",
                    "{reference_pdb}",
                ]
            }
        ),
        encoding="utf-8",
    )

    template = _load_command(config)
    command = _format_command(
        template,
        ligand_sdf=Path("ligand.sdf"),
        output_sdf=Path("output.sdf"),
        receptor_pdb=Path("receptor.pdb"),
        reference_pdb=Path("reference.pdb"),
    )

    assert command[0] == "dock"
    assert "ligand.sdf" in command
    assert "output.sdf" in command
    assert "receptor.pdb" in command
    assert "reference.pdb" in command


def test_invalid_docking_command_is_rejected(tmp_path: Path) -> None:
    config = tmp_path / "dock.json"
    config.write_text('{"command":["dock","{ligand_sdf}"]}', encoding="utf-8")

    with pytest.raises(ValueError, match="missing placeholders"):
        _load_command(config)


def test_pose_database_schema_matches_stage3_exporter(tmp_path: Path) -> None:
    database = tmp_path / "poses.db"
    with sqlite3.connect(database) as connection:
        _create_pose_schema(connection)
        columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(docked_poses)")
        }

    assert {
        "source_part",
        "molecule_uid",
        "pose_id",
        "coordinates_json",
        "status",
        "cavity_frac",
        "anchor_frac",
        "span_excess_A",
        "collision_frac",
    } <= columns
