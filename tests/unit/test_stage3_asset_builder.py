"""Tests for docking-command and pose-database preparation contracts."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from gpcr_ringgen.preprocessing.diffusion_dataset.build_stage3_assets import (
    _conformer_command,
    _create_pose_schema,
    _format_command,
    _load_command,
    _ProgressReporter,
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


def test_conformer_worker_command_carries_bounded_effort(tmp_path: Path) -> None:
    command = _conformer_command(
        "CCO",
        tmp_path / "start.sdf",
        max_conformers=2,
        embed_max_iterations=500,
        uff_max_iters=100,
    )

    assert "conformer_worker" in command[2]
    assert command[command.index("--max-conformers") + 1] == "2"
    assert command[command.index("--embed-max-iterations") + 1] == "500"
    assert command[command.index("--uff-max-iters") + 1] == "100"


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


def test_live_progress_reporter_publishes_phase_and_terminal_state(
    tmp_path: Path,
) -> None:
    progress = tmp_path / "progress.json"
    reporter = _ProgressReporter(progress, limit=100, heartbeat_seconds=0.01)

    reporter.start()
    reporter.update(
        phase="rdkit_conformer_generation",
        current_index=1,
        current_source_part="part-1",
        current_molecule_uid="mol-1",
    )
    running = json.loads(progress.read_text(encoding="utf-8"))
    reporter.finish("complete")
    complete = json.loads(progress.read_text(encoding="utf-8"))

    assert running["status"] == "running"
    assert running["phase"] == "rdkit_conformer_generation"
    assert running["current_index"] == 1
    assert running["current_molecule_uid"] == "mol-1"
    assert complete["status"] == "complete"
    assert complete["phase"] == "complete"
    assert complete["current_index"] is None
