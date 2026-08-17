"""Build the 8DPF pocket JSON and an atom-aligned docked-pose database.

RDKit creates starting conformers; a user-configured external docking command
produces pocket poses. Only poses passing the existing 8DPF QC are stored.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_SOURCE_DB = Path(
    "/share/home/weiliao02/lyx/zinc_project/5ht2c_3d_diffusion_library_v1/"
    "5ht2c_3d_diffusion_library_v1.db"
)
DEFAULT_RECEPTOR_PDB = Path(
    "/share/home/weiliao02/lyx/zinc_project/8DPF_V1_prepared/"
    "8DPF_receptor_V1_prepared.pdb"
)
DEFAULT_REFERENCE_PDB = Path(
    "/share/home/weiliao02/lyx/zinc_project/8DPF_V1_prepared/"
    "8DPF_lorcaserin_reference_raw.pdb"
)

DEFAULT_HEARTBEAT_SECONDS = 15.0
DEFAULT_COMMIT_EVERY = 1


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class _ProgressReporter:
    """Atomically publish live build state, including blocked phase duration."""

    def __init__(
        self,
        path: Path,
        *,
        limit: int | None,
        heartbeat_seconds: float,
    ) -> None:
        self.path = path
        self.heartbeat_seconds = heartbeat_seconds
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        now = _utc_now()
        self._phase_started = time.monotonic()
        self._state: dict[str, Any] = {
            "schema_version": 1,
            "pid": os.getpid(),
            "status": "running",
            "phase": "initializing",
            "started_at": now,
            "updated_at": now,
            "phase_started_at": now,
            "phase_elapsed_seconds": 0.0,
            "limit": limit,
            "current_index": None,
            "current_source_part": None,
            "current_molecule_uid": None,
            "processed": 0,
            "passed": 0,
            "failed": 0,
            "last_error": None,
        }

    def start(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            self._write_locked()
        self._thread = threading.Thread(
            target=self._heartbeat_loop,
            name="stage3-progress-heartbeat",
            daemon=True,
        )
        self._thread.start()

    def update(self, *, phase: str | None = None, **changes: Any) -> None:
        with self._lock:
            if phase is not None and phase != self._state["phase"]:
                now = _utc_now()
                self._state["phase"] = phase
                self._state["phase_started_at"] = now
                self._phase_started = time.monotonic()
            self._state.update(changes)
            self._write_locked()

    def finish(self, status: str, *, last_error: str | None = None) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.heartbeat_seconds * 2.0))
        self.update(
            phase=status,
            status=status,
            last_error=last_error,
            current_index=None,
            current_source_part=None,
            current_molecule_uid=None,
        )

    def _heartbeat_loop(self) -> None:
        while not self._stop.wait(self.heartbeat_seconds):
            try:
                with self._lock:
                    self._write_locked()
            except OSError as exc:
                print(f"progress heartbeat write failed: {exc}", flush=True)

    def _write_locked(self) -> None:
        self._state["updated_at"] = _utc_now()
        self._state["phase_elapsed_seconds"] = round(
            time.monotonic() - self._phase_started,
            1,
        )
        temporary = self.path.with_name(f".{self.path.name}.tmp")
        temporary.write_text(
            json.dumps(self._state, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(self.path)


def _read_only_uri(path: Path) -> str:
    return f"file:{path.resolve().as_posix()}?mode=ro"


def _load_command(path: Path) -> list[str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise TypeError("docking config root must be an object")
    command = raw.get("command")
    if (
        not isinstance(command, list)
        or not command
        or any(not isinstance(token, str) or not token for token in command)
    ):
        raise ValueError("docking config command must be a non-empty string list")
    required = {"{ligand_sdf}", "{output_sdf}", "{receptor_pdb}"}
    joined = " ".join(command)
    missing = sorted(token for token in required if token not in joined)
    if missing:
        raise ValueError(
            "docking command is missing placeholders: " + ", ".join(missing)
        )
    return command


def _format_command(
    template: Sequence[str],
    *,
    ligand_sdf: Path,
    output_sdf: Path,
    receptor_pdb: Path,
    reference_pdb: Path,
) -> list[str]:
    values = {
        "ligand_sdf": str(ligand_sdf),
        "output_sdf": str(output_sdf),
        "receptor_pdb": str(receptor_pdb),
        "reference_pdb": str(reference_pdb),
    }
    return [token.format(**values) for token in template]


def _pocket_payload(
    pocket_atoms: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "coordinates": [atom["coord"].tolist() for atom in pocket_atoms],
        "features": [[0.0, 0.0] for _ in pocket_atoms],
        "mask": [True for _ in pocket_atoms],
        "atom_types": [str(atom["atom_name"]) for atom in pocket_atoms],
        "elements": [str(atom["element"]).title() for atom in pocket_atoms],
        "residue_names": [str(atom["resname"]) for atom in pocket_atoms],
        "residue_ids": [str(atom["resseq"]) for atom in pocket_atoms],
        "chain_ids": [str(atom["chain"]) for atom in pocket_atoms],
    }


def _write_pocket_json(
    path: Path,
    pocket_atoms: Sequence[Mapping[str, Any]],
    *,
    receptor_pdb: Path,
    reference_pdb: Path,
) -> None:
    payload = {
        "pocket_id": "8DPF_lorcaserin_center_13A",
        "selection": {
            "receptor": str(receptor_pdb),
            "reference_ligand": str(reference_pdb),
            "heavy_atoms_only": True,
            "center": "mean_of_lorcaserin_heavy_atoms",
            "radius_A": 13.0,
        },
        "pocket": _pocket_payload(pocket_atoms),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _create_pose_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS build_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS docked_poses (
            source_part TEXT NOT NULL,
            molecule_uid TEXT NOT NULL,
            pose_id TEXT NOT NULL,
            coordinates_json TEXT,
            status TEXT NOT NULL,
            cavity_frac REAL,
            anchor_frac REAL,
            span_excess_A REAL,
            collision_frac REAL,
            soft_collision_frac REAL,
            min_distance_A REAL,
            docking_score REAL,
            error_text TEXT,
            PRIMARY KEY (source_part, molecule_uid)
        );
        CREATE INDEX IF NOT EXISTS idx_docked_poses_status
        ON docked_poses(status);
        """
    )


def _source_rows(
    source: sqlite3.Connection,
    *,
    limit: int | None,
) -> sqlite3.Cursor:
    query = """
        SELECT source_part, molecule_uid, original_smiles, atom_facts_json
        FROM selected_molecules
        ORDER BY source_part, molecule_uid
    """
    if limit is not None:
        return source.execute(query + " LIMIT ?", (limit,))
    return source.execute(query)


def _expected_symbols(atom_facts_json: str) -> list[str]:
    atoms = json.loads(atom_facts_json)
    if not isinstance(atoms, list) or not atoms:
        raise ValueError("atom_facts_json must be a non-empty list")
    result: list[str] = []
    for index, atom in enumerate(atoms):
        if not isinstance(atom, Mapping) or atom.get("atom_index") != index:
            raise ValueError("atom facts are not in contiguous atom-index order")
        symbol = atom.get("symbol")
        if not isinstance(symbol, str):
            raise TypeError("atom symbol must be a string")
        result.append(symbol)
    return result


def _conformer_command(
    smiles: str,
    output_sdf: Path,
    *,
    max_conformers: int,
    embed_max_iterations: int,
    uff_max_iters: int,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "gpcr_ringgen.preprocessing.diffusion_dataset.conformer_worker",
        "--smiles",
        smiles,
        "--output-sdf",
        str(output_sdf),
        "--max-conformers",
        str(max_conformers),
        "--embed-max-iterations",
        str(embed_max_iterations),
        "--uff-max-iters",
        str(uff_max_iters),
    ]


def _heavy_coordinates(
    molecule: Any,
    expected_symbols: Sequence[str],
) -> list[list[float]]:
    observed = [
        atom.GetSymbol()
        for atom in molecule.GetAtoms()
        if atom.GetAtomicNum() not in (0, 1)
    ]
    if observed != list(expected_symbols):
        raise ValueError("docking output changed heavy-atom identity or order")
    conformer = molecule.GetConformer()
    coordinates: list[list[float]] = []
    for atom in molecule.GetAtoms():
        if atom.GetAtomicNum() in (0, 1):
            continue
        point = conformer.GetAtomPosition(atom.GetIdx())
        coordinates.append([float(point.x), float(point.y), float(point.z)])
    return coordinates


def _docking_score(molecule: Any) -> float | None:
    for name in ("minimizedAffinity", "CNNaffinity", "docking_score", "SCORE"):
        if molecule.HasProp(name):
            try:
                return float(molecule.GetProp(name))
            except ValueError:
                continue
    return None


def _best_passing_pose(
    output_sdf: Path,
    expected_symbols: Sequence[str],
    context: Mapping[str, Any],
) -> tuple[list[list[float]], dict[str, Any], float | None]:
    import numpy as np
    from rdkit import Chem

    from gpcr_ringgen.preprocessing.pocket_prefilter.pipeline import (
        evaluate_pose,
        get_vdw_radius,
        pose_is_pass,
    )

    best: tuple[list[list[float]], dict[str, Any], float | None] | None = None
    supplier = Chem.SDMolSupplier(str(output_sdf), removeHs=False)
    for molecule in supplier:
        if molecule is None:
            continue
        coordinates = _heavy_coordinates(molecule, expected_symbols)
        radii = np.asarray([get_vdw_radius(symbol) for symbol in expected_symbols])
        metrics = evaluate_pose(
            np.asarray(coordinates),
            radii,
            context["receptor_coords"],
            context["receptor_radii"],
            context["ligand_coords"],
            context["cavity_indices"],
            context["grid_origin"],
            context["cavity_frame_center"],
            context["cavity_axes"],
            context["cavity_spans"],
        )
        if not pose_is_pass(metrics, context["pass_thresholds"]):
            continue
        score = _docking_score(molecule)
        candidate = (coordinates, metrics, score)
        if best is None or score is not None and (best[2] is None or score < best[2]) or score is None and best[2] is None and metrics["score"] < best[1]["score"]:
            best = candidate
    if best is None:
        raise ValueError("no docked pose passed calibrated pocket QC")
    return best


def _prepare_pocket_context(
    receptor_pdb: Path,
    reference_pdb: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import numpy as np

    from gpcr_ringgen.preprocessing.pocket_prefilter.pipeline import (
        BASE_PASS_SEVERE_ATOM_FRAC,
        BASE_PASS_SOFT_ATOM_FRAC,
        NATIVE_SEVERE_MARGIN,
        NATIVE_SOFT_MARGIN,
        POCKET_RECEPTOR_RADIUS_A,
        build_ligand_anchored_cavity_grid,
        compute_cavity_principal_frame,
        evaluate_pose,
        read_pdb_atoms,
    )

    receptor_atoms = read_pdb_atoms(receptor_pdb, True, False, True)
    ligand_atoms = read_pdb_atoms(reference_pdb, False, True, True)
    if not receptor_atoms or not ligand_atoms:
        raise RuntimeError("receptor/reference PDB did not yield heavy atoms")
    ligand_coords = np.asarray([atom["coord"] for atom in ligand_atoms])
    pocket_center = ligand_coords.mean(axis=0)
    pocket_atoms = [
        atom
        for atom in receptor_atoms
        if np.linalg.norm(atom["coord"] - pocket_center) <= POCKET_RECEPTOR_RADIUS_A
    ]
    receptor_coords = np.asarray([atom["coord"] for atom in pocket_atoms])
    receptor_radii = np.asarray([atom["radius"] for atom in pocket_atoms])
    cavity = build_ligand_anchored_cavity_grid(pocket_atoms, ligand_atoms)
    frame_center, axes, spans = compute_cavity_principal_frame(cavity["points"])
    native_radii = np.asarray([atom["radius"] for atom in ligand_atoms])
    native = evaluate_pose(
        ligand_coords,
        native_radii,
        receptor_coords,
        receptor_radii,
        ligand_coords,
        cavity["indices"],
        cavity["origin"],
        frame_center,
        axes,
        spans,
    )
    thresholds = {
        "pass_severe_frac": max(
            BASE_PASS_SEVERE_ATOM_FRAC,
            native["severe_atom_frac"] + NATIVE_SEVERE_MARGIN,
        ),
        "pass_soft_frac": max(
            BASE_PASS_SOFT_ATOM_FRAC,
            native["soft_atom_frac"] + NATIVE_SOFT_MARGIN,
        ),
    }
    context = {
        "receptor_coords": receptor_coords,
        "receptor_radii": receptor_radii,
        "ligand_coords": ligand_coords,
        "cavity_indices": cavity["indices"],
        "grid_origin": cavity["origin"],
        "cavity_frame_center": frame_center,
        "cavity_axes": axes,
        "cavity_spans": spans,
        "pass_thresholds": thresholds,
    }
    return pocket_atoms, context


def build_stage3_assets(
    source_db: Path,
    receptor_pdb: Path,
    reference_pdb: Path,
    docking_config: Path,
    *,
    output_dir: Path | None = None,
    limit: int | None = None,
    resume: bool = False,
    progress_file: Path | None = None,
    heartbeat_seconds: float = DEFAULT_HEARTBEAT_SECONDS,
    commit_every: int = DEFAULT_COMMIT_EVERY,
    max_conformers: int = 10,
    embed_max_iterations: int = 2000,
    uff_max_iters: int = 250,
    conformer_timeout_seconds: float = 1800.0,
) -> tuple[Path, Path]:
    """Generate sibling pose DB/pocket JSON and return their paths."""

    for path in (source_db, receptor_pdb, reference_pdb, docking_config):
        if not path.is_file():
            raise FileNotFoundError(path)
    if limit is not None and (isinstance(limit, bool) or limit < 1):
        raise ValueError("limit must be a positive integer or None")
    if heartbeat_seconds <= 0:
        raise ValueError("heartbeat_seconds must be positive")
    if conformer_timeout_seconds <= 0:
        raise ValueError("conformer_timeout_seconds must be positive")
    if isinstance(commit_every, bool) or commit_every < 1:
        raise ValueError("commit_every must be a positive integer")
    for name, value in (
        ("max_conformers", max_conformers),
        ("embed_max_iterations", embed_max_iterations),
        ("uff_max_iters", uff_max_iters),
    ):
        if isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    destination = source_db.parent if output_dir is None else output_dir
    destination.mkdir(parents=True, exist_ok=True)
    poses_db = destination / "8dpf_docked_poses.db"
    pocket_json = destination / "8dpf_13A_pocket.json"
    live_progress = (
        destination / "stage3_build_progress.json"
        if progress_file is None
        else progress_file
    )
    if not resume and (poses_db.exists() or pocket_json.exists()):
        raise FileExistsError("output exists; use --resume or choose another directory")

    reporter = _ProgressReporter(
        live_progress,
        limit=limit,
        heartbeat_seconds=heartbeat_seconds,
    )
    reporter.start()
    try:
        reporter.update(phase="loading_docking_configuration")
        command_template = _load_command(docking_config)
        executable = command_template[0]
        if not Path(executable).is_file() and shutil.which(executable) is None:
            raise FileNotFoundError(f"docking executable not found: {executable}")

        reporter.update(phase="preparing_pocket")
        pocket_atoms, context = _prepare_pocket_context(receptor_pdb, reference_pdb)
        if not pocket_json.exists():
            _write_pocket_json(
                pocket_json,
                pocket_atoms,
                receptor_pdb=receptor_pdb,
                reference_pdb=reference_pdb,
            )

        reporter.update(phase="opening_databases")
        source = sqlite3.connect(_read_only_uri(source_db), uri=True)
        output = sqlite3.connect(poses_db)
        try:
            _create_pose_schema(output)
            output.executemany(
                "INSERT OR REPLACE INTO build_metadata VALUES (?, ?)",
                (
                    ("source_db", str(source_db)),
                    ("receptor_pdb", str(receptor_pdb)),
                    ("reference_pdb", str(reference_pdb)),
                    ("pocket_json", str(pocket_json)),
                    ("progress_file", str(live_progress)),
                    ("max_conformers", str(max_conformers)),
                    ("embed_max_iterations", str(embed_max_iterations)),
                    ("uff_max_iters", str(uff_max_iters)),
                    (
                        "conformer_timeout_seconds",
                        str(conformer_timeout_seconds),
                    ),
                    ("coordinate_contract", "heavy atoms in atom_facts_json order"),
                    ("rdkit_conformer_role", "docking initialization only"),
                ),
            )
            output.commit()
            processed = 0
            passed = 0
            failed = 0
            reporter.update(phase="reading_source_rows")
            rows = _source_rows(source, limit=limit)
            for row_index, (
                source_part,
                molecule_uid,
                smiles,
                atom_facts,
            ) in enumerate(rows, start=1):
                if resume and output.execute(
                    "SELECT 1 FROM docked_poses WHERE source_part=? AND molecule_uid=?",
                    (source_part, molecule_uid),
                ).fetchone():
                    continue
                reporter.update(
                    phase="validating_source_record",
                    current_index=row_index,
                    current_source_part=source_part,
                    current_molecule_uid=molecule_uid,
                )
                status = "FAIL"
                coordinates_json = None
                metrics: dict[str, Any] = {}
                docking_score = None
                error_text = None
                try:
                    symbols = _expected_symbols(atom_facts)
                    with tempfile.TemporaryDirectory(
                        prefix="gpcr_stage3_"
                    ) as temp_name:
                        temp = Path(temp_name)
                        ligand_sdf = temp / "starting_conformers.sdf"
                        docked_sdf = temp / "docked.sdf"
                        reporter.update(phase="rdkit_conformer_generation")
                        subprocess.run(
                            _conformer_command(
                                smiles,
                                ligand_sdf,
                                max_conformers=max_conformers,
                                embed_max_iterations=embed_max_iterations,
                                uff_max_iters=uff_max_iters,
                            ),
                            check=True,
                            timeout=conformer_timeout_seconds,
                        )
                        command = _format_command(
                            command_template,
                            ligand_sdf=ligand_sdf,
                            output_sdf=docked_sdf,
                            receptor_pdb=receptor_pdb,
                            reference_pdb=reference_pdb,
                        )
                        reporter.update(phase="docking")
                        subprocess.run(command, check=True)
                        if not docked_sdf.is_file():
                            raise RuntimeError(
                                "docking command did not create output_sdf"
                            )
                        reporter.update(phase="pocket_quality_control")
                        coordinates, metrics, docking_score = _best_passing_pose(
                            docked_sdf,
                            symbols,
                            context,
                        )
                    status = "PASS"
                    coordinates_json = json.dumps(
                        coordinates,
                        separators=(",", ":"),
                    )
                except Exception as exc:  # noqa: BLE001 - persist row failure
                    error_text = f"{type(exc).__name__}: {exc}"

                reporter.update(phase="writing_result")
                output.execute(
                    """
                    INSERT OR REPLACE INTO docked_poses VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                    """,
                    (
                        source_part,
                        molecule_uid,
                        "best_pose_v1",
                        coordinates_json,
                        status,
                        metrics.get("cavity_frac"),
                        metrics.get("anchor_frac"),
                        metrics.get("max_span_excess_A"),
                        metrics.get("severe_atom_frac"),
                        metrics.get("soft_atom_frac"),
                        metrics.get("min_distance"),
                        docking_score,
                        error_text,
                    ),
                )
                processed += 1
                passed += int(status == "PASS")
                failed += int(status != "PASS")
                if processed % commit_every == 0:
                    output.commit()
                reporter.update(
                    phase="molecule_complete",
                    processed=processed,
                    passed=passed,
                    failed=failed,
                    last_error=error_text,
                )
                print(
                    f"[{processed}/{limit or '?'}] {status} "
                    f"{source_part}/{molecule_uid}"
                    + (f" | {error_text}" if error_text else ""),
                    flush=True,
                )
            output.commit()
            reporter.update(phase="database_integrity_check")
            if output.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                raise RuntimeError("pose database quick_check failed")
        finally:
            output.close()
            source.close()
    except BaseException as exc:
        status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        reporter.finish(status, last_error=f"{type(exc).__name__}: {exc}")
        raise
    reporter.finish("complete")
    return poses_db, pocket_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build docked x0 coordinates and the fixed 8DPF pocket JSON."
    )
    parser.add_argument("--source-db", type=Path, default=DEFAULT_SOURCE_DB)
    parser.add_argument("--receptor-pdb", type=Path, default=DEFAULT_RECEPTOR_PDB)
    parser.add_argument("--reference-pdb", type=Path, default=DEFAULT_REFERENCE_PDB)
    parser.add_argument("--docking-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress-file", type=Path)
    parser.add_argument(
        "--heartbeat-seconds",
        type=float,
        default=DEFAULT_HEARTBEAT_SECONDS,
    )
    parser.add_argument(
        "--commit-every",
        type=int,
        default=DEFAULT_COMMIT_EVERY,
    )
    parser.add_argument("--max-conformers", type=int, default=10)
    parser.add_argument("--embed-max-iterations", type=int, default=2000)
    parser.add_argument("--uff-max-iters", type=int, default=250)
    parser.add_argument(
        "--conformer-timeout-seconds",
        type=float,
        default=1800.0,
    )
    return parser


def cli(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    poses_db, pocket_json = build_stage3_assets(
        args.source_db,
        args.receptor_pdb,
        args.reference_pdb,
        args.docking_config,
        output_dir=args.output_dir,
        limit=args.limit,
        resume=args.resume,
        progress_file=args.progress_file,
        heartbeat_seconds=args.heartbeat_seconds,
        commit_every=args.commit_every,
        max_conformers=args.max_conformers,
        embed_max_iterations=args.embed_max_iterations,
        uff_max_iters=args.uff_max_iters,
        conformer_timeout_seconds=args.conformer_timeout_seconds,
    )
    print(f"Pose database: {poses_db}")
    print(f"Pocket JSON: {pocket_json}")


__all__ = ["build_parser", "build_stage3_assets", "cli"]
