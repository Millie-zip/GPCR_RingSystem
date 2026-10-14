"""Build pocket-independent 3D training assets from the selected molecule DB."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import struct
import subprocess
import sys
import threading
import time
from typing import Any

# The lightweight canonical module keeps offline jobs independent of torch.
from gpcr_ringgen.vocabulary import (
    ATOM_ID_TO_TYPE as SUPPORTED_ATOM_TYPES,
    BOND_ID_TO_TYPE as SUPPORTED_BOND_TYPES,
)


SOURCE_COLUMNS = (
    "source_part",
    "molecule_uid",
    "original_smiles",
    "canonical_isomeric_smiles",
    "atom_facts_json",
    "bond_facts_json",
)
RING_COLUMNS = (
    "source_part",
    "molecule_uid",
    "ring_system_uid",
    "ring_atom_indices_json",
)
ConformerRunner = Callable[
    [str, Sequence[Mapping[str, Any]], Sequence[Mapping[str, Any]], int, float],
    dict[str, Any],
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _readonly_uri(path: Path) -> str:
    return path.resolve().as_uri() + "?mode=ro"


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _json_list(value: str, name: str) -> list[Any]:
    parsed = json.loads(value)
    if not isinstance(parsed, list):
        raise TypeError(f"{name} must contain a JSON list")
    return parsed


def _require_columns(
    connection: sqlite3.Connection,
    table: str,
    required: Sequence[str],
) -> None:
    columns = {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}
    missing = sorted(set(required) - columns)
    if missing:
        raise ValueError(f"{table} is missing required columns: {', '.join(missing)}")


def _validate_source_graph(
    atoms: Sequence[Any], bonds: Sequence[Any]
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    if not atoms:
        raise ValueError("molecule contains no source atoms")
    normalized_atoms: list[Mapping[str, Any]] = []
    for expected, atom in enumerate(atoms):
        if not isinstance(atom, Mapping) or atom.get("atom_index") != expected:
            raise ValueError("atom_facts_json is not in contiguous atom-index order")
        symbol = atom.get("symbol")
        if symbol not in SUPPORTED_ATOM_TYPES:
            raise ValueError(f"unsupported atom type: {symbol!r}")
        normalized_atoms.append(atom)

    normalized_bonds: list[Mapping[str, Any]] = []
    for expected, bond in enumerate(bonds):
        if not isinstance(bond, Mapping) or bond.get("bond_index") != expected:
            raise ValueError("bond_facts_json is not in contiguous bond-index order")
        begin = bond.get("begin_atom_index")
        end = bond.get("end_atom_index")
        if not isinstance(begin, int) or not isinstance(end, int):
            raise ValueError("bond endpoint must be an integer atom index")
        if not (0 <= begin < len(atoms) and 0 <= end < len(atoms)):
            raise ValueError("bond endpoint is outside the atom range")
        if bond.get("bond_type") not in SUPPORTED_BOND_TYPES:
            raise ValueError(f"unsupported bond type: {bond.get('bond_type')!r}")
        normalized_bonds.append(bond)
    return normalized_atoms, normalized_bonds


def _pack_coordinates(coordinates: Sequence[Sequence[float]]) -> bytes:
    values = [float(value) for row in coordinates for value in row]
    return struct.pack(f"<{len(values)}f", *values)


def unpack_coordinates(blob: bytes, atom_count: int) -> list[list[float]]:
    """Decode the documented little-endian float32 coordinate representation."""

    expected = atom_count * 3
    if len(blob) != expected * 4:
        raise ValueError("coordinate blob size does not match atom_count * 3 float32")
    values = struct.unpack(f"<{expected}f", blob)
    return [list(values[index : index + 3]) for index in range(0, expected, 3)]


def _build_masks(atom_count: int, ring_indices_json: str) -> tuple[bytes, bytes]:
    raw = _json_list(ring_indices_json, "ring_atom_indices_json")
    ring_indices: set[int] = set()
    for value in raw:
        if not isinstance(value, int) or not 0 <= value < atom_count:
            raise ValueError("ring_atom_indices_json contains an invalid atom index")
        ring_indices.add(value)
    if not ring_indices:
        raise ValueError("ring system contains no ring atoms")
    ring_mask = bytes(index in ring_indices for index in range(atom_count))
    noise_mask = bytes(index not in ring_indices for index in range(atom_count))
    if not any(noise_mask):
        raise ValueError("ring system leaves no ring-external atom to diffuse")
    if any((ring_mask[index] + noise_mask[index]) != 1 for index in range(atom_count)):
        raise AssertionError("ring and noise masks are not complementary")
    return ring_mask, noise_mask


def _validate_geometry_qc(geometry: Mapping[str, Any]) -> None:
    """Reject only unmistakably collapsed or stretched generated geometries."""

    minimum_bond = geometry.get("min_bond_length_A")
    maximum_bond = geometry.get("max_bond_length_A")
    minimum_nonbonded = geometry.get("min_nonbonded_distance_A")
    if minimum_bond is not None and float(minimum_bond) < 0.6:
        raise ValueError(f"geometry QC failed: min bond length={minimum_bond} A")
    if maximum_bond is not None and float(maximum_bond) > 3.0:
        raise ValueError(f"geometry QC failed: max bond length={maximum_bond} A")
    if minimum_nonbonded is not None and float(minimum_nonbonded) < 0.5:
        raise ValueError(
            f"geometry QC failed: min nonbonded distance={minimum_nonbonded} A"
        )


def _run_worker(
    smiles: str,
    atoms: Sequence[Mapping[str, Any]],
    bonds: Sequence[Mapping[str, Any]],
    random_seed: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    request = {
        "smiles": smiles,
        "atoms": list(atoms),
        "bonds": list(bonds),
        "random_seed": random_seed,
    }
    command = [
        sys.executable,
        "-m",
        "gpcr_ringgen.preprocessing.diffusion_dataset.notebook_conformer_worker",
    ]
    try:
        completed = subprocess.run(
            command,
            input=json.dumps(request, separators=(",", ":")),
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(
            f"conformer generation exceeded {timeout_seconds:g} seconds"
        ) from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip().splitlines()
        message = detail[-1] if detail else "worker exited without an error message"
        raise RuntimeError(f"conformer worker failed: {message}")
    result = json.loads(completed.stdout)
    if not isinstance(result, dict):
        raise TypeError("conformer worker result must be a JSON object")
    return result


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA foreign_keys=ON;
        CREATE TABLE IF NOT EXISTS build_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS molecule_conformers (
            source_part TEXT NOT NULL,
            molecule_uid TEXT NOT NULL,
            selected_smiles TEXT NOT NULL,
            atom_count INTEGER NOT NULL,
            coordinate_dtype TEXT,
            coordinate_shape_json TEXT,
            x0_blob BLOB,
            source_to_rdkit_json TEXT,
            rdkit_to_source_json TEXT,
            mapping_method TEXT,
            mapping_match_count INTEGER,
            embedding_method TEXT,
            random_seed INTEGER NOT NULL,
            qc_json TEXT,
            status TEXT NOT NULL,
            error_code TEXT,
            error_text TEXT,
            updated_at_utc TEXT NOT NULL,
            PRIMARY KEY (source_part, molecule_uid)
        );
        CREATE TABLE IF NOT EXISTS ring_training_samples (
            source_part TEXT NOT NULL,
            molecule_uid TEXT NOT NULL,
            ring_system_uid TEXT NOT NULL,
            atom_count INTEGER NOT NULL,
            ring_atom_mask_blob BLOB,
            noise_mask_blob BLOB,
            status TEXT NOT NULL,
            error_code TEXT,
            error_text TEXT,
            updated_at_utc TEXT NOT NULL,
            PRIMARY KEY (source_part, ring_system_uid),
            FOREIGN KEY (source_part, molecule_uid)
              REFERENCES molecule_conformers(source_part, molecule_uid)
        );
        CREATE INDEX IF NOT EXISTS idx_ring_samples_molecule
        ON ring_training_samples(source_part, molecule_uid);
        CREATE TABLE IF NOT EXISTS processing_failures (
            failure_id INTEGER PRIMARY KEY AUTOINCREMENT,
            scope TEXT NOT NULL,
            source_part TEXT NOT NULL,
            molecule_uid TEXT NOT NULL,
            ring_system_uid TEXT,
            error_code TEXT NOT NULL,
            error_text TEXT NOT NULL,
            created_at_utc TEXT NOT NULL
        );
        """
    )


def _source_rows(
    connection: sqlite3.Connection,
    limit: int | None,
) -> Iterator[tuple[Any, ...]]:
    selected = """
        SELECT source_part, molecule_uid, original_smiles,
               canonical_isomeric_smiles, atom_facts_json, bond_facts_json
        FROM selected_molecules
        ORDER BY source_part, molecule_uid
    """
    parameters: tuple[Any, ...] = ()
    if limit is not None:
        selected += " LIMIT ?"
        parameters = (limit,)
    query = f"""
        WITH selected AS ({selected})
        SELECT s.source_part, s.molecule_uid, s.original_smiles,
               s.canonical_isomeric_smiles, s.atom_facts_json,
               s.bond_facts_json, r.ring_system_uid,
               r.ring_atom_indices_json
        FROM selected AS s
        JOIN selected_ring_systems AS r
          ON r.source_part = s.source_part
         AND r.molecule_uid = s.molecule_uid
        ORDER BY s.source_part, s.molecule_uid, r.ring_system_uid
    """
    yield from connection.execute(query, parameters)


def _group_molecules(rows: Iterator[tuple[Any, ...]]) -> Iterator[tuple[tuple[Any, ...], list[tuple[str, str]]]]:
    current_key: tuple[str, str] | None = None
    molecule: tuple[Any, ...] | None = None
    rings: list[tuple[str, str]] = []
    for row in rows:
        key = (row[0], row[1])
        if current_key is not None and key != current_key:
            assert molecule is not None
            yield molecule, rings
            rings = []
        current_key = key
        molecule = row[:6]
        rings.append((row[6], row[7]))
    if molecule is not None:
        yield molecule, rings


@dataclass
class _Progress:
    path: Path
    requested_limit: int | None
    started_at_utc: str = ""
    status: str = "initializing"
    phase: str = "initializing"
    processed_molecules: int = 0
    passed_molecules: int = 0
    failed_molecules: int = 0
    passed_ring_samples: int = 0
    failed_ring_samples: int = 0
    current_source_part: str | None = None
    current_molecule_uid: str | None = None
    message: str | None = None
    _started_monotonic: float = 0.0

    def start(self) -> None:
        self.started_at_utc = _utc_now()
        self._started_monotonic = time.monotonic()
        self.status = "running"
        self.publish()

    def publish(self) -> None:
        elapsed = max(0.0, time.monotonic() - self._started_monotonic)
        mean_seconds = (
            elapsed / self.processed_molecules
            if self.processed_molecules > 0
            else None
        )
        remaining = (
            max(0, self.requested_limit - self.processed_molecules) * mean_seconds
            if self.requested_limit is not None and mean_seconds is not None
            else None
        )
        _atomic_json(
            self.path,
            {
                "path": str(self.path),
                "requested_limit": self.requested_limit,
                "started_at_utc": self.started_at_utc,
                "status": self.status,
                "phase": self.phase,
                "processed_molecules": self.processed_molecules,
                "current_index": self.processed_molecules + 1
                if self.current_molecule_uid is not None
                else None,
                "passed_molecules": self.passed_molecules,
                "failed_molecules": self.failed_molecules,
                "passed_ring_samples": self.passed_ring_samples,
                "failed_ring_samples": self.failed_ring_samples,
                "current_source_part": self.current_source_part,
                "current_molecule_uid": self.current_molecule_uid,
                "message": self.message,
                "elapsed_seconds": elapsed,
                "mean_seconds_per_molecule": mean_seconds,
                "estimated_remaining_seconds": remaining,
                "updated_at_utc": _utc_now(),
            },
        )


def _run_with_heartbeat(
    runner: ConformerRunner,
    progress: _Progress,
    heartbeat_seconds: float,
    smiles: str,
    atoms: Sequence[Mapping[str, Any]],
    bonds: Sequence[Mapping[str, Any]],
    random_seed: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    stop = threading.Event()

    def publish_heartbeat() -> None:
        while not stop.wait(heartbeat_seconds):
            progress.publish()

    thread = threading.Thread(
        target=publish_heartbeat,
        name="training-asset-progress-heartbeat",
        daemon=True,
    )
    thread.start()
    try:
        return runner(smiles, atoms, bonds, random_seed, timeout_seconds)
    finally:
        stop.set()
        thread.join()


def _record_failure(
    output: sqlite3.Connection,
    scope: str,
    source_part: str,
    molecule_uid: str,
    ring_system_uid: str | None,
    exc: Exception,
) -> tuple[str, str]:
    code = type(exc).__name__
    text = str(exc)[:4000]
    output.execute(
        """INSERT INTO processing_failures(
               scope, source_part, molecule_uid, ring_system_uid,
               error_code, error_text, created_at_utc
           ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (scope, source_part, molecule_uid, ring_system_uid, code, text, _utc_now()),
    )
    return code, text


def build_training_assets(
    source_database: Path,
    output_database: Path,
    progress_json: Path,
    summary_json: Path,
    *,
    limit: int | None = 100,
    random_seed: int = 0xF00D,
    conformer_timeout_seconds: float = 300.0,
    heartbeat_seconds: float = 15.0,
    commit_every: int = 1,
    resume: bool = False,
    conformer_runner: ConformerRunner = _run_worker,
) -> dict[str, Any]:
    """Build a separate, resumable and auditable pocket-independent asset DB."""

    source_database = Path(source_database)
    output_database = Path(output_database)
    progress_json = Path(progress_json)
    summary_json = Path(summary_json)
    if not source_database.is_file():
        raise FileNotFoundError(source_database)
    if output_database.exists() and not resume:
        raise FileExistsError(
            f"refusing to overwrite existing output database: {output_database}"
        )
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive or omitted")
    if commit_every < 1:
        raise ValueError("commit_every must be positive")
    if conformer_timeout_seconds <= 0:
        raise ValueError("conformer_timeout_seconds must be positive")
    if heartbeat_seconds <= 0:
        raise ValueError("heartbeat_seconds must be positive")

    output_database.parent.mkdir(parents=True, exist_ok=True)
    progress = _Progress(progress_json, limit)
    progress.start()
    started = time.monotonic()

    source = sqlite3.connect(_readonly_uri(source_database), uri=True)
    output = sqlite3.connect(output_database)
    try:
        _require_columns(source, "selected_molecules", SOURCE_COLUMNS)
        _require_columns(source, "selected_ring_systems", RING_COLUMNS)
        _create_schema(output)
        metadata = {
            "schema_version": "1",
            "source_database": str(source_database.resolve()),
            "source_access": "READ_ONLY",
            "coordinate_dtype": "float32_le",
            "coordinate_shape": "[N,3]",
            "ring_mask_dtype": "uint8",
            "noise_mask_semantics": "1=ring_external_diffused,0=fixed_ring",
            "conformer_core": "RDKit ETKDGv3; random-coordinate fallback",
            "force_field_optimization": "DISABLED",
            "docking": "DISABLED",
            "pocket_alignment": "DISABLED",
            "random_seed": str(random_seed),
            "created_at_utc": _utc_now(),
        }
        output.executemany(
            "INSERT OR REPLACE INTO build_metadata(key, value) VALUES (?, ?)",
            metadata.items(),
        )
        output.commit()

        for molecule, rings in _group_molecules(_source_rows(source, limit)):
            source_part, molecule_uid, original_smiles, canonical_smiles, atom_json, bond_json = molecule
            progress.current_source_part = source_part
            progress.current_molecule_uid = molecule_uid
            progress.phase = "validate_source_graph"
            progress.publish()

            existing = output.execute(
                """SELECT status FROM molecule_conformers
                   WHERE source_part=? AND molecule_uid=?""",
                (source_part, molecule_uid),
            ).fetchone()
            if resume and existing is not None:
                progress.processed_molecules += 1
                if existing[0] == "PASS":
                    progress.passed_molecules += 1
                else:
                    progress.failed_molecules += 1
                progress.phase = "resume_skip"
                progress.publish()
                continue

            selected_smiles = original_smiles or canonical_smiles
            try:
                atoms, bonds = _validate_source_graph(
                    _json_list(atom_json, "atom_facts_json"),
                    _json_list(bond_json, "bond_facts_json"),
                )
                progress.phase = "etkdg_conformer_generation"
                progress.publish()
                result = _run_with_heartbeat(
                    conformer_runner,
                    progress,
                    heartbeat_seconds,
                    selected_smiles,
                    atoms,
                    bonds,
                    random_seed,
                    conformer_timeout_seconds,
                )
                coordinates = result["coordinates"]
                if len(coordinates) != len(atoms) or any(
                    not isinstance(row, list) or len(row) != 3 for row in coordinates
                ):
                    raise ValueError("worker coordinates do not have shape [N,3]")
                source_to_rdkit = result["source_to_rdkit"]
                rdkit_to_source = result["rdkit_to_source"]
                if sorted(source_to_rdkit) != list(range(len(atoms))):
                    raise ValueError("source_to_rdkit mapping is not bijective")
                if sorted(rdkit_to_source) != list(range(len(atoms))):
                    raise ValueError("rdkit_to_source mapping is not bijective")

                geometry = result.get("geometry", {})
                if not isinstance(geometry, Mapping):
                    raise TypeError("worker geometry QC result must be an object")
                _validate_geometry_qc(geometry)
                qc = {
                    "atom_count_match": True,
                    "mapping_bijective": True,
                    "coordinates_finite": True,
                    "topology_checked_by_worker": True,
                    "geometry": geometry,
                    "geometry_thresholds_A": {
                        "min_bond": 0.6,
                        "max_bond": 3.0,
                        "min_nonbonded": 0.5,
                    },
                    "etkdg_failure_counts": result.get("etkdg_failure_counts", []),
                    "embedding_attempts": result.get("embedding_attempts", []),
                }
                output.execute(
                    """INSERT OR REPLACE INTO molecule_conformers VALUES
                       (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        source_part, molecule_uid, selected_smiles, len(atoms),
                        "float32_le", json.dumps([len(atoms), 3]),
                        _pack_coordinates(coordinates),
                        json.dumps(source_to_rdkit, separators=(",", ":")),
                        json.dumps(rdkit_to_source, separators=(",", ":")),
                        result.get("mapping_method"),
                        int(result.get("mapping_match_count", 1)),
                        result.get("embedding_method"), random_seed,
                        json.dumps(qc, separators=(",", ":"), allow_nan=False),
                        "PASS", None, None, _utc_now(),
                    ),
                )
                progress.passed_molecules += 1

                for ring_system_uid, ring_indices_json in rings:
                    try:
                        ring_mask, noise_mask = _build_masks(
                            len(atoms), ring_indices_json
                        )
                        output.execute(
                            """INSERT OR REPLACE INTO ring_training_samples VALUES
                               (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                            (
                                source_part, molecule_uid, ring_system_uid,
                                len(atoms), ring_mask, noise_mask, "PASS",
                                None, None, _utc_now(),
                            ),
                        )
                        progress.passed_ring_samples += 1
                    except Exception as exc:
                        code, text = _record_failure(
                            output, "ring_sample", source_part, molecule_uid,
                            ring_system_uid, exc,
                        )
                        output.execute(
                            """INSERT OR REPLACE INTO ring_training_samples VALUES
                               (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                            (
                                source_part, molecule_uid, ring_system_uid,
                                len(atoms), None, None, "FAIL", code, text,
                                _utc_now(),
                            ),
                        )
                        progress.failed_ring_samples += 1
            except Exception as exc:
                code, text = _record_failure(
                    output, "molecule", source_part, molecule_uid, None, exc
                )
                atom_count = 0
                try:
                    atom_count = len(_json_list(atom_json, "atom_facts_json"))
                except Exception:
                    pass
                output.execute(
                    """INSERT OR REPLACE INTO molecule_conformers VALUES
                       (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        source_part, molecule_uid, selected_smiles, atom_count,
                        None, None, None, None, None, None, None, None,
                        random_seed, None, "FAIL", code, text, _utc_now(),
                    ),
                )
                progress.failed_molecules += 1

            progress.processed_molecules += 1
            progress.phase = "commit"
            if progress.processed_molecules % commit_every == 0:
                output.commit()
            progress.publish()

        output.commit()
        elapsed = time.monotonic() - started
        summary = {
            "status": "complete",
            "source_database": str(source_database.resolve()),
            "source_access": "READ_ONLY",
            "output_database": str(output_database.resolve()),
            "requested_limit": limit,
            "processed_molecules": progress.processed_molecules,
            "passed_molecules": progress.passed_molecules,
            "failed_molecules": progress.failed_molecules,
            "passed_ring_samples": progress.passed_ring_samples,
            "failed_ring_samples": progress.failed_ring_samples,
            "elapsed_seconds": elapsed,
            "completed_at_utc": _utc_now(),
        }
        output.executemany(
            "INSERT OR REPLACE INTO build_metadata(key, value) VALUES (?, ?)",
            ((key, str(value)) for key, value in summary.items()),
        )
        output.commit()
        _atomic_json(summary_json, summary)
        progress.status = "complete"
        progress.phase = "complete"
        progress.current_source_part = None
        progress.current_molecule_uid = None
        progress.publish()
        return summary
    except BaseException as exc:
        output.rollback()
        progress.status = "failed"
        progress.phase = "failed"
        progress.message = f"{type(exc).__name__}: {exc}"
        progress.publish()
        raise
    finally:
        source.close()
        output.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-db", type=Path, required=True)
    parser.add_argument("--output-db", type=Path, required=True)
    parser.add_argument("--progress-json", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--random-seed", type=int, default=0xF00D)
    parser.add_argument("--conformer-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--heartbeat-seconds", type=float, default=15.0)
    parser.add_argument("--commit-every", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def cli() -> None:
    args = parse_args()
    summary = build_training_assets(
        args.source_db,
        args.output_db,
        args.progress_json,
        args.summary_json,
        limit=args.limit,
        random_seed=args.random_seed,
        conformer_timeout_seconds=args.conformer_timeout_seconds,
        heartbeat_seconds=args.heartbeat_seconds,
        commit_every=args.commit_every,
        resume=args.resume,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


__all__ = ["build_training_assets", "cli", "unpack_coordinates"]
