"""Export the existing 5-HT2C library into the frozen Stage 3 data contract.

The source library and coordinate database are opened read-only. The exporter
never fabricates pocket-conditioned labels: every exported sample must have a
precomputed, atom-index-aligned docked pose in ``docked_poses``.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gpcr_ringgen.core.vocab import ATOM_TYPE_TO_ID, BOND_TYPE_TO_ID

SOURCE_QUERY = """
SELECT
    m.source_part, m.molecule_uid, r.ring_system_uid,
    m.canonical_isomeric_smiles, m.atom_facts_json, m.bond_facts_json,
    r.ring_atom_indices_json, p.pose_id, p.coordinates_json,
    p.cavity_frac, p.anchor_frac, p.span_excess_A, p.collision_frac
FROM selected_molecules AS m
JOIN selected_ring_systems AS r
  ON r.source_part = m.source_part AND r.molecule_uid = m.molecule_uid
JOIN poses.docked_poses AS p
  ON p.source_part = m.source_part AND p.molecule_uid = m.molecule_uid
WHERE p.status = 'PASS'
ORDER BY m.source_part, m.molecule_uid, r.ring_system_uid
"""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_only_uri(path: Path) -> str:
    return f"file:{path.resolve().as_posix()}?mode=ro"


def _json_list(value: str, name: str) -> list[Any]:
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} is invalid JSON: {exc}") from exc
    if not isinstance(decoded, list):
        raise TypeError(f"{name} must decode to a list")
    return decoded


def _require_columns(
    connection: sqlite3.Connection,
    schema: str,
    table: str,
    required: Iterable[str],
) -> None:
    rows = connection.execute(f'PRAGMA "{schema}".table_info("{table}")').fetchall()
    if not rows:
        raise RuntimeError(f"required table is missing: {schema}.{table}")
    present = {row[1] for row in rows}
    missing = sorted(set(required) - present)
    if missing:
        raise RuntimeError(
            f"{schema}.{table} is missing required columns: {', '.join(missing)}"
        )


def _load_pocket(path: Path) -> tuple[str, dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise TypeError("pocket JSON root must be an object")
    value = raw.get("pocket", raw)
    if not isinstance(value, Mapping):
        raise TypeError("pocket JSON must contain a pocket object")
    pocket = dict(value)
    coordinates = pocket.get("coordinates")
    mask = pocket.get("mask")
    if not isinstance(coordinates, list) or not coordinates:
        raise ValueError("pocket.coordinates must be a non-empty [P,3] list")
    if not isinstance(mask, list) or len(mask) != len(coordinates):
        raise ValueError("pocket.mask must have length P")
    for row in coordinates:
        if (
            not isinstance(row, list)
            or len(row) != 3
            or any(
                not isinstance(x, (int, float)) or not math.isfinite(x)
                for x in row
            )
        ):
            raise ValueError("pocket.coordinates must contain finite [x,y,z] rows")
    pocket_id = raw.get("pocket_id", "8DPF_13A_lorcaserin_center")
    if not isinstance(pocket_id, str) or not pocket_id.strip():
        raise ValueError("pocket_id must be a non-empty string")
    return pocket_id, pocket


def _normalize_coordinates(value: str, atom_count: int) -> list[list[float]]:
    rows = _json_list(value, "coordinates_json")
    if len(rows) != atom_count:
        raise ValueError(
            f"coordinate atom count mismatch: expected {atom_count}, got {len(rows)}"
        )
    result: list[list[float]] = []
    for row in rows:
        if not isinstance(row, list) or len(row) != 3:
            raise ValueError("coordinates_json must have shape [N,3]")
        xyz = [float(component) for component in row]
        if not all(math.isfinite(component) for component in xyz):
            raise ValueError("coordinates_json contains NaN or Inf")
        result.append(xyz)
    return result


def _validate_pose_qc(
    cavity_frac: object,
    anchor_frac: object,
    span_excess: object,
    collision_frac: object,
) -> None:
    values = (cavity_frac, anchor_frac, span_excess, collision_frac)
    if any(not isinstance(value, (int, float)) for value in values):
        raise TypeError("pose QC values must be numeric")
    numeric = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in numeric):
        raise ValueError("pose QC contains NaN or Inf")
    cavity, anchor, span, collision = numeric
    if cavity < 0.85:
        raise ValueError("pose fails cavity_frac >= 0.85")
    if anchor < 0.20:
        raise ValueError("pose fails anchor_frac >= 0.20")
    if span > 1.0:
        raise ValueError("pose fails span_excess_A <= 1.0")
    if not 0.0 <= collision <= 1.0:
        raise ValueError("collision_frac must be in [0,1]")


def _build_scaffold(
    atom_facts_json: str,
    bond_facts_json: str,
    ring_atom_indices_json: str,
    coordinates_json: str,
    *,
    atom_feature_dim: int,
    edge_feature_dim: int,
) -> dict[str, Any]:
    atoms = _json_list(atom_facts_json, "atom_facts_json")
    bonds = _json_list(bond_facts_json, "bond_facts_json")
    ring_indices_raw = _json_list(
        ring_atom_indices_json,
        "ring_atom_indices_json",
    )
    if not atoms:
        raise ValueError("molecule contains no atoms")

    atom_types: list[int] = []
    for expected_index, atom in enumerate(atoms):
        if not isinstance(atom, Mapping):
            raise TypeError("each atom fact must be an object")
        if atom.get("atom_index") != expected_index:
            raise ValueError("atom_facts_json is not in contiguous atom-index order")
        symbol = atom.get("symbol")
        if symbol not in ATOM_TYPE_TO_ID:
            raise ValueError(f"unsupported atom type: {symbol!r}")
        atom_types.append(ATOM_TYPE_TO_ID[symbol])

    edge_source: list[int] = []
    edge_target: list[int] = []
    bond_types: list[int] = []
    for expected_index, bond in enumerate(bonds):
        if not isinstance(bond, Mapping):
            raise TypeError("each bond fact must be an object")
        if bond.get("bond_index") != expected_index:
            raise ValueError("bond_facts_json is not in contiguous bond-index order")
        begin = bond.get("begin_atom_index")
        end = bond.get("end_atom_index")
        if not isinstance(begin, int) or not isinstance(end, int):
            raise TypeError("bond endpoints must be integers")
        if not (0 <= begin < len(atoms) and 0 <= end < len(atoms)):
            raise ValueError("bond endpoint is outside atom range")
        bond_type = bond.get("bond_type")
        if bond_type not in BOND_TYPE_TO_ID:
            raise ValueError(f"unsupported bond type: {bond_type!r}")
        edge_source.append(begin)
        edge_target.append(end)
        bond_types.append(BOND_TYPE_TO_ID[bond_type])

    ring_indices: set[int] = set()
    for value in ring_indices_raw:
        if not isinstance(value, int) or not 0 <= value < len(atoms):
            raise ValueError("ring_atom_indices_json contains an invalid atom index")
        ring_indices.add(value)
    if not ring_indices:
        raise ValueError("ring system contains no ring atoms")

    diffusion_mask = [index not in ring_indices for index in range(len(atoms))]
    if not any(diffusion_mask):
        raise ValueError("molecule has no ring-external atom to diffuse")

    return {
        "atom_features": [[0.0] * atom_feature_dim for _ in atoms],
        "atom_types": atom_types,
        "coordinates": _normalize_coordinates(coordinates_json, len(atoms)),
        "edge_index": [edge_source, edge_target],
        "bond_types": bond_types,
        "edge_features": [[0.0] * edge_feature_dim for _ in bonds],
        "atom_mask": [True] * len(atoms),
        "diffusion_mask": diffusion_mask,
        "edge_mask": [True] * len(bonds),
    }


def _create_output_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE export_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE pocket_templates (
            pocket_id TEXT PRIMARY KEY,
            pocket_json TEXT NOT NULL
        );
        CREATE TABLE stage3_samples (
            sample_id TEXT PRIMARY KEY,
            source_part TEXT NOT NULL,
            molecule_uid TEXT NOT NULL,
            ring_system_uid TEXT NOT NULL,
            pose_id TEXT NOT NULL,
            pocket_id TEXT NOT NULL,
            canonical_isomeric_smiles TEXT NOT NULL,
            scaffold_json TEXT NOT NULL,
            descriptor_json TEXT NOT NULL,
            qc_json TEXT NOT NULL,
            FOREIGN KEY (pocket_id) REFERENCES pocket_templates(pocket_id)
        );
        CREATE INDEX idx_stage3_samples_molecule
        ON stage3_samples(source_part, molecule_uid);
        CREATE TABLE export_failures (
            source_part TEXT,
            molecule_uid TEXT,
            ring_system_uid TEXT,
            reason TEXT NOT NULL
        );
        """
    )


def export_stage3_database(
    source_db: Path,
    poses_db: Path,
    pocket_json: Path,
    output_db: Path,
    *,
    atom_feature_dim: int = 4,
    edge_feature_dim: int = 2,
    descriptor_dim: int = 3,
    limit: int | None = None,
) -> tuple[int, int]:
    """Export validated samples and return ``(success_count, failure_count)``."""

    for path, label in ((source_db, "source"), (poses_db, "poses")):
        if not path.is_file():
            raise FileNotFoundError(f"{label} database not found: {path}")
    if not pocket_json.is_file():
        raise FileNotFoundError(f"pocket JSON not found: {pocket_json}")
    if output_db.exists():
        raise FileExistsError(f"refusing to overwrite output database: {output_db}")
    for value, name in (
        (atom_feature_dim, "atom_feature_dim"),
        (edge_feature_dim, "edge_feature_dim"),
        (descriptor_dim, "descriptor_dim"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if limit is not None and (isinstance(limit, bool) or limit < 1):
        raise ValueError("limit must be a positive integer or None")

    pocket_id, pocket = _load_pocket(pocket_json)
    output_db.parent.mkdir(parents=True, exist_ok=True)
    source = sqlite3.connect(_read_only_uri(source_db), uri=True)
    output: sqlite3.Connection | None = None
    try:
        source.execute("ATTACH DATABASE ? AS poses", (_read_only_uri(poses_db),))
        _require_columns(
            source,
            "main",
            "selected_molecules",
            ("source_part", "molecule_uid", "canonical_isomeric_smiles", "atom_facts_json", "bond_facts_json"),
        )
        _require_columns(
            source,
            "main",
            "selected_ring_systems",
            ("source_part", "molecule_uid", "ring_system_uid", "ring_atom_indices_json"),
        )
        _require_columns(
            source,
            "poses",
            "docked_poses",
            (
                "source_part", "molecule_uid", "pose_id", "coordinates_json",
                "status", "cavity_frac", "anchor_frac", "span_excess_A",
                "collision_frac",
            ),
        )

        output = sqlite3.connect(output_db)
        output.execute("PRAGMA foreign_keys=ON")
        output.execute("PRAGMA journal_mode=WAL")
        _create_output_schema(output)
        output.execute(
            "INSERT INTO pocket_templates VALUES (?, ?)",
            (pocket_id, json.dumps(pocket, separators=(",", ":"))),
        )
        metadata = {
            "schema_version": "stage3_ring_fixed_v1",
            "created_at_utc": _utc_now(),
            "source_db": str(source_db),
            "poses_db": str(poses_db),
            "source_access": "READ_ONLY",
            "atom_feature_schema": "zero_placeholder",
            "edge_feature_schema": "zero_placeholder",
            "atom_feature_dim": str(atom_feature_dim),
            "edge_feature_dim": str(edge_feature_dim),
            "descriptor_schema": "disabled_all_masked",
            "descriptor_dim": str(descriptor_dim),
            "diffusion_mask_semantics": "true=ring_external,false=fixed_ring",
            "pose_qc": "cavity>=0.85,anchor>=0.20,span<=1.0;collision_precalibrated",
        }
        output.executemany("INSERT INTO export_metadata VALUES (?, ?)", metadata.items())

        query = SOURCE_QUERY + (" LIMIT ?" if limit is not None else "")
        cursor = source.execute(query, () if limit is None else (limit,))
        successes = 0
        failures = 0
        for row in cursor:
            (
                source_part, molecule_uid, ring_system_uid, smiles, atom_facts,
                bond_facts, ring_indices, pose_id, coordinates, cavity_frac,
                anchor_frac, span_excess, collision_frac,
            ) = row
            try:
                _validate_pose_qc(
                    cavity_frac,
                    anchor_frac,
                    span_excess,
                    collision_frac,
                )
                scaffold = _build_scaffold(
                    atom_facts,
                    bond_facts,
                    ring_indices,
                    coordinates,
                    atom_feature_dim=atom_feature_dim,
                    edge_feature_dim=edge_feature_dim,
                )
                sample_id = f"{source_part}:{ring_system_uid}:{pose_id}"
                descriptor = {
                    "values": [0.0] * descriptor_dim,
                    "mask": [False] * descriptor_dim,
                }
                qc = {
                    "cavity_frac": cavity_frac,
                    "anchor_frac": anchor_frac,
                    "span_excess_A": span_excess,
                    "collision_frac": collision_frac,
                }
                output.execute(
                    "INSERT INTO stage3_samples VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        sample_id, source_part, molecule_uid, ring_system_uid,
                        pose_id, pocket_id, smiles,
                        json.dumps(scaffold, separators=(",", ":")),
                        json.dumps(descriptor, separators=(",", ":")),
                        json.dumps(qc, separators=(",", ":")),
                    ),
                )
                successes += 1
            except (TypeError, ValueError) as exc:
                output.execute(
                    "INSERT INTO export_failures VALUES (?, ?, ?, ?)",
                    (source_part, molecule_uid, ring_system_uid, str(exc)),
                )
                failures += 1
            if (successes + failures) % 10_000 == 0:
                output.commit()

        output.executemany(
            "INSERT INTO export_metadata VALUES (?, ?)",
            (
                ("successful_samples", str(successes)),
                ("failed_samples", str(failures)),
                ("completed_at_utc", _utc_now()),
            ),
        )
        quick_check = output.execute("PRAGMA quick_check").fetchone()[0]
        if quick_check != "ok":
            raise RuntimeError(f"output database quick_check failed: {quick_check}")
        output.commit()
        return successes, failures
    except Exception:
        if output is not None:
            output.close()
        raise
    finally:
        source.close()
        if output is not None:
            try:
                output.close()
            except sqlite3.Error:
                pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export ring-fixed Stage 3 samples without modifying source DBs."
    )
    parser.add_argument("--source-db", type=Path, required=True)
    parser.add_argument("--poses-db", type=Path, required=True)
    parser.add_argument("--pocket-json", type=Path, required=True)
    parser.add_argument("--output-db", type=Path, required=True)
    parser.add_argument("--atom-feature-dim", type=int, default=4)
    parser.add_argument("--edge-feature-dim", type=int, default=2)
    parser.add_argument("--descriptor-dim", type=int, default=3)
    parser.add_argument("--limit", type=int)
    return parser


def cli(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    successes, failures = export_stage3_database(
        args.source_db,
        args.poses_db,
        args.pocket_json,
        args.output_db,
        atom_feature_dim=args.atom_feature_dim,
        edge_feature_dim=args.edge_feature_dim,
        descriptor_dim=args.descriptor_dim,
        limit=args.limit,
    )
    print(f"Export complete: successful={successes:,}, failed={failures:,}")
    print(f"Output database: {args.output_db}")


__all__ = ["build_parser", "cli", "export_stage3_database"]
