#!/usr/bin/env python3
"""Make the full 5-HT2C diffusion library self-contained.

Adds referenced topology records plus source-part-qualified ring systems,
scaffold variants/atom mappings, and attachment sites. Existing core tables
are never overwritten. Processing is committed one source part at a time and
is resumable through upstream_enrichment_parts.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sqlite3
import sys
import time
from pathlib import Path
from urllib.parse import quote


RING_COLUMNS = [
    "ring_system_uid", "molecule_uid", "run_id", "system_index",
    "system_count_in_molecule", "basis_ring_indices_json", "basis_ring_sizes_json",
    "ring_atom_indices_json", "ring_bond_indices_json", "shared_atom_indices_json",
    "shared_bond_indices_json", "shared_bond_atom_pairs_json", "fused_ring_pair_count",
    "bridge_contact_pair_count", "sssr_basis_ring_count", "cycle_rank", "ring_system_type",
]

VARIANT_COLUMNS = [
    "scaffold_occurrence_uid", "ring_system_uid", "molecule_uid", "run_id",
    "variant_type", "scaffold_smiles", "mapped_scaffold_smiles",
    "full_atom_indices_json", "full_bond_records_json", "full_to_submol_map_json",
    "submol_to_full_map_json", "scaffold_atom_count", "scaffold_bond_count",
    "sanitize_status",
]

ATTACHMENT_COLUMNS = [
    "attachment_uid", "ring_system_uid", "molecule_uid", "run_id",
    "scaffold_variant_type", "attachment_order", "scaffold_full_atom_index",
    "external_full_atom_index", "original_bond_index", "original_bond_type",
    "original_bond_order",
]


def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def qident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def readonly_uri(path: Path) -> str:
    return "file:" + quote(str(path.resolve()).replace("\\", "/"), safe="/:_") + "?mode=ro"


def table_exists(con: sqlite3.Connection, table: str, schema: str = "main") -> bool:
    row = con.execute(
        f"SELECT 1 FROM {qident(schema)}.sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    return row is not None


def columns(con: sqlite3.Connection, table: str, schema: str = "main") -> list[str]:
    return [str(row[1]) for row in con.execute(f"PRAGMA {qident(schema)}.table_info({qident(table)})")]


def require_columns(con: sqlite3.Connection, table: str, required: set[str], schema="main"):
    actual = set(columns(con, table, schema))
    missing = required.difference(actual)
    if missing:
        raise RuntimeError(f"{schema}.{table} missing {sorted(missing)}; actual={sorted(actual)}")


def install_schema(con: sqlite3.Connection, create_secondary_indexes: bool = True) -> None:
    con.executescript(
        """
        CREATE TABLE IF NOT EXISTS selected_ring_systems (
            source_part TEXT NOT NULL,
            ring_system_uid TEXT NOT NULL,
            molecule_uid TEXT NOT NULL,
            run_id TEXT NOT NULL,
            system_index INTEGER NOT NULL,
            system_count_in_molecule INTEGER NOT NULL,
            basis_ring_indices_json TEXT NOT NULL,
            basis_ring_sizes_json TEXT NOT NULL,
            ring_atom_indices_json TEXT NOT NULL,
            ring_bond_indices_json TEXT NOT NULL,
            shared_atom_indices_json TEXT NOT NULL,
            shared_bond_indices_json TEXT NOT NULL,
            shared_bond_atom_pairs_json TEXT NOT NULL,
            fused_ring_pair_count INTEGER NOT NULL,
            bridge_contact_pair_count INTEGER NOT NULL,
            sssr_basis_ring_count INTEGER NOT NULL,
            cycle_rank INTEGER NOT NULL,
            ring_system_type TEXT NOT NULL,
            PRIMARY KEY (source_part, ring_system_uid)
        );

        CREATE TABLE IF NOT EXISTS selected_scaffold_variants (
            source_part TEXT NOT NULL,
            scaffold_occurrence_uid TEXT NOT NULL,
            ring_system_uid TEXT NOT NULL,
            molecule_uid TEXT NOT NULL,
            run_id TEXT NOT NULL,
            variant_type TEXT NOT NULL,
            scaffold_smiles TEXT NOT NULL,
            mapped_scaffold_smiles TEXT,
            full_atom_indices_json TEXT NOT NULL,
            full_bond_records_json TEXT NOT NULL,
            full_to_submol_map_json TEXT NOT NULL,
            submol_to_full_map_json TEXT NOT NULL,
            scaffold_atom_count INTEGER NOT NULL,
            scaffold_bond_count INTEGER NOT NULL,
            sanitize_status TEXT NOT NULL,
            PRIMARY KEY (source_part, scaffold_occurrence_uid)
        );

        CREATE TABLE IF NOT EXISTS selected_attachment_sites (
            source_part TEXT NOT NULL,
            attachment_uid TEXT NOT NULL,
            ring_system_uid TEXT NOT NULL,
            molecule_uid TEXT NOT NULL,
            run_id TEXT NOT NULL,
            scaffold_variant_type TEXT NOT NULL,
            attachment_order INTEGER NOT NULL,
            scaffold_full_atom_index INTEGER NOT NULL,
            external_full_atom_index INTEGER NOT NULL,
            original_bond_index INTEGER NOT NULL,
            original_bond_type TEXT NOT NULL,
            original_bond_order REAL NOT NULL,
            PRIMARY KEY (source_part, attachment_uid)
        );

        CREATE TABLE IF NOT EXISTS upstream_enrichment_parts (
            source_part TEXT PRIMARY KEY,
            source_path TEXT NOT NULL,
            status TEXT NOT NULL,
            wanted_ring_systems INTEGER,
            copied_ring_systems INTEGER,
            copied_scaffold_variants INTEGER,
            copied_attachment_sites INTEGER,
            started_at_utc TEXT,
            completed_at_utc TEXT,
            elapsed_seconds REAL,
            error_text TEXT
        );

        CREATE TABLE IF NOT EXISTS upstream_enrichment_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        """
    )
    if create_secondary_indexes:
        install_secondary_indexes(con)
    con.commit()


def install_secondary_indexes(con: sqlite3.Connection) -> None:
    con.executescript(
        """
        CREATE INDEX IF NOT EXISTS idx_selected_ring_systems_molecule
            ON selected_ring_systems(source_part, molecule_uid);
        CREATE INDEX IF NOT EXISTS idx_selected_variants_ring
            ON selected_scaffold_variants(source_part, ring_system_uid);
        CREATE INDEX IF NOT EXISTS idx_selected_variants_molecule
            ON selected_scaffold_variants(source_part, molecule_uid);
        CREATE INDEX IF NOT EXISTS idx_selected_attachments_ring
            ON selected_attachment_sites(source_part, ring_system_uid);
        CREATE INDEX IF NOT EXISTS idx_selected_attachments_molecule
            ON selected_attachment_sites(source_part, molecule_uid);
        """
    )


def validate_destination_schema(con: sqlite3.Connection) -> None:
    require_columns(con, "selected_ring_systems", {"source_part", *RING_COLUMNS})
    require_columns(con, "selected_scaffold_variants", {"source_part", *VARIANT_COLUMNS})
    require_columns(con, "selected_attachment_sites", {"source_part", *ATTACHMENT_COLUMNS})


def copy_topologies(con: sqlite3.Connection, global_db: Path) -> None:
    print("[topologies] attaching global scaffold database read-only", flush=True)
    con.execute("ATTACH DATABASE ? AS global_src", (readonly_uri(global_db),))
    try:
        require_columns(con, "topologies", {"topology_uid"}, "global_src")
        source_cols = columns(con, "topologies", "global_src")
        if not table_exists(con, "topologies"):
            source_sql = con.execute(
                "SELECT sql FROM global_src.sqlite_master WHERE type='table' AND name='topologies'"
            ).fetchone()[0]
            if not source_sql:
                raise RuntimeError("Cannot read CREATE TABLE SQL for global_src.topologies")
            con.execute(source_sql)
            con.commit()
        local_cols = columns(con, "topologies")
        if local_cols != source_cols:
            raise RuntimeError(
                f"Local topologies schema differs from upstream: local={local_cols}, upstream={source_cols}"
            )
        col_sql = ",".join(qident(c) for c in source_cols)
        started = time.time()
        con.execute("BEGIN IMMEDIATE")
        before = con.total_changes
        con.execute(
            f"""
            INSERT OR IGNORE INTO main.topologies ({col_sql})
            SELECT {','.join('g.' + qident(c) for c in source_cols)}
            FROM global_src.topologies AS g
            JOIN (SELECT DISTINCT topology_uid FROM main.selected_scaffolds
                  WHERE topology_uid IS NOT NULL) AS wanted
              ON wanted.topology_uid=g.topology_uid
            """
        )
        inserted = con.total_changes - before
        con.execute(
            "INSERT OR REPLACE INTO upstream_enrichment_metadata(key,value) VALUES (?,?)",
            ("topologies_last_copy", json.dumps({"inserted": inserted, "utc": now_utc()})),
        )
        con.commit()
        local_count = con.execute("SELECT COUNT(*) FROM topologies").fetchone()[0]
        print(
            f"[topologies] inserted={inserted:,}; local total={local_count:,}; "
            f"elapsed={time.time()-started:.1f}s",
            flush=True,
        )
    except Exception:
        con.rollback()
        raise
    finally:
        con.execute("DETACH DATABASE global_src")


def source_part_path(source_part: str, parts_dir: Path) -> Path:
    raw = Path(source_part)
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    candidates.extend([parts_dir / source_part, parts_dir / raw.name])
    if not raw.name.endswith(".db"):
        candidates.extend([parts_dir / f"{source_part}.db", parts_dir / f"{raw.name}.db"])
    digits = re.findall(r"\d+", raw.name)
    if digits:
        candidates.append(parts_dir / f"part_{int(digits[-1]):04d}.db")
    seen = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in seen and resolved.is_file():
            return resolved
        seen.add(resolved)
    raise FileNotFoundError(
        f"Cannot resolve source_part={source_part!r} under {parts_dir}; tried={candidates}"
    )


def validate_source(con: sqlite3.Connection) -> None:
    require_columns(con, "ring_systems", set(RING_COLUMNS), "src")
    require_columns(con, "scaffold_variants", set(VARIANT_COLUMNS), "src")
    require_columns(con, "attachment_sites", set(ATTACHMENT_COLUMNS), "src")


def progress_handler(stage_box: list[str], start: float):
    last = [time.time()]

    def handler():
        now = time.time()
        if now - last[0] >= 60:
            print(
                f"  still running: {stage_box[0]}; total elapsed={(now-start)/60:.1f} min",
                flush=True,
            )
            last[0] = now
        return 0

    return handler


def copy_one_part(con: sqlite3.Connection, source_part: str, path: Path) -> None:
    part_started = time.time()
    con.execute("DROP TABLE IF EXISTS temp.wanted_ring_systems")
    con.execute(
        "CREATE TEMP TABLE wanted_ring_systems(ring_system_uid TEXT PRIMARY KEY) WITHOUT ROWID"
    )
    con.execute(
        """
        INSERT OR IGNORE INTO wanted_ring_systems
        SELECT ring_system_uid
        FROM selected_scaffold_instances
        WHERE source_part=? AND ring_system_uid IS NOT NULL
        """,
        (source_part,),
    )
    wanted = con.execute("SELECT COUNT(*) FROM wanted_ring_systems").fetchone()[0]
    print(f"[{source_part}] wanted unique ring systems={wanted:,}; source={path}", flush=True)

    con.execute(
        """
        INSERT OR REPLACE INTO upstream_enrichment_parts
        (source_part,source_path,status,wanted_ring_systems,started_at_utc,error_text)
        VALUES (?,?,?,?,?,NULL)
        """,
        (source_part, str(path), "RUNNING", wanted, now_utc()),
    )
    con.commit()

    con.execute("ATTACH DATABASE ? AS src", (readonly_uri(path),))
    try:
        validate_source(con)
        stage = [f"{source_part}: beginning"]
        con.set_progress_handler(progress_handler(stage, part_started), 1_000_000)
        con.execute("BEGIN IMMEDIATE")

        before = con.total_changes
        stage[0] = f"{source_part}: copying ring_systems"
        con.execute(
            f"""
            INSERT OR IGNORE INTO selected_ring_systems
            (source_part,{','.join(qident(c) for c in RING_COLUMNS)})
            SELECT ?,{','.join('s.' + qident(c) for c in RING_COLUMNS)}
            FROM src.ring_systems AS s
            JOIN wanted_ring_systems AS w ON w.ring_system_uid=s.ring_system_uid
            """,
            (source_part,),
        )
        copied_rings = con.total_changes - before
        print(f"[{source_part}] ring systems inserted={copied_rings:,}", flush=True)

        before = con.total_changes
        stage[0] = f"{source_part}: copying scaffold_variants and atom mappings"
        con.execute(
            f"""
            INSERT OR IGNORE INTO selected_scaffold_variants
            (source_part,{','.join(qident(c) for c in VARIANT_COLUMNS)})
            SELECT ?,{','.join('s.' + qident(c) for c in VARIANT_COLUMNS)}
            FROM src.scaffold_variants AS s
            JOIN wanted_ring_systems AS w ON w.ring_system_uid=s.ring_system_uid
            """,
            (source_part,),
        )
        copied_variants = con.total_changes - before
        print(f"[{source_part}] scaffold variants inserted={copied_variants:,}", flush=True)

        before = con.total_changes
        stage[0] = f"{source_part}: copying attachment_sites"
        con.execute(
            f"""
            INSERT OR IGNORE INTO selected_attachment_sites
            (source_part,{','.join(qident(c) for c in ATTACHMENT_COLUMNS)})
            SELECT ?,{','.join('s.' + qident(c) for c in ATTACHMENT_COLUMNS)}
            FROM src.attachment_sites AS s
            JOIN wanted_ring_systems AS w ON w.ring_system_uid=s.ring_system_uid
            """,
            (source_part,),
        )
        copied_attachments = con.total_changes - before
        elapsed = time.time() - part_started
        print(f"[{source_part}] attachment sites inserted={copied_attachments:,}", flush=True)

        con.execute(
            """
            UPDATE upstream_enrichment_parts
            SET status='COMPLETE',copied_ring_systems=?,copied_scaffold_variants=?,
                copied_attachment_sites=?,completed_at_utc=?,elapsed_seconds=?,error_text=NULL
            WHERE source_part=?
            """,
            (
                copied_rings, copied_variants, copied_attachments,
                now_utc(), elapsed, source_part,
            ),
        )
        con.commit()
        print(f"[{source_part}] COMPLETE in {elapsed/60:.2f} min", flush=True)
    except BaseException as exc:
        con.rollback()
        con.set_progress_handler(None, 0)
        con.execute(
            """
            UPDATE upstream_enrichment_parts
            SET status='ERROR',completed_at_utc=?,elapsed_seconds=?,error_text=?
            WHERE source_part=?
            """,
            (now_utc(), time.time() - part_started, repr(exc), source_part),
        )
        con.commit()
        raise
    finally:
        con.set_progress_handler(None, 0)
        con.execute("DETACH DATABASE src")
        try:
            con.execute("PRAGMA wal_checkpoint(PASSIVE)")
        except sqlite3.DatabaseError:
            pass


def final_audit(con: sqlite3.Connection) -> dict:
    print("[final audit] checking local reference closure", flush=True)
    result = {}
    result["completed_parts"] = con.execute(
        "SELECT COUNT(*) FROM upstream_enrichment_parts WHERE status='COMPLETE'"
    ).fetchone()[0]
    result["expected_parts"] = con.execute(
        "SELECT COUNT(DISTINCT source_part) FROM selected_scaffold_instances"
    ).fetchone()[0]
    result["topologies"] = con.execute("SELECT COUNT(*) FROM topologies").fetchone()[0]
    result["selected_ring_systems"] = con.execute(
        "SELECT COUNT(*) FROM selected_ring_systems"
    ).fetchone()[0]
    result["selected_scaffold_variants"] = con.execute(
        "SELECT COUNT(*) FROM selected_scaffold_variants"
    ).fetchone()[0]
    result["selected_attachment_sites"] = con.execute(
        "SELECT COUNT(*) FROM selected_attachment_sites"
    ).fetchone()[0]
    result["missing_topology_references"] = con.execute(
        """
        SELECT COUNT(*) FROM (
          SELECT DISTINCT s.topology_uid
          FROM selected_scaffolds s
          LEFT JOIN topologies t ON t.topology_uid=s.topology_uid
          WHERE s.topology_uid IS NOT NULL AND t.topology_uid IS NULL
        )
        """
    ).fetchone()[0]
    result["missing_ring_system_references"] = con.execute(
        """
        SELECT COUNT(*) FROM (
          SELECT DISTINCT i.source_part,i.ring_system_uid
          FROM selected_scaffold_instances i
          LEFT JOIN selected_ring_systems r
            ON r.source_part=i.source_part AND r.ring_system_uid=i.ring_system_uid
          WHERE i.ring_system_uid IS NOT NULL AND r.ring_system_uid IS NULL
        )
        """
    ).fetchone()[0]
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    con.execute(
        "INSERT OR REPLACE INTO upstream_enrichment_metadata(key,value) VALUES (?,?)",
        ("final_audit", json.dumps({**result, "utc": now_utc()})),
    )
    con.commit()
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--library-db", required=True, type=Path)
    p.add_argument("--global-db", required=True, type=Path)
    p.add_argument("--parts-dir", required=True, type=Path)
    p.add_argument("--only-part", help="Process one exact source_part value for testing")
    p.add_argument("--resume", action="store_true", help="Skip parts already marked COMPLETE")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    for path in (args.library_db, args.global_db, args.parts_dir):
        if not path.exists():
            raise FileNotFoundError(path)
    # Required so read-only file: URIs used by ATTACH DATABASE are parsed.
    con = sqlite3.connect(args.library_db, timeout=120, uri=True)
    con.execute("PRAGMA busy_timeout=120000")
    con.execute("PRAGMA foreign_keys=OFF")
    try:
        for table in (
            "selected_molecules", "selected_scaffolds", "selected_scaffold_instances",
            "molecule_scaffold_map",
        ):
            if not table_exists(con, table):
                raise RuntimeError(f"Required core table missing: {table}")
        require_columns(
            con, "selected_scaffold_instances", {"source_part", "ring_system_uid", "molecule_uid"}
        )
        require_columns(con, "selected_scaffolds", {"chemical_uid", "topology_uid"})
        print("Installing additive destination tables (existing core tables untouched)...", flush=True)
        install_schema(con)
        validate_destination_schema(con)
        copy_topologies(con, args.global_db)

        all_parts = [
            str(row[0])
            for row in con.execute(
                "SELECT DISTINCT source_part FROM selected_scaffold_instances "
                "WHERE source_part IS NOT NULL ORDER BY source_part"
            )
        ]
        if args.only_part:
            if args.only_part not in all_parts:
                raise ValueError(
                    f"--only-part {args.only_part!r} is not an exact DB source_part value; "
                    f"examples={all_parts[:10]}"
                )
            all_parts = [args.only_part]
        completed = {
            str(row[0])
            for row in con.execute(
                "SELECT source_part FROM upstream_enrichment_parts WHERE status='COMPLETE'"
            )
        }
        pending = [p for p in all_parts if not (args.resume and p in completed)]
        print(
            f"Source parts: selected={len(all_parts)}, already complete={len(completed)}, "
            f"to process={len(pending)}",
            flush=True,
        )
        for index, source_part in enumerate(pending, start=1):
            print(f"\n=== part {index}/{len(pending)}: {source_part} ===", flush=True)
            path = source_part_path(source_part, args.parts_dir)
            copy_one_part(con, source_part, path)
        final_audit(con)
        print("Upstream closure augmentation finished.", flush=True)
    finally:
        con.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted. Current part rolled back; rerun with --resume.", file=sys.stderr, flush=True)
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise
