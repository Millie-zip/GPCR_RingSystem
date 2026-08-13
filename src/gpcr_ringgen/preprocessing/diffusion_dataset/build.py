#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build a 5-HT2C-selected scaffold -> strict scaffold instance -> original molecule library.

Upstream databases are opened READ-ONLY.
The script does not modify the prefilter DB, global scaffold index DB, or any of the 120 source part DBs.

Expected selected scaffold set:
    PASS + BORDERLINE = 314,659

Output beneath ``GPCR_RINGGEN_DATA_ROOT``:
    5ht2c_3d_diffusion_library_v1/
        5ht2c_3d_diffusion_library_v1.db
        selected_scaffolds.csv
        build_report.txt
"""

from __future__ import annotations

import csv
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from gpcr_ringgen.preprocessing.paths import data_root


# =============================================================================
# Fixed project paths
# =============================================================================

ROOT = data_root()

PREFILTER_DB = (
    ROOT
    / "gpcr_pocket_prefilter_final_parallel_v1"
    / "gpcr_pocket_prefilter_final_parallel_v1.db"
)

GLOBAL_DB = (
    ROOT
    / "global_scaffold_analysis_strict_full_v4"
    / "global_scaffold_index.db"
)

PARTS_DIR = (
    ROOT
    / "merged_fused_ring_database"
    / "fused_ring_master_library"
    / "_parts_master_v1"
)

OUTDIR = ROOT / "5ht2c_3d_diffusion_library_v1"
OUTDB = OUTDIR / "5ht2c_3d_diffusion_library_v1.db"
SCAFFOLD_CSV = OUTDIR / "selected_scaffolds.csv"
REPORT_FILE = OUTDIR / "build_report.txt"

EXPECTED_SELECTED_SCAFFOLDS = 314_659
EXPECTED_SOURCE_PARTS = 120
ACCEPTED_STATUSES = ("PASS", "BORDERLINE")


# =============================================================================
# Helpers
# =============================================================================

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def ro_uri(path: Path) -> str:
    # URI filename that forces SQLite read-only access.
    return f"file:{path.resolve()}?mode=ro"


def qident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def table_exists(conn: sqlite3.Connection, schema: str, table: str) -> bool:
    row = conn.execute(
        f"""
        SELECT 1
        FROM {qident(schema)}.sqlite_master
        WHERE type='table' AND name=?
        LIMIT 1
        """,
        (table,),
    ).fetchone()
    return row is not None


def table_info(conn: sqlite3.Connection, schema: str, table: str):
    return conn.execute(
        f"PRAGMA {qident(schema)}.table_info({qident(table)})"
    ).fetchall()


def columns(conn: sqlite3.Connection, schema: str, table: str):
    return [row[1] for row in table_info(conn, schema, table)]


def choose_column(all_columns, candidates, label):
    for candidate in candidates:
        if candidate in all_columns:
            return candidate
    raise RuntimeError(
        f"Could not find {label}. "
        f"Tried {candidates}. Columns present: {all_columns}"
    )


def count(conn: sqlite3.Connection, table: str) -> int:
    return conn.execute(f"SELECT COUNT(*) FROM {qident(table)}").fetchone()[0]


# =============================================================================
# Input validation
# =============================================================================

def validate_files():
    if not PREFILTER_DB.is_file():
        raise RuntimeError(f"Prefilter DB not found: {PREFILTER_DB}")

    if not GLOBAL_DB.is_file():
        raise RuntimeError(f"Global scaffold DB not found: {GLOBAL_DB}")

    if not PARTS_DIR.is_dir():
        raise RuntimeError(f"Parts directory not found: {PARTS_DIR}")

    part_files = sorted(PARTS_DIR.glob("part_*.db"))

    if len(part_files) != EXPECTED_SOURCE_PARTS:
        raise RuntimeError(
            f"Expected {EXPECTED_SOURCE_PARTS} source part DBs, "
            f"found {len(part_files)} in {PARTS_DIR}"
        )

    log("Input file validation OK")
    log(f"Prefilter DB : {PREFILTER_DB}")
    log(f"Global DB    : {GLOBAL_DB}")
    log(f"Parts dir    : {PARTS_DIR}")
    log(f"Found {len(part_files)} source part DBs")


# =============================================================================
# Main
# =============================================================================

def main():
    start = time.time()
    validate_files()

    OUTDIR.mkdir(parents=True, exist_ok=True)

    if OUTDB.exists():
        raise RuntimeError(
            f"Output database already exists:\n{OUTDB}\n"
            "This script refuses to overwrite an existing output. "
            "If it is a failed partial build, remove the output directory and rerun."
        )

    # Create writable output DB. uri=True is required so READ-ONLY ATTACH URIs
    # are interpreted correctly.
    conn = sqlite3.connect(str(OUTDB), timeout=300, uri=True)

    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA temp_store=FILE")
        conn.execute("PRAGMA cache_size=-262144")
        conn.execute("PRAGMA foreign_keys=OFF")

        # ---------------------------------------------------------------------
        # Attach upstream DBs READ-ONLY
        # ---------------------------------------------------------------------

        conn.execute("ATTACH DATABASE ? AS prefilter", (ro_uri(PREFILTER_DB),))
        conn.execute("ATTACH DATABASE ? AS gidx", (ro_uri(GLOBAL_DB),))

        # ---------------------------------------------------------------------
        # Validate actual schemas and auto-detect prefilter UID/SMILES columns
        # ---------------------------------------------------------------------

        if not table_exists(conn, "prefilter", "scaffold_results"):
            raise RuntimeError("prefilter.scaffold_results table not found")

        if not table_exists(conn, "gidx", "chemical_scaffolds"):
            raise RuntimeError("gidx.chemical_scaffolds table not found")

        if not table_exists(conn, "gidx", "strict_scaffold_instances"):
            raise RuntimeError("gidx.strict_scaffold_instances table not found")

        pre_cols = columns(conn, "prefilter", "scaffold_results")
        chem_cols = columns(conn, "gidx", "chemical_scaffolds")
        inst_cols = columns(conn, "gidx", "strict_scaffold_instances")

        pre_uid_col = choose_column(
            pre_cols,
            ["chemical_scaffold_id", "chemical_uid"],
            "prefilter chemical scaffold ID column",
        )
        pre_smiles_col = choose_column(
            pre_cols,
            ["scaffold_smiles", "chemical_smiles"],
            "prefilter scaffold SMILES column",
        )

        if "status" not in pre_cols:
            raise RuntimeError(
                f"status column missing from prefilter.scaffold_results. "
                f"Columns: {pre_cols}"
            )

        for required in ("chemical_uid", "chemical_smiles"):
            if required not in chem_cols:
                raise RuntimeError(
                    f"{required} missing from gidx.chemical_scaffolds. "
                    f"Columns: {chem_cols}"
                )

        for required in ("source_part", "molecule_uid", "chemical_uid"):
            if required not in inst_cols:
                raise RuntimeError(
                    f"{required} missing from gidx.strict_scaffold_instances. "
                    f"Columns: {inst_cols}"
                )

        log(
            "Schema validation OK: "
            f"prefilter UID={pre_uid_col}, prefilter SMILES={pre_smiles_col}"
        )

        # ---------------------------------------------------------------------
        # Build metadata
        # ---------------------------------------------------------------------

        conn.execute(
            """
            CREATE TABLE build_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )

        metadata = {
            "library_name": "5ht2c_3d_diffusion_library_v1",
            "build_started_utc": utc_now(),
            "prefilter_db": str(PREFILTER_DB),
            "global_scaffold_db": str(GLOBAL_DB),
            "source_parts_dir": str(PARTS_DIR),
            "source_part_count": str(EXPECTED_SOURCE_PARTS),
            "accepted_statuses": ",".join(ACCEPTED_STATUSES),
            "expected_selected_scaffolds": str(EXPECTED_SELECTED_SCAFFOLDS),
            "prefilter_uid_column": pre_uid_col,
            "prefilter_smiles_column": pre_smiles_col,
            "upstream_access_mode": "READ_ONLY",
        }

        conn.executemany(
            "INSERT INTO build_metadata(key, value) VALUES (?, ?)",
            metadata.items(),
        )
        conn.commit()

        # ---------------------------------------------------------------------
        # STEP 1: Select PASS + BORDERLINE scaffolds
        # ---------------------------------------------------------------------

        log("STEP 1/6: selecting PASS + BORDERLINE scaffolds")

        # Keep all global chemical scaffold fields and selected prefilter fields.
        # The join uses the ACTUAL prefilter column chemical_scaffold_id.
        conn.execute(
            f"""
            CREATE TABLE selected_scaffolds AS
            SELECT
                g.*,
                p.status AS prefilter_status,
                p.reason AS prefilter_reason,
                p.{qident(pre_smiles_col)} AS prefilter_scaffold_smiles
            FROM gidx.chemical_scaffolds AS g
            JOIN prefilter.scaffold_results AS p
              ON g.chemical_uid = p.{qident(pre_uid_col)}
            WHERE p.status IN ('PASS', 'BORDERLINE')
            """
        )

        conn.execute(
            """
            CREATE UNIQUE INDEX idx_selected_scaffolds_uid
            ON selected_scaffolds(chemical_uid)
            """
        )
        conn.execute(
            """
            CREATE INDEX idx_selected_scaffolds_status
            ON selected_scaffolds(prefilter_status)
            """
        )

        n_scaffolds = count(conn, "selected_scaffolds")
        log(f"selected_scaffolds = {n_scaffolds:,}")

        if n_scaffolds != EXPECTED_SELECTED_SCAFFOLDS:
            status_counts = conn.execute(
                """
                SELECT prefilter_status, COUNT(*)
                FROM selected_scaffolds
                GROUP BY prefilter_status
                ORDER BY prefilter_status
                """
            ).fetchall()
            raise RuntimeError(
                f"Selected scaffold count mismatch: expected "
                f"{EXPECTED_SELECTED_SCAFFOLDS:,}, got {n_scaffolds:,}. "
                f"Status counts={status_counts}"
            )

        # Extra audit: ensure no duplicate UID in selected set.
        n_unique_scaffolds = conn.execute(
            "SELECT COUNT(DISTINCT chemical_uid) FROM selected_scaffolds"
        ).fetchone()[0]

        if n_unique_scaffolds != n_scaffolds:
            raise RuntimeError(
                f"Duplicate selected chemical_uid values detected: "
                f"rows={n_scaffolds:,}, unique={n_unique_scaffolds:,}"
            )

        conn.commit()

        # ---------------------------------------------------------------------
        # STEP 2: Copy all strict scaffold instances for selected scaffolds
        # ---------------------------------------------------------------------

        log("STEP 2/6: selecting strict scaffold instances")

        conn.execute(
            """
            CREATE TABLE selected_scaffold_instances AS
            SELECT i.*
            FROM gidx.strict_scaffold_instances AS i
            JOIN selected_scaffolds AS s
              ON i.chemical_uid = s.chemical_uid
            """
        )

        conn.execute(
            """
            CREATE INDEX idx_ssi_chemical_uid
            ON selected_scaffold_instances(chemical_uid)
            """
        )
        conn.execute(
            """
            CREATE INDEX idx_ssi_source_molecule
            ON selected_scaffold_instances(source_part, molecule_uid)
            """
        )

        n_instances = count(conn, "selected_scaffold_instances")
        log(f"selected_scaffold_instances = {n_instances:,}")

        if n_instances == 0:
            raise RuntimeError("No strict scaffold instances were recovered")

        conn.commit()

        # ---------------------------------------------------------------------
        # STEP 3: Build unique molecule keys + molecule/scaffold map
        # ---------------------------------------------------------------------

        log("STEP 3/6: building molecule keys and molecule-scaffold map")

        conn.execute(
            """
            CREATE TABLE selected_molecule_keys AS
            SELECT DISTINCT
                source_part,
                molecule_uid
            FROM selected_scaffold_instances
            """
        )

        conn.execute(
            """
            CREATE UNIQUE INDEX idx_selected_molecule_keys
            ON selected_molecule_keys(source_part, molecule_uid)
            """
        )

        conn.execute(
            """
            CREATE TABLE molecule_scaffold_map AS
            SELECT DISTINCT
                i.source_part,
                i.molecule_uid,
                i.chemical_uid,
                s.chemical_smiles,
                s.prefilter_status
            FROM selected_scaffold_instances AS i
            JOIN selected_scaffolds AS s
              ON i.chemical_uid = s.chemical_uid
            """
        )

        conn.execute(
            """
            CREATE UNIQUE INDEX idx_molecule_scaffold_map_unique
            ON molecule_scaffold_map(source_part, molecule_uid, chemical_uid)
            """
        )
        conn.execute(
            """
            CREATE INDEX idx_molecule_scaffold_map_uid
            ON molecule_scaffold_map(chemical_uid)
            """
        )

        n_molecule_keys = count(conn, "selected_molecule_keys")
        n_map = count(conn, "molecule_scaffold_map")

        log(f"unique source_part+molecule_uid pairs = {n_molecule_keys:,}")
        log(f"molecule-scaffold relations = {n_map:,}")

        if n_molecule_keys == 0:
            raise RuntimeError("No molecule keys were recovered")

        conn.commit()

        # ---------------------------------------------------------------------
        # STEP 4: Discover and validate molecules schema across ALL 120 parts
        # ---------------------------------------------------------------------

        log("STEP 4/6: validating molecules table schema across 120 source parts")

        template_path = PARTS_DIR / "part_0000.db"
        template = sqlite3.connect(ro_uri(template_path), uri=True)
        try:
            template_info = template.execute("PRAGMA table_info(molecules)").fetchall()
        finally:
            template.close()

        if not template_info:
            raise RuntimeError(f"molecules table missing in {template_path}")

        molecule_cols = [row[1] for row in template_info]

        if "molecule_uid" not in molecule_cols:
            raise RuntimeError(
                f"molecule_uid missing from source molecules table. "
                f"Columns={molecule_cols}"
            )

        # Validate schema consistency before copying anything.
        for idx, part_path in enumerate(sorted(PARTS_DIR.glob("part_*.db")), 1):
            src = sqlite3.connect(ro_uri(part_path), uri=True)
            try:
                info = src.execute("PRAGMA table_info(molecules)").fetchall()
            finally:
                src.close()

            if not info:
                raise RuntimeError(f"molecules table missing in {part_path.name}")

            cols_here = [row[1] for row in info]
            if cols_here != molecule_cols:
                raise RuntimeError(
                    f"molecules schema mismatch in {part_path.name}\n"
                    f"Expected: {molecule_cols}\n"
                    f"Found:    {cols_here}"
                )

            if idx % 20 == 0 or idx == EXPECTED_SOURCE_PARTS:
                log(f"validated source molecule schema: {idx}/{EXPECTED_SOURCE_PARTS}")

        # Create destination table preserving every source molecule column,
        # plus source_part so the source location is never lost.
        defs = ['"source_part" TEXT NOT NULL']

        for _cid, name, dtype, notnull, default_value, _pk in template_info:
            dtype = dtype or "TEXT"
            piece = f"{qident(name)} {dtype}"
            if notnull:
                piece += " NOT NULL"
            if default_value is not None:
                piece += f" DEFAULT {default_value}"
            defs.append(piece)

        conn.execute(
            f"""
            CREATE TABLE selected_molecules (
                {", ".join(defs)},
                PRIMARY KEY (source_part, molecule_uid)
            )
            """
        )
        conn.commit()

        # ---------------------------------------------------------------------
        # STEP 5: Copy complete original molecule rows from required parts
        # ---------------------------------------------------------------------

        log("STEP 5/6: copying complete original molecule records")

        required_parts = [
            row[0]
            for row in conn.execute(
                """
                SELECT DISTINCT source_part
                FROM selected_molecule_keys
                ORDER BY source_part
                """
            )
        ]

        unknown_parts = [
            p for p in required_parts
            if not (PARTS_DIR / p).is_file()
        ]
        if unknown_parts:
            raise RuntimeError(
                "selected_molecule_keys references missing source parts: "
                + ", ".join(unknown_parts[:20])
            )

        dst_cols_sql = ", ".join(
            [qident("source_part")] + [qident(c) for c in molecule_cols]
        )
        src_cols_sql = ", ".join(
            ["m." + qident(c) for c in molecule_cols]
        )

        for idx, source_part in enumerate(required_parts, 1):
            part_path = PARTS_DIR / source_part

            conn.execute(
                "ATTACH DATABASE ? AS srcpart",
                (ro_uri(part_path),),
            )

            try:
                before = conn.total_changes

                conn.execute(
                    f"""
                    INSERT INTO selected_molecules ({dst_cols_sql})
                    SELECT
                        ?,
                        {src_cols_sql}
                    FROM srcpart.molecules AS m
                    JOIN selected_molecule_keys AS k
                      ON k.source_part = ?
                     AND k.molecule_uid = m.molecule_uid
                    """,
                    (source_part, source_part),
                )

                conn.commit()
                added = conn.total_changes - before

            finally:
                conn.execute("DETACH DATABASE srcpart")

            if idx == 1 or idx % 10 == 0 or idx == len(required_parts):
                current = count(conn, "selected_molecules")
                log(
                    f"[{idx:03d}/{len(required_parts):03d}] "
                    f"{source_part}: +{added:,}; cumulative={current:,}"
                )

        n_molecules = count(conn, "selected_molecules")

        # Verify every selected key was recovered.
        n_missing = conn.execute(
            """
            SELECT COUNT(*)
            FROM selected_molecule_keys AS k
            LEFT JOIN selected_molecules AS m
              ON m.source_part = k.source_part
             AND m.molecule_uid = k.molecule_uid
            WHERE m.molecule_uid IS NULL
            """
        ).fetchone()[0]

        log(f"selected_molecules = {n_molecules:,}")
        log(f"missing molecule records = {n_missing:,}")

        if n_molecules != n_molecule_keys:
            raise RuntimeError(
                f"Recovered molecule count mismatch: "
                f"expected {n_molecule_keys:,} keys, copied {n_molecules:,} molecules"
            )

        if n_missing != 0:
            raise RuntimeError(
                f"{n_missing:,} selected molecule records were not recovered"
            )

        # Useful downstream indexes.
        if "canonical_isomeric_smiles" in molecule_cols:
            conn.execute(
                """
                CREATE INDEX idx_selected_molecules_iso_smiles
                ON selected_molecules(canonical_isomeric_smiles)
                """
            )

        if "canonical_nonisomeric_smiles" in molecule_cols:
            conn.execute(
                """
                CREATE INDEX idx_selected_molecules_noniso_smiles
                ON selected_molecules(canonical_nonisomeric_smiles)
                """
            )

        if "source_cid" in molecule_cols:
            conn.execute(
                """
                CREATE INDEX idx_selected_molecules_source_cid
                ON selected_molecules(source_cid)
                """
            )

        conn.commit()

        # ---------------------------------------------------------------------
        # STEP 6: Final QA + CSV + report
        # ---------------------------------------------------------------------

        log("STEP 6/6: final QA, CSV export and report")

        status_counts = conn.execute(
            """
            SELECT prefilter_status, COUNT(*)
            FROM selected_scaffolds
            GROUP BY prefilter_status
            ORDER BY prefilter_status
            """
        ).fetchall()

        # Exact expected accepted status counts from the completed prefilter DB.
        status_dict = dict(status_counts)
        if status_dict.get("PASS", 0) != 299_633:
            raise RuntimeError(
                f"PASS count mismatch: expected 299,633, "
                f"got {status_dict.get('PASS', 0):,}"
            )
        if status_dict.get("BORDERLINE", 0) != 15_026:
            raise RuntimeError(
                f"BORDERLINE count mismatch: expected 15,026, "
                f"got {status_dict.get('BORDERLINE', 0):,}"
            )

        # Sanity check that scaffold SMILES match between the prefilter result
        # and global scaffold index. Canonicalization differences are not
        # expected here because the prefilter was built from this scaffold table.
        n_smiles_mismatch = conn.execute(
            """
            SELECT COUNT(*)
            FROM selected_scaffolds
            WHERE chemical_smiles IS NOT prefilter_scaffold_smiles
            """
        ).fetchone()[0]

        quick_check = conn.execute("PRAGMA quick_check").fetchone()[0]
        if quick_check != "ok":
            raise RuntimeError(f"SQLite PRAGMA quick_check failed: {quick_check}")

        # Export one row per selected chemical scaffold.
        scaffold_cols = columns(conn, "main", "selected_scaffolds")

        with open(SCAFFOLD_CSV, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(scaffold_cols)

            cur = conn.execute(
                "SELECT "
                + ", ".join(qident(c) for c in scaffold_cols)
                + " FROM selected_scaffolds ORDER BY chemical_uid"
            )

            while True:
                rows = cur.fetchmany(10_000)
                if not rows:
                    break
                writer.writerows(rows)

        elapsed = time.time() - start

        final_meta = {
            "build_completed_utc": utc_now(),
            "pass_scaffolds": str(status_dict.get("PASS", 0)),
            "borderline_scaffolds": str(status_dict.get("BORDERLINE", 0)),
            "selected_scaffold_count": str(n_scaffolds),
            "selected_strict_instance_count": str(n_instances),
            "selected_unique_molecule_count": str(n_molecules),
            "molecule_scaffold_relation_count": str(n_map),
            "missing_molecule_record_count": str(n_missing),
            "scaffold_smiles_mismatch_count": str(n_smiles_mismatch),
            "sqlite_quick_check": quick_check,
            "elapsed_seconds": f"{elapsed:.3f}",
        }

        conn.executemany(
            "INSERT OR REPLACE INTO build_metadata(key, value) VALUES (?, ?)",
            final_meta.items(),
        )

        conn.execute("ANALYZE")
        conn.execute("PRAGMA optimize")
        conn.commit()

        with open(REPORT_FILE, "w", encoding="utf-8") as report:
            report.write("5-HT2C 3D DIFFUSION LIBRARY V1 BUILD REPORT\n")
            report.write("=" * 78 + "\n\n")

            report.write("SELECTION\n")
            report.write("-" * 78 + "\n")
            report.write(f"PASS                         : {status_dict.get('PASS', 0):,}\n")
            report.write(f"BORDERLINE                   : {status_dict.get('BORDERLINE', 0):,}\n")
            report.write(f"SELECTED SCAFFOLDS           : {n_scaffolds:,}\n\n")

            report.write("MAPPING / MOLECULE COUNTS\n")
            report.write("-" * 78 + "\n")
            report.write(f"selected_scaffold_instances  : {n_instances:,}\n")
            report.write(f"unique selected molecules    : {n_molecules:,}\n")
            report.write(f"molecule-scaffold relations  : {n_map:,}\n")
            report.write(f"missing molecule records     : {n_missing:,}\n")
            report.write(f"scaffold SMILES mismatches   : {n_smiles_mismatch:,}\n\n")

            report.write("OUTPUT\n")
            report.write("-" * 78 + "\n")
            report.write(f"database                     : {OUTDB}\n")
            report.write(f"selected scaffold CSV        : {SCAFFOLD_CSV}\n")
            report.write(f"source parts                 : {len(required_parts)}\n")
            report.write("upstream access              : READ-ONLY\n")
            report.write(f"SQLite quick_check           : {quick_check}\n")
            report.write(f"elapsed minutes              : {elapsed / 60:.2f}\n")

        log("=" * 78)
        log("5-HT2C 3D DIFFUSION LIBRARY V1 BUILD COMPLETE")
        log("=" * 78)
        log(f"PASS                       = {status_dict.get('PASS', 0):,}")
        log(f"BORDERLINE                 = {status_dict.get('BORDERLINE', 0):,}")
        log(f"selected scaffolds         = {n_scaffolds:,}")
        log(f"strict scaffold instances  = {n_instances:,}")
        log(f"unique selected molecules  = {n_molecules:,}")
        log(f"molecule-scaffold links    = {n_map:,}")
        log(f"missing molecule records   = {n_missing:,}")
        log(f"SMILES mismatches          = {n_smiles_mismatch:,}")
        log(f"SQLite quick_check         = {quick_check}")
        log(f"elapsed                    = {elapsed / 60:.2f} min")
        log(f"output DB                  = {OUTDB}")
        log(f"selected scaffold CSV      = {SCAFFOLD_CSV}")
        log(f"report                     = {REPORT_FILE}")

    finally:
        conn.close()


def cli() -> None:
    """Run the diffusion-dataset materialization command."""

    try:
        main()
    except Exception as exc:
        print("\n" + "=" * 78, file=sys.stderr)
        print("FATAL ERROR", file=sys.stderr)
        print("=" * 78, file=sys.stderr)
        print(str(exc), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    cli()
