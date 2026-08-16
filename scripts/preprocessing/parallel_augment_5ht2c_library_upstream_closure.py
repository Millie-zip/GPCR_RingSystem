#!/usr/bin/env python3
"""Parallel fast upstream-closure augmentation for the full 5-HT2C library.

Phase A extracts independent source parts concurrently into disposable staging
SQLite files. Phase B bulk-merges them through one writer into the original
library. This avoids SQLite multi-writer contention while utilizing server I/O.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

import augment_5ht2c_library_upstream_closure as core


def staging_paths(staging_dir: Path, source_part: str) -> tuple[Path, Path]:
    digest = hashlib.sha1(source_part.encode("utf-8")).hexdigest()[:12]
    stem = Path(source_part).name.replace(".db", "") or "part"
    db = staging_dir / f"{stem}_{digest}.stage.db"
    return db, Path(str(db) + ".complete.json")


def remove_exact_staging_files(db: Path, marker: Path) -> None:
    for path in (db, Path(str(db) + "-wal"), Path(str(db) + "-shm"), marker):
        if path.exists():
            path.unlink()


def ro_connect(path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(core.readonly_uri(path), uri=True, timeout=120)
    con.execute("PRAGMA query_only=ON")
    con.execute("PRAGMA busy_timeout=120000")
    return con


def extraction_worker(job: dict) -> dict:
    source_part = job["source_part"]
    source_path = Path(job["source_path"])
    library_db = Path(job["library_db"])
    staging_db = Path(job["staging_db"])
    marker = Path(job["marker"])
    resume = bool(job["resume"])
    started = time.time()
    prefix = f"[extract pid={os.getpid()} {source_part}]"

    if resume and marker.is_file() and staging_db.is_file():
        data = json.loads(marker.read_text(encoding="utf-8"))
        data["reused"] = True
        print(f"{prefix} reuse completed staging file", flush=True)
        return data

    remove_exact_staging_files(staging_db, marker)
    staging_db.parent.mkdir(parents=True, exist_ok=True)
    library = ro_connect(library_db)
    # uri=True sets SQLITE_OPEN_URI on the connection, which is also required
    # for read-only file: URIs used later by ATTACH DATABASE.
    out = sqlite3.connect(staging_db, timeout=120, uri=True)
    try:
        out.execute("PRAGMA journal_mode=OFF")
        out.execute("PRAGMA synchronous=OFF")
        out.execute("PRAGMA temp_store=MEMORY")
        out.execute("PRAGMA cache_size=-524288")
        core.install_schema(out, create_secondary_indexes=False)
        out.execute("DROP TABLE IF EXISTS temp.wanted_ring_systems")
        out.execute(
            "CREATE TEMP TABLE wanted_ring_systems(ring_system_uid TEXT PRIMARY KEY) WITHOUT ROWID"
        )

        print(f"{prefix} loading wanted ring-system UIDs", flush=True)
        cursor = library.execute(
            """
            SELECT DISTINCT ring_system_uid
            FROM selected_scaffold_instances
            WHERE source_part=? AND ring_system_uid IS NOT NULL
            """,
            (source_part,),
        )
        wanted = 0
        while True:
            rows = cursor.fetchmany(50_000)
            if not rows:
                break
            out.executemany("INSERT OR IGNORE INTO wanted_ring_systems VALUES (?)", rows)
            wanted += len(rows)
        wanted_unique = out.execute("SELECT COUNT(*) FROM wanted_ring_systems").fetchone()[0]
        print(f"{prefix} wanted unique={wanted_unique:,}", flush=True)

        # executemany() above opens an implicit Python sqlite3 transaction.
        # Commit the temporary wanted-key set before starting the explicit
        # atomic extraction transaction below.
        out.commit()

        out.execute("ATTACH DATABASE ? AS src", (core.readonly_uri(source_path),))
        core.validate_source(out)
        stage = ["ring_systems"]
        last = [time.time()]

        def progress():
            now = time.time()
            if now - last[0] >= 60:
                print(
                    f"{prefix} still extracting {stage[0]}; elapsed={(now-started)/60:.1f} min",
                    flush=True,
                )
                last[0] = now
            return 0

        out.set_progress_handler(progress, 1_000_000)
        out.execute("BEGIN")
        stage[0] = "ring_systems"
        out.execute(
            f"""
            INSERT INTO selected_ring_systems
            (source_part,{','.join(core.qident(c) for c in core.RING_COLUMNS)})
            SELECT ?,{','.join('s.' + core.qident(c) for c in core.RING_COLUMNS)}
            FROM src.ring_systems s
            JOIN wanted_ring_systems w ON w.ring_system_uid=s.ring_system_uid
            """,
            (source_part,),
        )
        rings = out.execute("SELECT COUNT(*) FROM selected_ring_systems").fetchone()[0]
        print(f"{prefix} ring_systems={rings:,}", flush=True)

        stage[0] = "scaffold_variants/atom mappings"
        out.execute(
            f"""
            INSERT INTO selected_scaffold_variants
            (source_part,{','.join(core.qident(c) for c in core.VARIANT_COLUMNS)})
            SELECT ?,{','.join('s.' + core.qident(c) for c in core.VARIANT_COLUMNS)}
            FROM src.scaffold_variants s
            JOIN wanted_ring_systems w ON w.ring_system_uid=s.ring_system_uid
            """,
            (source_part,),
        )
        variants = out.execute("SELECT COUNT(*) FROM selected_scaffold_variants").fetchone()[0]
        print(f"{prefix} scaffold_variants={variants:,}", flush=True)

        stage[0] = "attachment_sites"
        out.execute(
            f"""
            INSERT INTO selected_attachment_sites
            (source_part,{','.join(core.qident(c) for c in core.ATTACHMENT_COLUMNS)})
            SELECT ?,{','.join('s.' + core.qident(c) for c in core.ATTACHMENT_COLUMNS)}
            FROM src.attachment_sites s
            JOIN wanted_ring_systems w ON w.ring_system_uid=s.ring_system_uid
            """,
            (source_part,),
        )
        attachments = out.execute("SELECT COUNT(*) FROM selected_attachment_sites").fetchone()[0]
        out.commit()
        out.set_progress_handler(None, 0)
        out.execute("DETACH DATABASE src")
        elapsed = time.time() - started
        manifest = {
            "source_part": source_part,
            "source_path": str(source_path),
            "staging_db": str(staging_db),
            "wanted_ring_systems": wanted_unique,
            "ring_systems": rings,
            "scaffold_variants": variants,
            "attachment_sites": attachments,
            "elapsed_seconds": elapsed,
            "size_bytes": staging_db.stat().st_size,
            "completed_at_utc": core.now_utc(),
            "reused": False,
        }
        temp_marker = Path(str(marker) + ".tmp")
        temp_marker.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        temp_marker.replace(marker)
        print(
            f"{prefix} COMPLETE elapsed={elapsed/60:.2f}min size={manifest['size_bytes']/2**30:.2f}GiB",
            flush=True,
        )
        return manifest
    except BaseException:
        out.rollback()
        raise
    finally:
        out.close()
        library.close()


def completed_main_parts(library_db: Path) -> set[str]:
    con = sqlite3.connect(library_db, timeout=120, uri=True)
    try:
        if not core.table_exists(con, "upstream_enrichment_parts"):
            return set()
        return {
            str(row[0])
            for row in con.execute(
                "SELECT source_part FROM upstream_enrichment_parts WHERE status='COMPLETE'"
            )
        }
    finally:
        con.close()


def merge_staging(library_db: Path, global_db: Path, manifests: list[dict], full_run: bool) -> None:
    # Required so attached staging/global file: URIs honor mode=ro.
    con = sqlite3.connect(library_db, timeout=120, uri=True)
    con.execute("PRAGMA busy_timeout=120000")
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute("PRAGMA cache_size=-1048576")
    try:
        core.install_schema(con, create_secondary_indexes=False)
        core.validate_destination_schema(con)
        core.copy_topologies(con, global_db)
        already = completed_main_parts(library_db)
        pending = [m for m in manifests if m["source_part"] not in already]
        print(f"[merge] staging files={len(manifests)}, pending={len(pending)}", flush=True)
        for index, manifest in enumerate(pending, start=1):
            part = manifest["source_part"]
            stage_path = Path(manifest["staging_db"])
            print(f"[merge {index}/{len(pending)}] {part}", flush=True)
            con.execute("ATTACH DATABASE ? AS stage", (core.readonly_uri(stage_path) + "&immutable=1",))
            try:
                con.execute("BEGIN IMMEDIATE")
                before = con.total_changes
                ring_cols = ["source_part", *core.RING_COLUMNS]
                con.execute(
                    f"INSERT OR IGNORE INTO selected_ring_systems ({','.join(map(core.qident, ring_cols))}) "
                    f"SELECT {','.join(map(core.qident, ring_cols))} FROM stage.selected_ring_systems"
                )
                rings = con.total_changes - before
                before = con.total_changes
                variant_cols = ["source_part", *core.VARIANT_COLUMNS]
                con.execute(
                    f"INSERT OR IGNORE INTO selected_scaffold_variants "
                    f"({','.join(map(core.qident, variant_cols))}) SELECT "
                    f"{','.join(map(core.qident, variant_cols))} FROM stage.selected_scaffold_variants"
                )
                variants = con.total_changes - before
                before = con.total_changes
                attachment_cols = ["source_part", *core.ATTACHMENT_COLUMNS]
                con.execute(
                    f"INSERT OR IGNORE INTO selected_attachment_sites "
                    f"({','.join(map(core.qident, attachment_cols))}) SELECT "
                    f"{','.join(map(core.qident, attachment_cols))} FROM stage.selected_attachment_sites"
                )
                attachments = con.total_changes - before
                con.execute(
                    """
                    INSERT OR REPLACE INTO upstream_enrichment_parts
                    (source_part,source_path,status,wanted_ring_systems,copied_ring_systems,
                     copied_scaffold_variants,copied_attachment_sites,started_at_utc,
                     completed_at_utc,elapsed_seconds,error_text)
                    VALUES (?,?,?,?,?,?,?,?,?,?,NULL)
                    """,
                    (
                        part, manifest["source_path"], "COMPLETE",
                        manifest["wanted_ring_systems"], rings, variants, attachments,
                        manifest["completed_at_utc"], core.now_utc(),
                        manifest["elapsed_seconds"],
                    ),
                )
                con.commit()
                print(
                    f"[merge] inserted rings={rings:,} variants={variants:,} attachments={attachments:,}",
                    flush=True,
                )
            except BaseException:
                con.rollback()
                raise
            finally:
                con.execute("DETACH DATABASE stage")
            if index % 4 == 0:
                con.execute("PRAGMA wal_checkpoint(PASSIVE)")

        if full_run:
            print("[merge] building secondary indexes (final bulk step)", flush=True)
            started = time.time()
            last = [time.time()]

            def progress():
                now = time.time()
                if now - last[0] >= 60:
                    print(f"[indexes] still running; elapsed={(now-started)/60:.1f}min", flush=True)
                    last[0] = now
                return 0

            con.set_progress_handler(progress, 1_000_000)
            core.install_secondary_indexes(con)
            con.commit()
            con.set_progress_handler(None, 0)
            core.final_audit(con)
            con.execute("PRAGMA wal_checkpoint(PASSIVE)")
    finally:
        con.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--library-db", required=True, type=Path)
    p.add_argument("--global-db", required=True, type=Path)
    p.add_argument("--parts-dir", required=True, type=Path)
    p.add_argument("--staging-dir", required=True, type=Path)
    p.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    p.add_argument("--max-parts", type=int, help="Benchmark/test only: process first N pending parts")
    p.add_argument("--extract-only", action="store_true")
    p.add_argument("--merge-only", action="store_true")
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.extract_only and args.merge_only:
        raise ValueError("--extract-only and --merge-only are mutually exclusive")
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    for path in (args.library_db, args.global_db, args.parts_dir):
        if not path.exists():
            raise FileNotFoundError(path)
    args.staging_dir.mkdir(parents=True, exist_ok=True)

    lib = ro_connect(args.library_db)
    try:
        parts = [
            str(row[0])
            for row in lib.execute(
                "SELECT DISTINCT source_part FROM selected_scaffold_instances "
                "WHERE source_part IS NOT NULL ORDER BY source_part"
            )
        ]
    finally:
        lib.close()
    main_complete = completed_main_parts(args.library_db) if args.resume else set()
    pending = [p for p in parts if p not in main_complete]
    if args.max_parts is not None:
        pending = pending[: args.max_parts]
    jobs = []
    for part in pending:
        source = core.source_part_path(part, args.parts_dir)
        staging_db, marker = staging_paths(args.staging_dir, part)
        jobs.append(
            {
                "source_part": part,
                "source_path": str(source),
                "library_db": str(args.library_db),
                "staging_db": str(staging_db),
                "marker": str(marker),
                "resume": args.resume,
            }
        )
    print(
        f"full selected parts={len(parts)}; main complete={len(main_complete)}; "
        f"this run={len(jobs)}; workers={args.workers}",
        flush=True,
    )

    manifests = []
    wall_started = time.time()
    if not args.merge_only:
        with cf.ProcessPoolExecutor(max_workers=args.workers) as pool:
            future_map = {pool.submit(extraction_worker, job): job for job in jobs}
            for future in cf.as_completed(future_map):
                job = future_map[future]
                try:
                    manifests.append(future.result())
                except BaseException as exc:
                    print(f"FAILED {job['source_part']}: {exc!r}", file=sys.stderr, flush=True)
                    for other in future_map:
                        other.cancel()
                    raise
        wall = time.time() - wall_started
        if manifests:
            rate = len(manifests) / wall
            remaining = max(0, len(parts) - len(main_complete) - len(manifests))
            print(
                f"extraction wall time={wall/60:.2f}min; throughput={rate*3600:.2f} parts/hour; "
                f"rough remaining extraction ETA={remaining/rate/3600:.2f}h",
                flush=True,
            )
    else:
        for job in jobs:
            marker = Path(job["marker"])
            if not marker.is_file():
                raise FileNotFoundError(f"Missing completed staging marker: {marker}")
            manifests.append(json.loads(marker.read_text(encoding="utf-8")))

    if not args.extract_only:
        full_run = args.max_parts is None
        merge_staging(args.library_db, args.global_db, manifests, full_run)
    print("parallel augmentation command completed", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted; rerun with --resume.", file=sys.stderr, flush=True)
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise
