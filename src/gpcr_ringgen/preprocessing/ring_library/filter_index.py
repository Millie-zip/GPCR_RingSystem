#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v3：只建立“合格分子索引”，不复制完整分子/骨架大表。

用途
----
从 build_fused_ring_master_library_server.py 产生的 part_XXXX.db 中，
筛选满足以下“分子级”条件的记录：

1) 13 <= heavy_atom_count <= 35（包含 13 和 35）
2) 该分子至少存在一个连续稠环 ring system，其
   sssr_basis_ring_count >= 3

一旦分子满足条件：
- 这个分子整体视为“可用”
- 后续分析时保留它的全部 ring systems，包括其中的 2-ring system
- 本脚本此时不复制 molecules / ring_systems / scaffold_variants /
  attachment_sites 等大表
- 只记录 source_part + molecule_uid + 必要筛选摘要

这样下一步可以：
eligible_index.db + 原始120个part
    -> 只读取合格分子
    -> 做 scaffold rescue / 全局骨架去重 / 最终库构建
    -> 真正大规模复制只发生一次（写最终库）

Failure 规则
------------
A. 可恢复 scaffold failure：
   - build_pure_ring_scaffolds
   - build_expanded_scaffolds

   如果所属分子满足筛选条件，不会被删除。
   eligible_molecules 中会记录对应计数与标记，供下一步 rescue。

B. unresolved early failure：
   - missing_smiles
   - parse_smiles
   - molecule_facts
   - find_strict_fused_ring_systems
   - unexpected_worker_error

   这些记录单独写入 unresolved_failures.db。

C. no_strict_fused_system：
   这是“已经明确判断没有严格连续稠环”，不是 unresolved。

安全
----
- 原始 part 只读：本脚本不会 DELETE / UPDATE / INSERT 原 part
- 不创建原 part 索引
- 不 VACUUM 原 part
- 不复制 TB 级结构表
- 每个 part 独立扫描，临时 sidecar 最后汇总

典型运行
--------
nohup python -u filter_fused_ring_parts_index_v3.py \
  --input-dir merged_fused_ring_database/fused_ring_master_library/_parts_master_v1 \
  --output-dir fused_ring_filter_index_h13_35_3plus \
  --workers 8 \
  --overwrite \
  > filter_index_v3.log 2>&1 &

输出
----
fused_ring_filter_index_h13_35_3plus/
├── eligible_index.db
├── unresolved_failures.db
├── filter_summary.csv
└── filter_summary.json

eligible_index.db 中主要表：
- eligible_molecules
- part_summary
- filter_metadata

下一步只需要根据：
    source_part + molecule_uid
即可定点回原始 120 个 part 读取全部结构事实。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sqlite3
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FILTER_VERSION = "filter_index_h13_35_ge3_v3"

UNRESOLVED_STAGES = (
    "missing_smiles",
    "parse_smiles",
    "molecule_facts",
    "find_strict_fused_ring_systems",
    "unexpected_worker_error",
)

RECOVERABLE_STAGES = (
    "build_pure_ring_scaffolds",
    "build_expanded_scaffolds",
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def remove_sqlite_files(path: Path) -> None:
    for candidate in (path, Path(str(path) + "-wal"), Path(str(path) + "-shm")):
        if candidate.exists():
            candidate.unlink()


def quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def table_exists(conn: sqlite3.Connection, db_alias: str, table: str) -> bool:
    row = conn.execute(
        f"SELECT 1 FROM {db_alias}.sqlite_master "
        "WHERE type='table' AND name=? LIMIT 1",
        (table,),
    ).fetchone()
    return row is not None


def table_columns(conn: sqlite3.Connection, db_alias: str, table: str) -> set[str]:
    return {
        row[1]
        for row in conn.execute(
            f"PRAGMA {db_alias}.table_info({quote_identifier(table)})"
        )
    }


def validate_source_part(conn: sqlite3.Connection) -> None:
    required_tables = ("molecules", "ring_systems", "failures")
    missing_tables = [
        t for t in required_tables if not table_exists(conn, "src", t)
    ]
    if missing_tables:
        raise RuntimeError(
            "源分片缺少必要表：" + ", ".join(missing_tables)
        )

    mol_required = {
        "molecule_uid",
        "heavy_atom_count",
        "canonical_nonisomeric_smiles",
        "source_rowid",
    }
    rs_required = {
        "molecule_uid",
        "sssr_basis_ring_count",
    }
    fail_required = {
        "failure_uid",
        "molecule_uid",
        "failure_stage",
    }

    mol_cols = table_columns(conn, "src", "molecules")
    rs_cols = table_columns(conn, "src", "ring_systems")
    fail_cols = table_columns(conn, "src", "failures")

    if not mol_required.issubset(mol_cols):
        raise RuntimeError(
            "molecules 缺字段：" +
            ", ".join(sorted(mol_required - mol_cols))
        )
    if not rs_required.issubset(rs_cols):
        raise RuntimeError(
            "ring_systems 缺字段：" +
            ", ".join(sorted(rs_required - rs_cols))
        )
    if not fail_required.issubset(fail_cols):
        raise RuntimeError(
            "failures 缺字段：" +
            ", ".join(sorted(fail_required - fail_cols))
        )


def create_eligible_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS eligible_molecules (
            source_part TEXT NOT NULL,
            molecule_uid TEXT NOT NULL,
            source_rowid INTEGER,
            canonical_nonisomeric_smiles TEXT,
            heavy_atom_count INTEGER NOT NULL,

            ring_system_count INTEGER NOT NULL,
            min_ring_system_ring_count INTEGER,
            max_ring_system_ring_count INTEGER NOT NULL,
            ring_system_2_count INTEGER NOT NULL,
            ring_system_ge3_count INTEGER NOT NULL,

            recoverable_failure_count INTEGER NOT NULL,
            pure_scaffold_failure_count INTEGER NOT NULL,
            expanded_scaffold_failure_count INTEGER NOT NULL,
            has_recoverable_failure INTEGER NOT NULL,

            filter_version TEXT NOT NULL,
            created_at_utc TEXT NOT NULL,

            PRIMARY KEY (source_part, molecule_uid)
        );

        CREATE INDEX IF NOT EXISTS idx_eligible_part
            ON eligible_molecules(source_part);

        CREATE INDEX IF NOT EXISTS idx_eligible_uid
            ON eligible_molecules(molecule_uid);

        CREATE INDEX IF NOT EXISTS idx_eligible_heavy
            ON eligible_molecules(heavy_atom_count);

        CREATE INDEX IF NOT EXISTS idx_eligible_max_ring
            ON eligible_molecules(max_ring_system_ring_count);

        CREATE INDEX IF NOT EXISTS idx_eligible_recoverable
            ON eligible_molecules(has_recoverable_failure);

        CREATE TABLE IF NOT EXISTS part_summary (
            source_part TEXT PRIMARY KEY,
            qualifying_by_ring_count INTEGER NOT NULL,
            eligible_molecule_count INTEGER NOT NULL,
            eligible_with_recoverable_failure INTEGER NOT NULL,
            eligible_pure_failure_records INTEGER NOT NULL,
            eligible_expanded_failure_records INTEGER NOT NULL,
            unresolved_failure_records INTEGER NOT NULL,
            elapsed_seconds REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS filter_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        """
    )


def create_unresolved_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS unresolved_failures (
            source_part TEXT NOT NULL,
            failure_uid TEXT,
            run_id TEXT,
            source_rowid INTEGER,
            source_record_id TEXT,
            source_cid TEXT,
            smiles TEXT,
            molecule_uid TEXT,
            system_index INTEGER,
            failure_stage TEXT NOT NULL,
            error_type TEXT,
            error_message TEXT,
            created_at_utc TEXT,

            known_heavy_atom_count INTEGER,
            canonical_nonisomeric_smiles TEXT,

            filter_relevance TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_unresolved_part
            ON unresolved_failures(source_part);
        CREATE INDEX IF NOT EXISTS idx_unresolved_stage
            ON unresolved_failures(failure_stage);
        CREATE INDEX IF NOT EXISTS idx_unresolved_uid
            ON unresolved_failures(molecule_uid);
        CREATE INDEX IF NOT EXISTS idx_unresolved_rowid
            ON unresolved_failures(source_rowid);
        """
    )


def process_one_part(args: tuple[Any, ...]) -> dict[str, Any]:
    (
        part_path_str,
        eligible_sidecar_str,
        unresolved_sidecar_str,
        min_heavy,
        max_heavy,
        min_fused_rings,
    ) = args

    started = time.time()
    part_path = Path(part_path_str)
    eligible_sidecar = Path(eligible_sidecar_str)
    unresolved_sidecar = Path(unresolved_sidecar_str)

    eligible_sidecar.parent.mkdir(parents=True, exist_ok=True)
    unresolved_sidecar.parent.mkdir(parents=True, exist_ok=True)

    remove_sqlite_files(eligible_sidecar)
    remove_sqlite_files(unresolved_sidecar)

    stats: dict[str, Any] = {
        "source_part": part_path.name,
        "qualifying_by_ring_count": 0,
        "eligible_molecule_count": 0,
        "eligible_with_recoverable_failure": 0,
        "eligible_pure_failure_records": 0,
        "eligible_expanded_failure_records": 0,
        "unresolved_failure_records": 0,
        "elapsed_seconds": 0.0,
    }

    print(
        f"[{part_path.name}] STEP 1/5 打开并检查源分片 ...",
        flush=True,
    )

    # 每个 worker 只写自己的小 sidecar，避免并发抢一个输出 DB。
    with sqlite3.connect(eligible_sidecar) as out:
        out.execute("PRAGMA journal_mode=OFF")
        out.execute("PRAGMA synchronous=OFF")
        out.execute("PRAGMA temp_store=MEMORY")
        out.execute("PRAGMA cache_size=-131072")      # ~128 MiB
        out.execute("PRAGMA mmap_size=1073741824")    # 1 GiB
        create_eligible_schema(out)

        source_path = str(part_path.resolve())
        out.execute("ATTACH DATABASE ? AS src", (source_path,))
        validate_source_part(out)

        print(
            f"[{part_path.name}] STEP 2/5 顺序扫描 ring_systems，"
            f"找出至少一个 >= {min_fused_rings} 环 system 的分子 ...",
            flush=True,
        )

        # TEMP ring_summary 只存与筛选和后续索引有关的小摘要。
        out.execute(
            """
            CREATE TEMP TABLE ring_summary (
                molecule_uid TEXT PRIMARY KEY,
                ring_system_count INTEGER NOT NULL,
                min_ring_count INTEGER,
                max_ring_count INTEGER NOT NULL,
                ring2_count INTEGER NOT NULL,
                ring_ge3_count INTEGER NOT NULL
            ) WITHOUT ROWID
            """
        )

        # 一次聚合扫描 ring_systems。
        out.execute(
            """
            INSERT INTO ring_summary (
                molecule_uid,
                ring_system_count,
                min_ring_count,
                max_ring_count,
                ring2_count,
                ring_ge3_count
            )
            SELECT
                molecule_uid,
                COUNT(*) AS ring_system_count,
                MIN(sssr_basis_ring_count) AS min_ring_count,
                MAX(sssr_basis_ring_count) AS max_ring_count,
                SUM(CASE WHEN sssr_basis_ring_count = 2 THEN 1 ELSE 0 END),
                SUM(CASE WHEN sssr_basis_ring_count >= ? THEN 1 ELSE 0 END)
            FROM src.ring_systems
            GROUP BY molecule_uid
            HAVING MAX(sssr_basis_ring_count) >= ?
            """,
            (min_fused_rings, min_fused_rings),
        )

        stats["qualifying_by_ring_count"] = out.execute(
            "SELECT COUNT(*) FROM ring_summary"
        ).fetchone()[0]

        print(
            f"[{part_path.name}] STEP 2/5 完成："
            f"{stats['qualifying_by_ring_count']:,} 个分子满足环数条件",
            flush=True,
        )

        print(
            f"[{part_path.name}] STEP 3/5 检查重原子 "
            f"{min_heavy}-{max_heavy}，并写小型 eligible 索引 ...",
            flush=True,
        )

        # recoverable failure 也只做一次聚合摘要。
        out.execute(
            """
            CREATE TEMP TABLE recoverable_summary (
                molecule_uid TEXT PRIMARY KEY,
                recoverable_failure_count INTEGER NOT NULL,
                pure_failure_count INTEGER NOT NULL,
                expanded_failure_count INTEGER NOT NULL
            ) WITHOUT ROWID
            """
        )

        out.execute(
            """
            INSERT INTO recoverable_summary (
                molecule_uid,
                recoverable_failure_count,
                pure_failure_count,
                expanded_failure_count
            )
            SELECT
                molecule_uid,
                COUNT(*) AS recoverable_failure_count,
                SUM(CASE
                    WHEN failure_stage='build_pure_ring_scaffolds'
                    THEN 1 ELSE 0 END),
                SUM(CASE
                    WHEN failure_stage='build_expanded_scaffolds'
                    THEN 1 ELSE 0 END)
            FROM src.failures
            WHERE failure_stage IN (
                'build_pure_ring_scaffolds',
                'build_expanded_scaffolds'
            )
              AND molecule_uid IS NOT NULL
            GROUP BY molecule_uid
            """
        )

        # 这里最终只写每个合格分子“一行索引”，不复制结构事实大字段。
        out.execute(
            """
            INSERT INTO eligible_molecules (
                source_part,
                molecule_uid,
                source_rowid,
                canonical_nonisomeric_smiles,
                heavy_atom_count,
                ring_system_count,
                min_ring_system_ring_count,
                max_ring_system_ring_count,
                ring_system_2_count,
                ring_system_ge3_count,
                recoverable_failure_count,
                pure_scaffold_failure_count,
                expanded_scaffold_failure_count,
                has_recoverable_failure,
                filter_version,
                created_at_utc
            )
            SELECT
                ? AS source_part,
                m.molecule_uid,
                m.source_rowid,
                m.canonical_nonisomeric_smiles,
                m.heavy_atom_count,
                r.ring_system_count,
                r.min_ring_count,
                r.max_ring_count,
                r.ring2_count,
                r.ring_ge3_count,
                COALESCE(f.recoverable_failure_count, 0),
                COALESCE(f.pure_failure_count, 0),
                COALESCE(f.expanded_failure_count, 0),
                CASE
                    WHEN COALESCE(f.recoverable_failure_count, 0) > 0
                    THEN 1 ELSE 0
                END,
                ?,
                ?
            FROM ring_summary AS r
            JOIN src.molecules AS m
              ON m.molecule_uid = r.molecule_uid
            LEFT JOIN recoverable_summary AS f
              ON f.molecule_uid = r.molecule_uid
            WHERE m.heavy_atom_count BETWEEN ? AND ?
            """,
            (
                part_path.name,
                FILTER_VERSION,
                utc_now_iso(),
                min_heavy,
                max_heavy,
            ),
        )

        stats["eligible_molecule_count"] = out.execute(
            "SELECT COUNT(*) FROM eligible_molecules"
        ).fetchone()[0]
        stats["eligible_with_recoverable_failure"] = out.execute(
            """
            SELECT COUNT(*)
            FROM eligible_molecules
            WHERE has_recoverable_failure=1
            """
        ).fetchone()[0]
        stats["eligible_pure_failure_records"] = out.execute(
            """
            SELECT COALESCE(SUM(pure_scaffold_failure_count), 0)
            FROM eligible_molecules
            """
        ).fetchone()[0]
        stats["eligible_expanded_failure_records"] = out.execute(
            """
            SELECT COALESCE(SUM(expanded_scaffold_failure_count), 0)
            FROM eligible_molecules
            """
        ).fetchone()[0]

        out.commit()
        out.execute("DETACH DATABASE src")

    print(
        f"[{part_path.name}] STEP 3/5 完成："
        f"{stats['eligible_molecule_count']:,} 个合格分子，"
        f"其中 {stats['eligible_with_recoverable_failure']:,} 个带可恢复 failure",
        flush=True,
    )

    print(
        f"[{part_path.name}] STEP 4/5 汇总 unresolved early failures ...",
        flush=True,
    )

    with sqlite3.connect(unresolved_sidecar) as u:
        u.execute("PRAGMA journal_mode=OFF")
        u.execute("PRAGMA synchronous=OFF")
        u.execute("PRAGMA temp_store=MEMORY")
        u.execute("PRAGMA cache_size=-65536")
        create_unresolved_schema(u)

        source_path = str(part_path.resolve())
        u.execute("ATTACH DATABASE ? AS src", (source_path,))

        placeholders = ",".join("?" for _ in UNRESOLVED_STAGES)
        u.execute(
            f"""
            INSERT INTO unresolved_failures (
                source_part,
                failure_uid,
                run_id,
                source_rowid,
                source_record_id,
                source_cid,
                smiles,
                molecule_uid,
                system_index,
                failure_stage,
                error_type,
                error_message,
                created_at_utc,
                known_heavy_atom_count,
                canonical_nonisomeric_smiles,
                filter_relevance
            )
            SELECT
                ?,
                f.failure_uid,
                f.run_id,
                f.source_rowid,
                f.source_record_id,
                f.source_cid,
                f.smiles,
                f.molecule_uid,
                f.system_index,
                f.failure_stage,
                f.error_type,
                f.error_message,
                f.created_at_utc,
                m.heavy_atom_count,
                m.canonical_nonisomeric_smiles,
                CASE
                    WHEN m.heavy_atom_count IS NULL
                        THEN 'target_status_unknown'
                    WHEN m.heavy_atom_count < ?
                        THEN 'heavy_atom_below_range_but_extraction_failed'
                    WHEN m.heavy_atom_count > ?
                        THEN 'heavy_atom_above_range_but_extraction_failed'
                    ELSE 'heavy_atom_in_range_but_ring_eligibility_unresolved'
                END
            FROM src.failures AS f
            LEFT JOIN src.molecules AS m
              ON m.molecule_uid = f.molecule_uid
            WHERE f.failure_stage IN ({placeholders})
            """,
            (
                part_path.name,
                min_heavy,
                max_heavy,
                *UNRESOLVED_STAGES,
            ),
        )

        stats["unresolved_failure_records"] = u.execute(
            "SELECT COUNT(*) FROM unresolved_failures"
        ).fetchone()[0]

        u.commit()
        u.execute("DETACH DATABASE src")

    stats["elapsed_seconds"] = time.time() - started

    # 写 sidecar 自己的 part_summary。
    with sqlite3.connect(eligible_sidecar) as out:
        out.execute(
            """
            INSERT OR REPLACE INTO part_summary VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                stats["source_part"],
                stats["qualifying_by_ring_count"],
                stats["eligible_molecule_count"],
                stats["eligible_with_recoverable_failure"],
                stats["eligible_pure_failure_records"],
                stats["eligible_expanded_failure_records"],
                stats["unresolved_failure_records"],
                stats["elapsed_seconds"],
            ),
        )
        out.commit()

    print(
        f"[{part_path.name}] STEP 5/5 DONE | "
        f"eligible={stats['eligible_molecule_count']:,} | "
        f"unresolved={stats['unresolved_failure_records']:,} | "
        f"{stats['elapsed_seconds']/60:.2f} min",
        flush=True,
    )

    return stats


def merge_eligible_sidecars(
    sidecars: list[Path],
    final_db: Path,
    metadata: dict[str, Any],
) -> None:
    remove_sqlite_files(final_db)

    with sqlite3.connect(final_db) as out:
        out.execute("PRAGMA journal_mode=OFF")
        out.execute("PRAGMA synchronous=OFF")
        out.execute("PRAGMA temp_store=MEMORY")
        create_eligible_schema(out)

        for i, sidecar in enumerate(sidecars, start=1):
            alias = f"p{i}"
            out.execute(f"ATTACH DATABASE ? AS {alias}", (str(sidecar),))

            out.execute(
                f"""
                INSERT OR REPLACE INTO eligible_molecules
                SELECT * FROM {alias}.eligible_molecules
                """
            )
            out.execute(
                f"""
                INSERT OR REPLACE INTO part_summary
                SELECT * FROM {alias}.part_summary
                """
            )

            out.commit()
            out.execute(f"DETACH DATABASE {alias}")

        out.executemany(
            """
            INSERT OR REPLACE INTO filter_metadata(key, value)
            VALUES (?, ?)
            """,
            [(k, str(v)) for k, v in metadata.items()],
        )

        # 后续分析常用视图。
        out.executescript(
            """
            DROP VIEW IF EXISTS eligible_by_part;
            CREATE VIEW eligible_by_part AS
            SELECT
                source_part,
                COUNT(*) AS eligible_molecule_count,
                SUM(has_recoverable_failure) AS molecules_with_recoverable_failure,
                SUM(ring_system_count) AS total_ring_system_count,
                SUM(ring_system_2_count) AS total_2ring_system_count,
                SUM(ring_system_ge3_count) AS total_ge3ring_system_count
            FROM eligible_molecules
            GROUP BY source_part;

            DROP VIEW IF EXISTS eligible_global_summary;
            CREATE VIEW eligible_global_summary AS
            SELECT
                COUNT(*) AS eligible_molecule_count,
                MIN(heavy_atom_count) AS min_heavy_atom_count,
                MAX(heavy_atom_count) AS max_heavy_atom_count,
                SUM(ring_system_count) AS total_ring_system_count,
                SUM(ring_system_2_count) AS total_2ring_system_count,
                SUM(ring_system_ge3_count) AS total_ge3ring_system_count,
                SUM(has_recoverable_failure) AS molecules_with_recoverable_failure,
                SUM(pure_scaffold_failure_count) AS pure_scaffold_failure_records,
                SUM(expanded_scaffold_failure_count) AS expanded_scaffold_failure_records
            FROM eligible_molecules;
            """
        )
        out.commit()


def merge_unresolved_sidecars(
    sidecars: list[Path],
    final_db: Path,
) -> None:
    remove_sqlite_files(final_db)

    with sqlite3.connect(final_db) as out:
        out.execute("PRAGMA journal_mode=OFF")
        out.execute("PRAGMA synchronous=OFF")
        out.execute("PRAGMA temp_store=MEMORY")
        create_unresolved_schema(out)

        for i, sidecar in enumerate(sidecars, start=1):
            alias = f"u{i}"
            out.execute(f"ATTACH DATABASE ? AS {alias}", (str(sidecar),))
            out.execute(
                f"""
                INSERT INTO unresolved_failures
                SELECT * FROM {alias}.unresolved_failures
                """
            )
            out.commit()
            out.execute(f"DETACH DATABASE {alias}")

        out.execute(
            """
            CREATE TABLE unresolved_summary AS
            SELECT
                failure_stage,
                filter_relevance,
                COUNT(*) AS record_count
            FROM unresolved_failures
            GROUP BY failure_stage, filter_relevance
            ORDER BY failure_stage, filter_relevance
            """
        )
        out.commit()


def write_summary_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    if not rows:
        return

    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

        total: dict[str, Any] = {"source_part": "TOTAL"}
        for key in fields:
            if key == "source_part":
                continue
            values = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
            if values:
                total[key] = sum(values)
        writer.writerow(total)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "只生成合格分子索引，不复制大表："
            "13-35 重原子 + 至少一个 >=3-ring strict fused system。"
        )
    )

    p.add_argument(
        "--input-dir",
        type=Path,
        default=Path(
            "merged_fused_ring_database/"
            "fused_ring_master_library/_parts_master_v1"
        ),
        help="原始 part_XXXX.db 目录。",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("fused_ring_filter_index_h13_35_3plus"),
        help="输出索引目录。",
    )
    p.add_argument(
        "--glob",
        default="part_*.db",
        help="输入分片匹配模式，默认 part_*.db。",
    )
    p.add_argument(
        "--min-heavy",
        type=int,
        default=13,
        help="最小重原子数（包含），默认 13。",
    )
    p.add_argument(
        "--max-heavy",
        type=int,
        default=35,
        help="最大重原子数（包含），默认 35。",
    )
    p.add_argument(
        "--min-fused-rings",
        type=int,
        default=3,
        help="分子必须至少有一个此环数以上的 ring system，默认 3。",
    )

    default_workers = min(8, max(1, (os.cpu_count() or 8) - 2))
    p.add_argument(
        "--workers",
        type=int,
        default=default_workers,
        help=(
            f"并行 part 数，默认 {default_workers}。"
            "共享存储下不建议盲目开很大。"
        ),
    )

    p.add_argument(
        "--overwrite",
        action="store_true",
        help="允许删除并重建已有输出目录。",
    )
    p.add_argument(
        "--keep-sidecars",
        action="store_true",
        help="保留每个 part 的临时索引 sidecar，默认最终合并后删除。",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.min_heavy < 1:
        raise ValueError("--min-heavy 必须 >= 1")
    if args.max_heavy < args.min_heavy:
        raise ValueError("--max-heavy 必须 >= --min-heavy")
    if args.min_fused_rings < 2:
        raise ValueError("--min-fused-rings 必须 >= 2")
    if args.workers < 1:
        raise ValueError("--workers 必须 >= 1")

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    eligible_tmp_dir = output_dir / "_eligible_sidecars_tmp"
    unresolved_tmp_dir = output_dir / "_unresolved_sidecars_tmp"

    eligible_final = output_dir / "eligible_index.db"
    unresolved_final = output_dir / "unresolved_failures.db"
    summary_csv = output_dir / "filter_summary.csv"
    summary_json = output_dir / "filter_summary.json"

    if not input_dir.exists():
        raise FileNotFoundError(f"找不到输入目录：{input_dir}")

    part_paths = sorted(input_dir.glob(args.glob))
    if not part_paths:
        raise FileNotFoundError(
            f"没有找到匹配 {args.glob} 的 part：{input_dir}"
        )

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"输出目录已存在：{output_dir}\n"
                "若确认重建，请增加 --overwrite。"
            )
        shutil.rmtree(output_dir)

    eligible_tmp_dir.mkdir(parents=True, exist_ok=True)
    unresolved_tmp_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 96, flush=True)
    print("v3 合格分子索引筛选开始", flush=True)
    print("filter_version:", FILTER_VERSION, flush=True)
    print("输入目录:", input_dir, flush=True)
    print("分片数量:", len(part_paths), flush=True)
    print(
        "筛选条件:",
        f"{args.min_heavy} <= heavy_atom_count <= {args.max_heavy}",
        "AND",
        f"至少一个 sssr_basis_ring_count >= {args.min_fused_rings}",
        flush=True,
    )
    print(
        "保留规则:",
        "分子通过后，后续分析保留它的全部 ring systems，包括 2-ring",
        flush=True,
    )
    print(
        "本步骤:",
        "只写小型索引，不复制 molecules/ring_systems/scaffold/attachment 大表",
        flush=True,
    )
    print("workers:", args.workers, flush=True)
    print("GPU:", "不使用；此步骤主要是 SQLite 顺序扫描 + 聚合", flush=True)
    print("输出目录:", output_dir, flush=True)
    print("=" * 96, flush=True)

    jobs: list[tuple[Any, ...]] = []
    eligible_sidecars: list[Path] = []
    unresolved_sidecars: list[Path] = []

    for part in part_paths:
        e_sidecar = eligible_tmp_dir / f"{part.stem}_eligible.db"
        u_sidecar = unresolved_tmp_dir / f"{part.stem}_unresolved.db"

        eligible_sidecars.append(e_sidecar)
        unresolved_sidecars.append(u_sidecar)

        jobs.append(
            (
                str(part),
                str(e_sidecar),
                str(u_sidecar),
                args.min_heavy,
                args.max_heavy,
                args.min_fused_rings,
            )
        )

    started = time.time()
    results: list[dict[str, Any]] = []

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        future_to_part = {
            executor.submit(process_one_part, job): Path(job[0]).name
            for job in jobs
        }

        for completed, future in enumerate(
            as_completed(future_to_part), start=1
        ):
            part_name = future_to_part[future]

            try:
                stats = future.result()
            except Exception as exc:
                raise RuntimeError(
                    f"处理分片失败：{part_name}: {exc}"
                ) from exc

            results.append(stats)

            elapsed = max(1e-9, time.time() - started)
            part_rate = completed / elapsed
            remaining = len(jobs) - completed
            eta_min = (
                remaining / part_rate / 60
                if part_rate > 0 else float("inf")
            )

            print(
                "[PROGRESS] "
                f"{completed}/{len(jobs)} | "
                f"{part_name} | "
                f"eligible={stats['eligible_molecule_count']:,} | "
                f"unresolved={stats['unresolved_failure_records']:,} | "
                f"part_time={stats['elapsed_seconds']/60:.2f} min | "
                f"ETA={eta_min:.1f} min",
                flush=True,
            )

    results.sort(key=lambda r: r["source_part"])

    print("[MERGE] 合并 eligible sidecars ...", flush=True)

    metadata = {
        "filter_version": FILTER_VERSION,
        "created_at_utc": utc_now_iso(),
        "input_dir": str(input_dir),
        "part_count": len(part_paths),
        "min_heavy_atom_count_inclusive": args.min_heavy,
        "max_heavy_atom_count_inclusive": args.max_heavy,
        "min_fused_ring_count": args.min_fused_rings,
        "molecule_level_policy":
            "a molecule is eligible if it has >=1 qualifying ring system; "
            "all ring systems of an eligible molecule remain eligible for downstream analysis",
        "large_table_copy_policy":
            "do not copy source structural tables at filtering stage",
        "source_parts_policy":
            "read-only; unchanged",
        "recoverable_failure_stages":
            json.dumps(RECOVERABLE_STAGES, ensure_ascii=False),
        "unresolved_failure_stages":
            json.dumps(UNRESOLVED_STAGES, ensure_ascii=False),
    }

    merge_eligible_sidecars(
        eligible_sidecars,
        eligible_final,
        metadata,
    )

    print("[MERGE] 合并 unresolved failure sidecars ...", flush=True)
    merge_unresolved_sidecars(
        unresolved_sidecars,
        unresolved_final,
    )

    write_summary_csv(summary_csv, results)

    totals = {
        "qualifying_by_ring_count":
            sum(int(r["qualifying_by_ring_count"]) for r in results),
        "eligible_molecule_count":
            sum(int(r["eligible_molecule_count"]) for r in results),
        "eligible_with_recoverable_failure":
            sum(int(r["eligible_with_recoverable_failure"]) for r in results),
        "eligible_pure_failure_records":
            sum(int(r["eligible_pure_failure_records"]) for r in results),
        "eligible_expanded_failure_records":
            sum(int(r["eligible_expanded_failure_records"]) for r in results),
        "unresolved_failure_records":
            sum(int(r["unresolved_failure_records"]) for r in results),
    }

    total_elapsed = time.time() - started

    report = {
        "filter_version": FILTER_VERSION,
        "created_at_utc": utc_now_iso(),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "part_count": len(part_paths),
        "workers": args.workers,
        "criteria": {
            "min_heavy_atom_count_inclusive": args.min_heavy,
            "max_heavy_atom_count_inclusive": args.max_heavy,
            "must_have_at_least_one_ring_system_with_sssr_basis_ring_count_ge":
                args.min_fused_rings,
            "retain_all_ring_systems_for_eligible_molecule_downstream": True,
        },
        "outputs": {
            "eligible_index_db": str(eligible_final),
            "unresolved_failures_db": str(unresolved_final),
            "summary_csv": str(summary_csv),
        },
        "totals": totals,
        "elapsed_minutes": total_elapsed / 60,
        "notes": [
            "原始 120 个 part 只读，不修改。",
            "本步骤不复制 TB 级 molecules/ring_systems/scaffold/attachment 大表。",
            "eligible_index.db 中 source_part + molecule_uid 是下一步回源读取的定位键。",
            "build_pure_ring_scaffolds/build_expanded_scaffolds 属于可恢复 failure。",
            "真正无法可靠判定的 early failures 单独进入 unresolved_failures.db。",
            "no_strict_fused_system 不属于 unresolved。",
        ],
    }

    summary_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if not args.keep_sidecars:
        shutil.rmtree(eligible_tmp_dir, ignore_errors=True)
        shutil.rmtree(unresolved_tmp_dir, ignore_errors=True)

    print("=" * 96, flush=True)
    print("v3 合格分子索引筛选完成", flush=True)
    print("总耗时:", f"{total_elapsed/60:.2f} 分钟", flush=True)
    print(
        "至少有一个 >=3-ring 的分子:",
        f"{totals['qualifying_by_ring_count']:,}",
        flush=True,
    )
    print(
        "最终 13-35 重原子合格分子:",
        f"{totals['eligible_molecule_count']:,}",
        flush=True,
    )
    print(
        "其中带可恢复 scaffold failure 的分子:",
        f"{totals['eligible_with_recoverable_failure']:,}",
        flush=True,
    )
    print(
        "pure scaffold failure 记录:",
        f"{totals['eligible_pure_failure_records']:,}",
        flush=True,
    )
    print(
        "expanded scaffold failure 记录:",
        f"{totals['eligible_expanded_failure_records']:,}",
        flush=True,
    )
    print(
        "unresolved early failure 记录:",
        f"{totals['unresolved_failure_records']:,}",
        flush=True,
    )
    print("合格分子索引:", eligible_final, flush=True)
    print("无法判定失败库:", unresolved_final, flush=True)
    print("统计 CSV:", summary_csv, flush=True)
    print("统计 JSON:", summary_json, flush=True)
    print("=" * 96, flush=True)


if __name__ == "__main__":
    main()
