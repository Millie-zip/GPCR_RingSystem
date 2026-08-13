#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 build_fused_ring_master_library_server.py 产生的 120 个 part_XXXX.db 中，
筛选“可用分子”，同时单独汇总真正无法可靠判定的早期失败记录。

筛选规则（分子级）：
1) 完整分子的 heavy_atom_count 必须满足 min_heavy <= H <= max_heavy。
   默认：13 <= H <= 35，包含边界。
2) 该分子至少存在一个连续稠环 ring system，其
   sssr_basis_ring_count >= min_fused_rings。
   默认：>= 3。
3) 一旦分子满足上述两条，保留这个分子的“全部” ring systems，
   包括其中的 2 环 ring system；同时保留已有的全部 scaffold_variants、
   attachment_sites 以及与该分子关联的 failure 记录。
4) build_pure_ring_scaffolds / build_expanded_scaffolds 属于“可恢复骨架失败”，
   不会因为失败而把满足筛选条件的分子删除。
5) 下列早期失败单独汇总到 unresolved_failures.db：
   - missing_smiles
   - parse_smiles
   - molecule_facts
   - find_strict_fused_ring_systems
   - unexpected_worker_error
   这些记录不混入“可用库”，也不会被静默丢弃。
6) no_strict_fused_system 是“已成功判断为没有严格连续稠环”，不是 unresolved。

安全原则：
- 原始 120 个 part 只读，绝不执行 DELETE/UPDATE。
- 输出到全新目录。
- 每个 part 独立处理；内部采用顺序扫描优先策略，减少共享存储上的随机 I/O。
- 这一步主要受 SQLite I/O / CPU 限制，不使用 GPU；A100 留给后续模型训练。

典型运行：
    nohup python -u filter_fused_ring_parts_13_35_3plus.py \
      --input-dir fused_ring_master_library/_parts_master_v1 \
      --output-dir fused_ring_filtered_h13_35_3plus \
      --workers 16 \
      --overwrite \
      > filter_h13_35_3plus.log 2>&1 &

输出：
    fused_ring_filtered_h13_35_3plus/
      _parts_filtered_v1/
        part_0000.db
        ...
      _unresolved_parts_tmp/       # 正常完成后自动删除
      unresolved_failures.db
      filter_summary.csv
      filter_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sqlite3
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FILTER_VERSION = "filter_h13_35_ge3_v2_sequential"

UNRESOLVED_STAGES = (
    "missing_smiles",
    "parse_smiles",
    "molecule_facts",
    "find_strict_fused_ring_systems",
    "unexpected_worker_error",
)

RECOVERABLE_SCAFFOLD_FAILURE_STAGES = (
    "build_pure_ring_scaffolds",
    "build_expanded_scaffolds",
)

EXPECTED_TABLES = (
    "extraction_runs",
    "molecules",
    "ring_systems",
    "scaffold_variants",
    "attachment_sites",
    "failures",
    "metadata",
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def remove_sqlite_files(path: Path) -> None:
    for p in (path, Path(str(path) + "-wal"), Path(str(path) + "-shm")):
        if p.exists():
            p.unlink()


def table_exists(conn: sqlite3.Connection, db_alias: str, table: str) -> bool:
    row = conn.execute(
        f"SELECT 1 FROM {db_alias}.sqlite_master "
        "WHERE type='table' AND name=? LIMIT 1",
        (table,),
    ).fetchone()
    return row is not None


def get_table_columns(
    conn: sqlite3.Connection,
    db_alias: str,
    table: str,
) -> list[str]:
    return [
        row[1]
        for row in conn.execute(
            f"PRAGMA {db_alias}.table_info({quote_identifier(table)})"
        )
    ]


def create_table_like_source(
    conn: sqlite3.Connection,
    source_alias: str,
    table: str,
) -> None:
    """
    在输出库中创建与源表同列结构的表。
    使用 CREATE TABLE AS ... WHERE 0，故不复制源 FK/UNIQUE 约束，
    避免 part 库中 extraction_runs 为空时产生无意义的 FK 冲突。
    数据字段和字段顺序保持一致。
    """
    conn.execute(
        f"CREATE TABLE {quote_identifier(table)} AS "
        f"SELECT * FROM {source_alias}.{quote_identifier(table)} WHERE 0"
    )


def create_filtered_indexes(conn: sqlite3.Connection) -> None:
    # 只创建后续跨分片筛选/去重最常用的索引。
    index_sql = (
        "CREATE INDEX IF NOT EXISTS idx_molecules_uid "
        "ON molecules(molecule_uid)",
        "CREATE INDEX IF NOT EXISTS idx_molecules_heavy "
        "ON molecules(heavy_atom_count)",
        "CREATE INDEX IF NOT EXISTS idx_molecules_canonical_noniso "
        "ON molecules(canonical_nonisomeric_smiles)",
        "CREATE INDEX IF NOT EXISTS idx_ring_systems_molecule "
        "ON ring_systems(molecule_uid)",
        "CREATE INDEX IF NOT EXISTS idx_ring_systems_ring_count "
        "ON ring_systems(sssr_basis_ring_count)",
        "CREATE INDEX IF NOT EXISTS idx_scaffold_molecule "
        "ON scaffold_variants(molecule_uid)",
        "CREATE INDEX IF NOT EXISTS idx_scaffold_ring_system "
        "ON scaffold_variants(ring_system_uid)",
        "CREATE INDEX IF NOT EXISTS idx_scaffold_type_smiles "
        "ON scaffold_variants(variant_type, scaffold_smiles)",
        "CREATE INDEX IF NOT EXISTS idx_attachment_molecule "
        "ON attachment_sites(molecule_uid)",
        "CREATE INDEX IF NOT EXISTS idx_failures_molecule "
        "ON failures(molecule_uid)",
        "CREATE INDEX IF NOT EXISTS idx_failures_stage "
        "ON failures(failure_stage)",
    )
    for sql in index_sql:
        try:
            conn.execute(sql)
        except sqlite3.OperationalError:
            # 若未来 schema 缺少某字段，不让索引创建影响主体数据复制。
            pass


def validate_source_part(conn: sqlite3.Connection, source_alias: str) -> None:
    missing = [
        t for t in ("molecules", "ring_systems", "scaffold_variants",
                    "attachment_sites", "failures")
        if not table_exists(conn, source_alias, t)
    ]
    if missing:
        raise RuntimeError(
            "源分片缺少必要数据表：" + ", ".join(missing)
        )

    mol_cols = set(get_table_columns(conn, source_alias, "molecules"))
    rs_cols = set(get_table_columns(conn, source_alias, "ring_systems"))
    required_mol = {"molecule_uid", "heavy_atom_count"}
    required_rs = {"molecule_uid", "sssr_basis_ring_count"}

    if not required_mol.issubset(mol_cols):
        raise RuntimeError(
            f"molecules 缺少字段：{sorted(required_mol - mol_cols)}"
        )
    if not required_rs.issubset(rs_cols):
        raise RuntimeError(
            f"ring_systems 缺少字段：{sorted(required_rs - rs_cols)}"
        )


def make_filter_metadata_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS filter_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )


def write_filter_metadata(
    conn: sqlite3.Connection,
    *,
    source_part: str,
    min_heavy: int,
    max_heavy: int,
    min_fused_rings: int,
) -> None:
    values = {
        "filter_version": FILTER_VERSION,
        "source_part": source_part,
        "min_heavy_atom_count": min_heavy,
        "max_heavy_atom_count": max_heavy,
        "min_fused_ring_count_for_molecule_eligibility": min_fused_rings,
        "policy":
            "molecule-level eligibility; if molecule has >=1 qualifying "
            "ring system, retain ALL ring systems for that molecule",
        "created_at_utc": utc_now_iso(),
    }
    conn.executemany(
        "INSERT OR REPLACE INTO filter_metadata(key, value) VALUES (?, ?)",
        [(k, str(v)) for k, v in values.items()],
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

        CREATE INDEX IF NOT EXISTS idx_unresolved_stage
            ON unresolved_failures(failure_stage);
        CREATE INDEX IF NOT EXISTS idx_unresolved_rowid
            ON unresolved_failures(source_rowid);
        CREATE INDEX IF NOT EXISTS idx_unresolved_molecule
            ON unresolved_failures(molecule_uid);
        """
    )


def process_one_part(args: tuple[Any, ...]) -> dict[str, Any]:
    """
    高速版核心策略：
    1. 顺序扫描 ring_systems 一次，把 >= min_fused_rings 的 molecule_uid
       收进 TEMP qualifying_molecules。
    2. 只针对 qualifying_molecules，通过 molecules 的主键索引检查重原子范围，
       得到 selected_molecules。
    3. 对各明细表各做一次顺序扫描/连接复制。

    这样避免旧版针对数百万 molecules 逐条执行 correlated EXISTS 所造成的
    大量随机 I/O，尤其适合共享文件系统上的几十 GB SQLite 分片。
    """
    (
        part_path_str,
        output_part_str,
        unresolved_part_str,
        min_heavy,
        max_heavy,
        min_fused_rings,
    ) = args

    part_path = Path(part_path_str)
    output_part = Path(output_part_str)
    unresolved_part = Path(unresolved_part_str)

    remove_sqlite_files(output_part)
    remove_sqlite_files(unresolved_part)

    output_part.parent.mkdir(parents=True, exist_ok=True)
    unresolved_part.parent.mkdir(parents=True, exist_ok=True)

    stats: dict[str, Any] = {
        "part": part_path.name,
        "qualifying_molecule_ids_ge3": 0,
        "eligible_molecules": 0,
        "eligible_ring_systems_all": 0,
        "eligible_ring_systems_2ring": 0,
        "eligible_ring_systems_ge3": 0,
        "eligible_scaffolds": 0,
        "eligible_attachments": 0,
        "eligible_failure_records": 0,
        "eligible_pure_failures": 0,
        "eligible_expanded_failures": 0,
        "unresolved_failure_records": 0,
    }

    print(f"[{part_path.name}] STEP 1/7 打开源分片并检查 schema ...", flush=True)

    with sqlite3.connect(output_part) as out:
        out.execute("PRAGMA journal_mode=OFF")
        out.execute("PRAGMA synchronous=OFF")
        out.execute("PRAGMA temp_store=MEMORY")
        out.execute("PRAGMA foreign_keys=OFF")
        out.execute("PRAGMA cache_size=-262144")
        out.execute("PRAGMA mmap_size=1073741824")

        source_path = str(part_path.resolve())
        out.execute("ATTACH DATABASE ? AS src", (source_path,))
        validate_source_part(out, "src")

        print(
            f"[{part_path.name}] STEP 2/7 顺序扫描 ring_systems，"
            f"收集 >= {min_fused_rings} 环的 molecule_uid ...",
            flush=True,
        )

        out.execute(
            """
            CREATE TEMP TABLE qualifying_molecules (
                molecule_uid TEXT PRIMARY KEY
            ) WITHOUT ROWID
            """
        )
        out.execute(
            """
            INSERT OR IGNORE INTO qualifying_molecules(molecule_uid)
            SELECT molecule_uid
            FROM src.ring_systems
            WHERE sssr_basis_ring_count >= ?
            """,
            (min_fused_rings,),
        )

        stats["qualifying_molecule_ids_ge3"] = out.execute(
            "SELECT COUNT(*) FROM qualifying_molecules"
        ).fetchone()[0]

        print(
            f"[{part_path.name}] STEP 2/7 完成："
            f"{stats['qualifying_molecule_ids_ge3']:,} 个分子至少有一个 "
            f">={min_fused_rings}-ring system",
            flush=True,
        )

        print(
            f"[{part_path.name}] STEP 3/7 按 molecules 主键检查重原子 "
            f"{min_heavy}-{max_heavy} ...",
            flush=True,
        )

        out.execute(
            """
            CREATE TEMP TABLE selected_molecules (
                molecule_uid TEXT PRIMARY KEY
            ) WITHOUT ROWID
            """
        )

        out.execute(
            """
            INSERT OR IGNORE INTO selected_molecules(molecule_uid)
            SELECT q.molecule_uid
            FROM qualifying_molecules AS q
            WHERE EXISTS (
                SELECT 1
                FROM src.molecules AS m
                WHERE m.molecule_uid = q.molecule_uid
                  AND m.heavy_atom_count BETWEEN ? AND ?
            )
            """,
            (min_heavy, max_heavy),
        )

        stats["eligible_molecules"] = out.execute(
            "SELECT COUNT(*) FROM selected_molecules"
        ).fetchone()[0]

        print(
            f"[{part_path.name}] STEP 3/7 完成："
            f"{stats['eligible_molecules']:,} 个分子最终通过",
            flush=True,
        )

        print(
            f"[{part_path.name}] STEP 4/7 创建输出表并复制 molecules / ring_systems ...",
            flush=True,
        )

        for table in EXPECTED_TABLES:
            if table_exists(out, "src", table):
                create_table_like_source(out, "src", table)

        make_filter_metadata_table(out)

        for table in ("extraction_runs", "metadata"):
            if table_exists(out, "main", table) and table_exists(out, "src", table):
                out.execute(
                    f"INSERT INTO {quote_identifier(table)} "
                    f"SELECT * FROM src.{quote_identifier(table)}"
                )

        out.execute(
            """
            INSERT INTO molecules
            SELECT m.*
            FROM selected_molecules AS s
            JOIN src.molecules AS m
              ON m.molecule_uid = s.molecule_uid
            """
        )

        out.execute(
            """
            INSERT INTO ring_systems
            SELECT r.*
            FROM src.ring_systems AS r
            WHERE r.molecule_uid IN (
                SELECT molecule_uid FROM selected_molecules
            )
            """
        )

        stats["eligible_ring_systems_all"] = out.execute(
            "SELECT COUNT(*) FROM ring_systems"
        ).fetchone()[0]
        stats["eligible_ring_systems_2ring"] = out.execute(
            "SELECT COUNT(*) FROM ring_systems WHERE sssr_basis_ring_count=2"
        ).fetchone()[0]
        stats["eligible_ring_systems_ge3"] = out.execute(
            "SELECT COUNT(*) FROM ring_systems WHERE sssr_basis_ring_count>=?",
            (min_fused_rings,),
        ).fetchone()[0]

        print(
            f"[{part_path.name}] STEP 4/7 完成："
            f"保留 {stats['eligible_ring_systems_all']:,} 个 ring systems，"
            f"其中 2-ring={stats['eligible_ring_systems_2ring']:,}",
            flush=True,
        )

        print(
            f"[{part_path.name}] STEP 5/7 顺序复制 scaffold / attachment / "
            f"可恢复 failure ...",
            flush=True,
        )

        out.execute(
            """
            INSERT INTO scaffold_variants
            SELECT v.*
            FROM src.scaffold_variants AS v
            WHERE v.molecule_uid IN (
                SELECT molecule_uid FROM selected_molecules
            )
            """
        )
        print(f"[{part_path.name}]   scaffold_variants 完成", flush=True)

        out.execute(
            """
            INSERT INTO attachment_sites
            SELECT a.*
            FROM src.attachment_sites AS a
            WHERE a.molecule_uid IN (
                SELECT molecule_uid FROM selected_molecules
            )
            """
        )
        print(f"[{part_path.name}]   attachment_sites 完成", flush=True)

        out.execute(
            """
            INSERT INTO failures
            SELECT f.*
            FROM src.failures AS f
            WHERE f.molecule_uid IN (
                SELECT molecule_uid FROM selected_molecules
            )
            """
        )
        print(f"[{part_path.name}]   failures 完成", flush=True)

        write_filter_metadata(
            out,
            source_part=part_path.name,
            min_heavy=min_heavy,
            max_heavy=max_heavy,
            min_fused_rings=min_fused_rings,
        )

        stats["eligible_scaffolds"] = out.execute(
            "SELECT COUNT(*) FROM scaffold_variants"
        ).fetchone()[0]
        stats["eligible_attachments"] = out.execute(
            "SELECT COUNT(*) FROM attachment_sites"
        ).fetchone()[0]
        stats["eligible_failure_records"] = out.execute(
            "SELECT COUNT(*) FROM failures"
        ).fetchone()[0]
        stats["eligible_pure_failures"] = out.execute(
            """
            SELECT COUNT(*) FROM failures
            WHERE failure_stage='build_pure_ring_scaffolds'
            """
        ).fetchone()[0]
        stats["eligible_expanded_failures"] = out.execute(
            """
            SELECT COUNT(*) FROM failures
            WHERE failure_stage='build_expanded_scaffolds'
            """
        ).fetchone()[0]

        print(f"[{part_path.name}] STEP 6/7 创建输出索引 ...", flush=True)
        create_filtered_indexes(out)

        out.commit()
        out.execute("DETACH DATABASE src")

    print(
        f"[{part_path.name}] STEP 7/7 汇总真正无法判定的早期 failures ...",
        flush=True,
    )

    with sqlite3.connect(unresolved_part) as u:
        u.execute("PRAGMA journal_mode=OFF")
        u.execute("PRAGMA synchronous=OFF")
        u.execute("PRAGMA temp_store=MEMORY")
        u.execute("PRAGMA cache_size=-131072")
        create_unresolved_schema(u)

        source_path = str(part_path.resolve())
        u.execute("ATTACH DATABASE ? AS src", (source_path,))

        placeholders = ",".join("?" for _ in UNRESOLVED_STAGES)
        u.execute(
            f"""
            INSERT INTO unresolved_failures (
                source_part,
                failure_uid, run_id, source_rowid, source_record_id,
                source_cid, smiles, molecule_uid, system_index,
                failure_stage, error_type, error_message, created_at_utc,
                known_heavy_atom_count,
                canonical_nonisomeric_smiles,
                filter_relevance
            )
            SELECT
                ? AS source_part,
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

    print(
        f"[{part_path.name}] DONE | eligible molecules="
        f"{stats['eligible_molecules']:,} | retained ring systems="
        f"{stats['eligible_ring_systems_all']:,} | unresolved="
        f"{stats['unresolved_failure_records']:,}",
        flush=True,
    )
    return stats

def merge_unresolved_parts(
    sidecars: list[Path],
    final_db: Path,
) -> int:
    remove_sqlite_files(final_db)
    with sqlite3.connect(final_db) as out:
        out.execute("PRAGMA journal_mode=OFF")
        out.execute("PRAGMA synchronous=OFF")
        out.execute("PRAGMA temp_store=MEMORY")
        create_unresolved_schema(out)

        for i, sidecar in enumerate(sidecars, start=1):
            alias = f"u{i}"
            out.execute("ATTACH DATABASE ? AS " + alias, (str(sidecar),))
            out.execute(
                f"""
                INSERT INTO unresolved_failures
                SELECT * FROM {alias}.unresolved_failures
                """
            )
            out.commit()
            out.execute(f"DETACH DATABASE {alias}")

        total = out.execute(
            "SELECT COUNT(*) FROM unresolved_failures"
        ).fetchone()[0]

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
        return int(total)


def sum_numeric_stats(rows: list[dict[str, Any]]) -> dict[str, int]:
    if not rows:
        return {}
    numeric_keys = [
        k for k, v in rows[0].items()
        if k != "part" and isinstance(v, int)
    ]
    return {
        key: sum(int(row.get(key, 0)) for row in rows)
        for key in numeric_keys
    }


def write_summary_csv(
    path: Path,
    part_stats: list[dict[str, Any]],
    totals: dict[str, int],
) -> None:
    if not part_stats:
        return
    fields = list(part_stats[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in part_stats:
            writer.writerow(row)
        total_row: dict[str, Any] = {"part": "TOTAL"}
        for field in fields:
            if field == "part":
                continue
            total_row[field] = totals.get(field, "")
        writer.writerow(total_row)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "跨 120 个 fused-ring part 做分子级筛选："
            "heavy atoms 13-35 + 至少一个 >=3-ring fused system；"
            "通过后保留该分子的全部 ring systems。"
        )
    )
    p.add_argument(
        "--input-dir",
        type=Path,
        default=Path("fused_ring_master_library/_parts_master_v1"),
        help="原始 part_XXXX.db 所在目录。",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("fused_ring_filtered_h13_35_3plus"),
        help="新筛选结果目录。",
    )
    p.add_argument(
        "--glob",
        default="part_*.db",
        help="分片文件匹配模式，默认 part_*.db。",
    )
    p.add_argument(
        "--min-heavy",
        type=int,
        default=13,
        help="最小重原子数，包含边界，默认 13。",
    )
    p.add_argument(
        "--max-heavy",
        type=int,
        default=35,
        help="最大重原子数，包含边界，默认 35。",
    )
    p.add_argument(
        "--min-fused-rings",
        type=int,
        default=3,
        help="判定分子可用所需的最小 ring-system 环数，默认 3。",
    )
    default_workers = min(16, max(1, (os.cpu_count() or 8) - 2))
    p.add_argument(
        "--workers",
        type=int,
        default=default_workers,
        help=(
            f"并行处理 part 的进程数，默认 {default_workers}。"
            "SQLite/I/O 任务不建议盲目开满所有 CPU。"
        ),
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="若输出目录存在，允许先删除后重建。",
    )
    p.add_argument(
        "--keep-unresolved-parts",
        action="store_true",
        help="保留 unresolved 临时 sidecar 分片，默认最终合并后删除。",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.min_heavy < 1:
        raise ValueError("--min-heavy 必须 >= 1")
    if args.max_heavy < args.min_heavy:
        raise ValueError("--max-heavy 必须 >= --min-heavy")
    if args.min_fused_rings < 2:
        raise ValueError("--min-fused-rings 建议 >= 2")
    if args.workers < 1:
        raise ValueError("--workers 必须 >= 1")

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    filtered_dir = output_dir / "_parts_filtered_v1"
    unresolved_tmp_dir = output_dir / "_unresolved_parts_tmp"
    final_unresolved = output_dir / "unresolved_failures.db"

    if not input_dir.exists():
        raise FileNotFoundError(f"找不到输入目录：{input_dir}")

    part_paths = sorted(input_dir.glob(args.glob))
    if not part_paths:
        raise FileNotFoundError(
            f"输入目录中没有匹配 {args.glob} 的数据库：{input_dir}"
        )

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"输出目录已存在：{output_dir}\n"
                "若确认重建，请增加 --overwrite。"
            )
        shutil.rmtree(output_dir)

    filtered_dir.mkdir(parents=True, exist_ok=True)
    unresolved_tmp_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 92, flush=True)
    print("Fused-ring 分片筛选开始", flush=True)
    print("filter_version:", FILTER_VERSION, flush=True)
    print("输入目录:", input_dir, flush=True)
    print("发现分片:", len(part_paths), flush=True)
    print(
        "分子筛选:",
        f"{args.min_heavy} <= heavy_atom_count <= {args.max_heavy}",
        "AND",
        f"至少一个 ring_system.sssr_basis_ring_count >= {args.min_fused_rings}",
        flush=True,
    )
    print(
        "保留策略:",
        "一旦分子通过，保留该分子的全部 ring systems（包括 2-ring）",
        flush=True,
    )
    print("并行 workers:", args.workers, flush=True)
    print(
        "GPU:",
        "本步骤不使用；SQLite/RDKit 数据筛选主要由 CPU + I/O 决定",
        flush=True,
    )
    print("输出目录:", output_dir, flush=True)
    print("=" * 92, flush=True)

    jobs: list[tuple[Any, ...]] = []
    sidecars: list[Path] = []

    for part in part_paths:
        out_part = filtered_dir / part.name
        unresolved_part = unresolved_tmp_dir / (
            part.stem + "_unresolved.db"
        )
        sidecars.append(unresolved_part)
        jobs.append(
            (
                str(part),
                str(out_part),
                str(unresolved_part),
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

        for done_idx, future in enumerate(
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
            rate = done_idx / elapsed
            remaining = len(jobs) - done_idx
            eta_min = (remaining / rate / 60) if rate > 0 else float("inf")

            print(
                "[PROGRESS] "
                f"{done_idx}/{len(jobs)} | "
                f"{part_name} | "
                f"eligible molecules={stats['eligible_molecules']:,} | "
                f"all retained ring systems={stats['eligible_ring_systems_all']:,} | "
                f"unresolved={stats['unresolved_failure_records']:,} | "
                f"ETA={eta_min:.1f} min",
                flush=True,
            )

    results.sort(key=lambda x: x["part"])
    totals = sum_numeric_stats(results)

    print("[MERGE] 合并 unresolved failure sidecars ...", flush=True)
    unresolved_total = merge_unresolved_parts(sidecars, final_unresolved)

    if not args.keep_unresolved_parts:
        shutil.rmtree(unresolved_tmp_dir, ignore_errors=True)

    summary_csv = output_dir / "filter_summary.csv"
    summary_json = output_dir / "filter_summary.json"

    write_summary_csv(summary_csv, results, totals)

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
            "molecule_must_have_at_least_one_ring_system_with_sssr_basis_ring_count_ge":
                args.min_fused_rings,
            "retain_all_ring_systems_after_molecule_is_eligible": True,
        },
        "unresolved_stages": list(UNRESOLVED_STAGES),
        "recoverable_scaffold_failure_stages":
            list(RECOVERABLE_SCAFFOLD_FAILURE_STAGES),
        "totals": totals,
        "final_unresolved_failure_count": unresolved_total,
        "notes": [
            "原始 part 数据库只读，未修改。",
            "build_pure_ring_scaffolds/build_expanded_scaffolds 失败不自动删除满足筛选条件的分子。",
            "no_strict_fused_system 表示已判定无严格连续稠环，不进入 unresolved。",
            "GPU 不用于本步骤；性能主要取决于 CPU、SQLite 查询和存储 I/O。",
        ],
    }
    summary_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    elapsed = time.time() - started

    print("=" * 92, flush=True)
    print("筛选完成", flush=True)
    print("耗时:", f"{elapsed / 60:.2f} 分钟", flush=True)
    print("原始分片:", len(part_paths), flush=True)
    print(
        "最终可用 molecules:",
        f"{totals.get('eligible_molecules', 0):,}",
        flush=True,
    )
    print(
        "保留的全部 ring systems:",
        f"{totals.get('eligible_ring_systems_all', 0):,}",
        flush=True,
    )
    print(
        "其中 2-ring systems（因所属分子可用而保留）:",
        f"{totals.get('eligible_ring_systems_2ring', 0):,}",
        flush=True,
    )
    print(
        f"其中 >={args.min_fused_rings}-ring systems:",
        f"{totals.get('eligible_ring_systems_ge3', 0):,}",
        flush=True,
    )
    print(
        "可用分子中的 pure scaffold failures:",
        f"{totals.get('eligible_pure_failures', 0):,}",
        flush=True,
    )
    print(
        "可用分子中的 expanded scaffold failures:",
        f"{totals.get('eligible_expanded_failures', 0):,}",
        flush=True,
    )
    print(
        "最终 unresolved failures:",
        f"{unresolved_total:,}",
        flush=True,
    )
    print("可用分片目录:", filtered_dir, flush=True)
    print("无法判定失败库:", final_unresolved, flush=True)
    print("统计 CSV:", summary_csv, flush=True)
    print("统计 JSON:", summary_json, flush=True)
    print("=" * 92, flush=True)


if __name__ == "__main__":
    main()
