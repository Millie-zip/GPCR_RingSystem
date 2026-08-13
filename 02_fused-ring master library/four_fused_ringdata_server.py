# -*- coding: utf-8 -*-
"""
四个稠环数据库合并、标准化、完整分子去重（Linux服务器版）

适用场景
--------
- 四个 SQLite 数据库合计约 30 GB
- ZINC 单库约 2.28 亿条记录
- Windows + RDKit + 多核 CPU
- 目标：在当前约 8000 条/秒基础上尽量提升到约 10000~15000 条/秒

核心优化
--------
1. 22 个 RDKit 工作进程
2. 每批 20000 条，减少数据库往返
3. pool.map chunksize=500，减少多进程调度开销
4. 每 20 万条保存一次进度
5. SQLite 1 GB 缓存
6. synchronous=OFF，减少磁盘同步等待
7. locking_mode=EXCLUSIVE
8. 合并期间不创建非必要索引
9. 主进程独占 SQLite 读写，子进程只处理 SMILES
10. 支持 Ctrl+C 后续跑

注意
----
synchronous=OFF 会提高速度，但如果电脑突然断电，当前正在写入的一小批数据
可能损坏。正常停止、正常关机一般没有问题。程序支持中断续跑。
"""

import os
import re
import csv
import time
import sqlite3
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize


# ============================================================
# 1. 输入数据库
# ============================================================

DATABASES = {
    "ZINC": {
        "path": Path("/share/home/weiliao02/lyx/zinc_project/ZINC_Fused_Ring.db"),
        "table": "fused_ring_molecules",
        "smiles_column": "smiles",
        "id_column": "id",
    },

    "PubChem": {
        "path": Path("/share/home/weiliao02/lyx/zinc_project/PubChem_Fused_Ring.db"),
        "table": None,
        "smiles_column": None,
        "id_column": None,
    },

    "COCONUT": {
        "path": Path("/share/home/weiliao02/lyx/zinc_project/COCONUT_Fused_Ring.db"),
        "table": None,
        "smiles_column": None,
        "id_column": None,
    },

    "ChEMBL": {
        "path": Path("/share/home/weiliao02/lyx/zinc_project/ChEMBL_Fused_Ring.db"),
        "table": None,
        "smiles_column": None,
        "id_column": None,
    },
}


# ============================================================
# 2. 输出设置
# ============================================================

OUTPUT_DIR = Path("/share/home/weiliao02/lyx/zinc_project/merged_fused_ring_database")
OUTPUT_DB = OUTPUT_DIR / "merged_fused_ring_unique.db"
LOG_FILE = OUTPUT_DIR / "merge_run.log"

# 大数据库默认不立即导出 CSV/SMI。
# 四库全部合并完成后，如需导出，改成 True 再运行一次。
EXPORT_CSV_SMI = False

OUTPUT_CSV = OUTPUT_DIR / "merged_fused_ring_unique.csv"
OUTPUT_SMI = OUTPUT_DIR / "merged_fused_ring_unique.smi"


# ============================================================
# 3. 加速参数
# ============================================================

# 你的电脑显示 24 逻辑核心，使用 22 个。
CPU_WORKERS = 28

# 每批处理 2 万条，减少 SQLite 读取、写入与多进程调用次数。
BATCH_SIZE = 40000

# 每个子进程一次领取 500 条任务。
POOL_CHUNKSIZE = 1000

# 每 20 万条提交一次进度。
CHECKPOINT_EVERY = 400000

# SQLite 缓存约 1 GB，单位 KiB。
# 如果电脑只有 16 GB 内存，可改为 -524288。
SQLITE_CACHE_KIB = -8388608

# True：去盐、保留最大主体、规范化可中和电荷后去重。
STANDARDIZE_PARENT = True


# ============================================================
# 4. 表和字段识别
# ============================================================

SMILES_CANDIDATES = [
    "canonical_isomeric_smiles",
    "canonical_smiles",
    "isomeric_smiles",
    "standardized_smiles",
    "standard_smiles",
    "smiles",
    "canonicalsmiles",
    "isomericsmiles",
    "mol_smiles",
]

ID_CANDIDATES = [
    "id",
    "cid",
    "zinc_id",
    "pubchem_cid",
    "chembl_id",
    "coconut_id",
    "molecule_id",
    "compound_id",
    "mol_id",
    "record_id",
    "identifier",
    "name",
]

SKIP_TABLES = {
    "sqlite_sequence",
    "completed_tasks",
    "download_tasks",
    "task_status",
    "metadata",
    "merge_progress",
    "merge_statistics",
    "rejected_records",
    "molecules",
}


# ============================================================
# 5. 基础工具
# ============================================================

def log(message: str = "") -> None:
    text = str(message)
    print(text, flush=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with LOG_FILE.open("a", encoding="utf-8") as fh:
        fh.write(text + "\n")


def qident(name: str) -> str:
    """安全引用 SQLite 表名或字段名。"""
    return '"' + str(name).replace('"', '""') + '"'


def normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def find_named_column(
    columns: List[str],
    candidates: List[str],
) -> Optional[str]:
    exact_map = {
        column.lower(): column
        for column in columns
    }

    for candidate in candidates:
        if candidate.lower() in exact_map:
            return exact_map[candidate.lower()]

    normalized_map = {
        normalize_name(column): column
        for column in columns
    }

    for candidate in candidates:
        key = normalize_name(candidate)

        if key in normalized_map:
            return normalized_map[key]

    return None


def get_tables(conn: sqlite3.Connection) -> List[str]:
    rows = conn.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type='table'
          AND name NOT LIKE 'sqlite_%'
        ORDER BY name
        """
    ).fetchall()

    return [row[0] for row in rows]


def get_columns(
    conn: sqlite3.Connection,
    table_name: str,
) -> List[str]:
    rows = conn.execute(
        f"PRAGMA table_info({qident(table_name)})"
    ).fetchall()

    return [row[1] for row in rows]


def choose_input_tables(
    source: str,
    config: dict,
) -> List[dict]:
    """
    优先使用手工指定表结构。
    未指定时，只按明确字段名识别，不尝试解析 URL、日期等字段。
    """

    db_path = config["path"]

    with sqlite3.connect(str(db_path)) as conn:
        configured_table = config.get("table")

        if configured_table:
            columns = get_columns(conn, configured_table)

            if not columns:
                raise RuntimeError(
                    f"{source}: 找不到指定表 {configured_table}"
                )

            smiles_col = config.get("smiles_column")

            if smiles_col not in columns:
                raise RuntimeError(
                    f"{source}: 表 {configured_table} 中找不到 "
                    f"SMILES 字段 {smiles_col}；实际字段：{columns}"
                )

            id_col = config.get("id_column")

            if id_col and id_col not in columns:
                log(
                    f"警告：{source}/{configured_table} 没有 ID 字段 "
                    f"{id_col}，改用 rowid。"
                )
                id_col = None

            return [{
                "table": configured_table,
                "smiles_column": smiles_col,
                "id_column": id_col,
            }]

        detected = []

        for table_name in get_tables(conn):
            if table_name.lower() in SKIP_TABLES:
                continue

            columns = get_columns(conn, table_name)

            smiles_col = find_named_column(
                columns,
                SMILES_CANDIDATES,
            )

            if smiles_col is None:
                continue

            id_col = find_named_column(
                columns,
                ID_CANDIDATES,
            )

            detected.append({
                "table": table_name,
                "smiles_column": smiles_col,
                "id_column": id_col,
            })

        if not detected:
            schemas = []

            for table_name in get_tables(conn):
                schemas.append(
                    f"{table_name}: {get_columns(conn, table_name)}"
                )

            raise RuntimeError(
                f"{source}: 未找到包含明确 SMILES 字段的数据表。\n"
                + "\n".join(schemas)
            )

        return detected


# ============================================================
# 6. SQLite 设置
# ============================================================

def configure_input_connection(
    conn: sqlite3.Connection,
) -> None:
    conn.execute("PRAGMA query_only=ON")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute(f"PRAGMA cache_size={SQLITE_CACHE_KIB}")


def configure_output_connection(
    conn: sqlite3.Connection,
) -> None:
    # 加速设置
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA locking_mode=EXCLUSIVE")
    conn.execute("PRAGMA wal_autocheckpoint=100000")
    conn.execute(f"PRAGMA cache_size={SQLITE_CACHE_KIB}")
    conn.execute("PRAGMA busy_timeout=60000")


def create_output_schema(
    conn: sqlite3.Connection,
) -> None:
    """
    合并阶段只保留必需约束。
    不创建 source_mask 和 rejected_records 的额外索引，减少写入负担。
    """

    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS molecules (
            molecule_id INTEGER PRIMARY KEY AUTOINCREMENT,
            canonical_isomeric_smiles TEXT NOT NULL UNIQUE,
            first_source TEXT NOT NULL,
            first_source_id TEXT,
            source_mask INTEGER NOT NULL DEFAULT 0,
            occurrence_count INTEGER NOT NULL DEFAULT 1
        );

        CREATE TABLE IF NOT EXISTS merge_progress (
            source TEXT NOT NULL,
            input_table TEXT NOT NULL,
            last_rowid INTEGER NOT NULL DEFAULT 0,
            processed_rows INTEGER NOT NULL DEFAULT 0,
            valid_rows INTEGER NOT NULL DEFAULT 0,
            invalid_rows INTEGER NOT NULL DEFAULT 0,
            started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            completed INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (source, input_table)
        );

        CREATE TABLE IF NOT EXISTS merge_statistics (
            source TEXT NOT NULL,
            input_table TEXT NOT NULL,
            database_path TEXT NOT NULL,
            input_rows INTEGER NOT NULL DEFAULT 0,
            processed_rows INTEGER NOT NULL DEFAULT 0,
            valid_rows INTEGER NOT NULL DEFAULT 0,
            invalid_rows INTEGER NOT NULL DEFAULT 0,
            completed INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (source, input_table)
        );

        CREATE TABLE IF NOT EXISTS rejected_records (
            rejected_id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,
            input_table TEXT NOT NULL,
            input_rowid INTEGER,
            original_id TEXT,
            original_smiles TEXT,
            reason TEXT
        );
        """
    )

    conn.commit()


# ============================================================
# 7. 续跑进度
# ============================================================

def get_resume_state(
    conn: sqlite3.Connection,
    source: str,
    table_name: str,
) -> Tuple[int, int, int, int, bool]:

    row = conn.execute(
        """
        SELECT
            last_rowid,
            processed_rows,
            valid_rows,
            invalid_rows,
            completed
        FROM merge_progress
        WHERE source=? AND input_table=?
        """,
        (source, table_name),
    ).fetchone()

    if row is None:
        return 0, 0, 0, 0, False

    return (
        int(row[0]),
        int(row[1]),
        int(row[2]),
        int(row[3]),
        bool(row[4]),
    )


def save_progress(
    conn: sqlite3.Connection,
    source: str,
    table_name: str,
    last_rowid: int,
    processed_rows: int,
    valid_rows: int,
    invalid_rows: int,
    completed: bool,
) -> None:

    conn.execute(
        """
        INSERT INTO merge_progress (
            source,
            input_table,
            last_rowid,
            processed_rows,
            valid_rows,
            invalid_rows,
            completed,
            updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(source, input_table) DO UPDATE SET
            last_rowid=excluded.last_rowid,
            processed_rows=excluded.processed_rows,
            valid_rows=excluded.valid_rows,
            invalid_rows=excluded.invalid_rows,
            completed=excluded.completed,
            updated_at=CURRENT_TIMESTAMP
        """,
        (
            source,
            table_name,
            last_rowid,
            processed_rows,
            valid_rows,
            invalid_rows,
            int(completed),
        ),
    )


# ============================================================
# 8. RDKit 子进程函数
# ============================================================

def init_worker() -> None:
    # 关闭大量红色解析提示。
    # 无效记录仍会写入 rejected_records。
    RDLogger.DisableLog("rdApp.error")
    RDLogger.DisableLog("rdApp.warning")


def standardize_one(
    item: Tuple[int, object, object],
) -> Tuple[int, str, str, Optional[str], Optional[str]]:
    """
    输入：
        (rowid, original_id, original_smiles)

    输出：
        (
            rowid,
            original_id_text,
            original_smiles_text,
            canonical_smiles,
            error
        )
    """

    rowid, original_id, original_smiles = item

    original_id_text = (
        ""
        if original_id is None
        else str(original_id)
    )

    original_smiles_text = (
        ""
        if original_smiles is None
        else str(original_smiles).strip()
    )

    if not original_smiles_text:
        return (
            int(rowid),
            original_id_text,
            original_smiles_text,
            None,
            "SMILES为空",
        )

    try:
        mol = Chem.MolFromSmiles(
            original_smiles_text,
            sanitize=True,
        )

        if mol is None:
            return (
                int(rowid),
                original_id_text,
                original_smiles_text,
                None,
                "RDKit无法解析SMILES",
            )

        if STANDARDIZE_PARENT:
            mol = rdMolStandardize.Cleanup(mol)

            chooser = rdMolStandardize.LargestFragmentChooser()
            mol = chooser.choose(mol)

            uncharger = rdMolStandardize.Uncharger()
            mol = uncharger.uncharge(mol)

            mol = rdMolStandardize.Cleanup(mol)

        Chem.SanitizeMol(mol)

        Chem.AssignStereochemistry(
            mol,
            cleanIt=True,
            force=True,
        )

        canonical = Chem.MolToSmiles(
            mol,
            canonical=True,
            isomericSmiles=True,
        )

        if not canonical:
            return (
                int(rowid),
                original_id_text,
                original_smiles_text,
                None,
                "标准化后未生成SMILES",
            )

        return (
            int(rowid),
            original_id_text,
            original_smiles_text,
            canonical,
            None,
        )

    except Exception as exc:
        return (
            int(rowid),
            original_id_text,
            original_smiles_text,
            None,
            f"{type(exc).__name__}: {exc}",
        )


# ============================================================
# 9. 输入数据库分批读取
# ============================================================

def get_row_count(
    conn: sqlite3.Connection,
    table_name: str,
) -> int:

    return int(
        conn.execute(
            f"SELECT COUNT(*) FROM {qident(table_name)}"
        ).fetchone()[0]
    )


def get_max_rowid(
    conn: sqlite3.Connection,
    table_name: str,
) -> int:

    return int(
        conn.execute(
            f"""
            SELECT COALESCE(MAX(rowid), 0)
            FROM {qident(table_name)}
            """
        ).fetchone()[0]
    )


def fetch_next_batch(
    conn: sqlite3.Connection,
    table_name: str,
    smiles_column: str,
    id_column: Optional[str],
    last_rowid: int,
    batch_size: int,
) -> List[Tuple[int, object, object]]:

    id_expression = (
        qident(id_column)
        if id_column
        else "rowid"
    )

    sql = (
        f"SELECT rowid, {id_expression}, {qident(smiles_column)} "
        f"FROM {qident(table_name)} "
        f"WHERE rowid > ? "
        f"ORDER BY rowid "
        f"LIMIT ?"
    )

    return conn.execute(
        sql,
        (last_rowid, batch_size),
    ).fetchall()


# ============================================================
# 10. 输出数据库写入
# ============================================================

def write_results(
    out_conn: sqlite3.Connection,
    source: str,
    table_name: str,
    source_mask: int,
    results: List[
        Tuple[int, str, str, Optional[str], Optional[str]]
    ],
) -> Tuple[int, int, int]:

    valid_records = []
    rejected_records = []
    batch_last_rowid = 0

    for (
        rowid,
        original_id,
        original_smiles,
        canonical,
        error,
    ) in results:

        batch_last_rowid = max(
            batch_last_rowid,
            int(rowid),
        )

        if canonical is None:
            rejected_records.append(
                (
                    source,
                    table_name,
                    int(rowid),
                    original_id,
                    original_smiles,
                    error,
                )
            )
        else:
            valid_records.append(
                (
                    canonical,
                    source,
                    original_id,
                    source_mask,
                )
            )

    if valid_records:
        out_conn.executemany(
            """
            INSERT INTO molecules (
                canonical_isomeric_smiles,
                first_source,
                first_source_id,
                source_mask,
                occurrence_count
            )
            VALUES (?, ?, ?, ?, 1)
            ON CONFLICT(canonical_isomeric_smiles) DO UPDATE SET
                source_mask =
                    molecules.source_mask | excluded.source_mask,
                occurrence_count =
                    molecules.occurrence_count + 1
            """,
            valid_records,
        )

    if rejected_records:
        out_conn.executemany(
            """
            INSERT INTO rejected_records (
                source,
                input_table,
                input_rowid,
                original_id,
                original_smiles,
                reason
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            rejected_records,
        )

    return (
        batch_last_rowid,
        len(valid_records),
        len(rejected_records),
    )


# ============================================================
# 11. 单个表处理
# ============================================================

def process_table(
    pool,
    out_conn: sqlite3.Connection,
    source: str,
    source_mask: int,
    db_path: Path,
    table_info: dict,
) -> None:

    table_name = table_info["table"]
    smiles_column = table_info["smiles_column"]
    id_column = table_info["id_column"]

    (
        last_rowid,
        processed_rows,
        valid_rows,
        invalid_rows,
        completed,
    ) = get_resume_state(
        out_conn,
        source,
        table_name,
    )

    if completed:
        log(f"{source}/{table_name} 已完成，跳过。")
        return

    db_uri = db_path.resolve().as_uri() + "?mode=ro"

    with sqlite3.connect(
        db_uri,
        uri=True,
        timeout=60,
    ) as in_conn:

        configure_input_connection(in_conn)

        total_rows = get_row_count(
            in_conn,
            table_name,
        )

        max_rowid = get_max_rowid(
            in_conn,
            table_name,
        )

        out_conn.execute(
            """
            INSERT INTO merge_statistics (
                source,
                input_table,
                database_path,
                input_rows,
                processed_rows,
                valid_rows,
                invalid_rows,
                completed
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, 0)
            ON CONFLICT(source, input_table) DO UPDATE SET
                database_path=excluded.database_path,
                input_rows=excluded.input_rows,
                processed_rows=excluded.processed_rows,
                valid_rows=excluded.valid_rows,
                invalid_rows=excluded.invalid_rows,
                completed=0,
                updated_at=CURRENT_TIMESTAMP
            """,
            (
                source,
                table_name,
                str(db_path),
                total_rows,
                processed_rows,
                valid_rows,
                invalid_rows,
            ),
        )

        out_conn.commit()

        remaining = max(
            0,
            total_rows - processed_rows,
        )

        log("")
        log("=" * 72)
        log(f"来源：{source}")
        log(f"文件：{db_path}")
        log(f"表：{table_name}")
        log(f"SMILES字段：{smiles_column}")
        log(f"ID字段：{id_column or 'rowid'}")
        log(f"总记录：{total_rows:,}")
        log(f"续跑起点 rowid：{last_rowid:,}")
        log(f"剩余约：{remaining:,}")
        log("=" * 72)

        progress = tqdm(
            total=remaining,
            desc=f"{source}/{table_name}",
            unit="mol",
            dynamic_ncols=True,
        )

        checkpoint_counter = 0
        current_run_processed = 0
        start_time = time.time()

        try:
            while True:
                batch = fetch_next_batch(
                    in_conn,
                    table_name,
                    smiles_column,
                    id_column,
                    last_rowid,
                    BATCH_SIZE,
                )

                if not batch:
                    break

                results = pool.map(
                    standardize_one,
                    batch,
                    chunksize=POOL_CHUNKSIZE,
                )

                (
                    batch_last_rowid,
                    batch_valid,
                    batch_invalid,
                ) = write_results(
                    out_conn,
                    source,
                    table_name,
                    source_mask,
                    results,
                )

                batch_count = len(batch)

                last_rowid = batch_last_rowid
                processed_rows += batch_count
                current_run_processed += batch_count
                valid_rows += batch_valid
                invalid_rows += batch_invalid
                checkpoint_counter += batch_count

                progress.update(batch_count)

                if checkpoint_counter >= CHECKPOINT_EVERY:
                    save_progress(
                        out_conn,
                        source,
                        table_name,
                        last_rowid,
                        processed_rows,
                        valid_rows,
                        invalid_rows,
                        completed=False,
                    )

                    out_conn.execute(
                        """
                        UPDATE merge_statistics
                        SET
                            processed_rows=?,
                            valid_rows=?,
                            invalid_rows=?,
                            completed=0,
                            updated_at=CURRENT_TIMESTAMP
                        WHERE source=? AND input_table=?
                        """,
                        (
                            processed_rows,
                            valid_rows,
                            invalid_rows,
                            source,
                            table_name,
                        ),
                    )

                    out_conn.commit()
                    checkpoint_counter = 0

        except KeyboardInterrupt:
            save_progress(
                out_conn,
                source,
                table_name,
                last_rowid,
                processed_rows,
                valid_rows,
                invalid_rows,
                completed=False,
            )

            out_conn.execute(
                """
                UPDATE merge_statistics
                SET
                    processed_rows=?,
                    valid_rows=?,
                    invalid_rows=?,
                    completed=0,
                    updated_at=CURRENT_TIMESTAMP
                WHERE source=? AND input_table=?
                """,
                (
                    processed_rows,
                    valid_rows,
                    invalid_rows,
                    source,
                    table_name,
                ),
            )

            out_conn.commit()
            progress.close()

            log("")
            log("收到 Ctrl+C，当前进度已保存。")
            log("下次运行会从当前 rowid 继续。")

            raise

        progress.close()

        save_progress(
            out_conn,
            source,
            table_name,
            max_rowid,
            processed_rows,
            valid_rows,
            invalid_rows,
            completed=True,
        )

        out_conn.execute(
            """
            UPDATE merge_statistics
            SET
                processed_rows=?,
                valid_rows=?,
                invalid_rows=?,
                completed=1,
                updated_at=CURRENT_TIMESTAMP
            WHERE source=? AND input_table=?
            """,
            (
                processed_rows,
                valid_rows,
                invalid_rows,
                source,
                table_name,
            ),
        )

        out_conn.commit()

        elapsed = time.time() - start_time

        speed = (
            current_run_processed / elapsed
            if elapsed > 0
            else 0
        )

        log(
            f"{source}/{table_name} 完成："
            f"累计有效 {valid_rows:,}，"
            f"累计无效 {invalid_rows:,}，"
            f"本次平均约 {speed:,.0f} 条/秒。"
        )


# ============================================================
# 12. CSV / SMI 导出
# ============================================================

def export_csv_and_smi(
    conn: sqlite3.Connection,
) -> None:

    log("开始导出 CSV 和 SMI……")

    cursor = conn.execute(
        """
        SELECT
            molecule_id,
            canonical_isomeric_smiles,
            first_source,
            first_source_id,
            source_mask,
            occurrence_count
        FROM molecules
        ORDER BY molecule_id
        """
    )

    with (
        OUTPUT_CSV.open(
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as csv_fh,
        OUTPUT_SMI.open(
            "w",
            encoding="utf-8",
        ) as smi_fh,
    ):

        writer = csv.writer(csv_fh)

        writer.writerow([
            "molecule_id",
            "canonical_isomeric_smiles",
            "first_source",
            "first_source_id",
            "source_mask",
            "occurrence_count",
        ])

        while True:
            rows = cursor.fetchmany(50000)

            if not rows:
                break

            writer.writerows(rows)

            for row in rows:
                molecule_id = int(row[0])
                smiles = row[1]

                smi_fh.write(
                    f"{smiles}\tFRM_{molecule_id:09d}\n"
                )

    log(f"CSV：{OUTPUT_CSV}")
    log(f"SMI：{OUTPUT_SMI}")


# ============================================================
# 13. 主程序
# ============================================================

def main() -> None:

    mp.freeze_support()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    log("")
    log("#" * 72)
    log("四库稠环分子合并去重开始（Linux服务器版）")
    log(f"CPU工作进程：{CPU_WORKERS}")
    log(f"每批处理：{BATCH_SIZE:,}")
    log(f"进程任务块：{POOL_CHUNKSIZE}")
    log(f"每次进度提交：{CHECKPOINT_EVERY:,}")
    log(f"SQLite缓存：约 {abs(SQLITE_CACHE_KIB) / 1024 / 1024:.1f} GB")
    log(f"输出数据库：{OUTPUT_DB}")
    log("#" * 72)

    missing_files = []

    for source, config in DATABASES.items():
        db_path = config["path"]

        if not db_path.is_file():
            missing_files.append(
                f"{source}: {db_path}"
            )

    if missing_files:
        log("以下数据库文件不存在：")

        for item in missing_files:
            log(f"  - {item}")

        raise FileNotFoundError(
            "请检查脚本顶部 DATABASES 中的路径。"
        )

    all_table_configs: Dict[str, List[dict]] = {}

    for source, config in DATABASES.items():
        detected_tables = choose_input_tables(
            source,
            config,
        )

        all_table_configs[source] = detected_tables

        for table_info in detected_tables:
            log(
                f"识别成功：{source} -> "
                f"{table_info['table']} / "
                f"{table_info['smiles_column']} / "
                f"{table_info['id_column'] or 'rowid'}"
            )

    with sqlite3.connect(
        str(OUTPUT_DB),
        timeout=60,
    ) as out_conn:

        configure_output_connection(out_conn)
        create_output_schema(out_conn)

        source_masks = {
            source: 1 << index
            for index, source in enumerate(DATABASES.keys())
        }

        pool = mp.Pool(
            processes=CPU_WORKERS,
            initializer=init_worker,
            maxtasksperchild=100000,
        )

        try:
            for source, config in DATABASES.items():

                db_path = config["path"]
                source_mask = source_masks[source]

                for table_info in all_table_configs[source]:

                    process_table(
                        pool,
                        out_conn,
                        source,
                        source_mask,
                        db_path,
                        table_info,
                    )

        except KeyboardInterrupt:
            pool.terminate()
            pool.join()

            log("")
            log("程序已安全停止。")
            log("重新运行脚本即可从保存位置继续。")
            return

        except Exception:
            pool.terminate()
            pool.join()
            raise

        else:
            pool.close()
            pool.join()

        unique_count = int(
            out_conn.execute(
                "SELECT COUNT(*) FROM molecules"
            ).fetchone()[0]
        )

        valid_occurrences = int(
            out_conn.execute(
                """
                SELECT COALESCE(SUM(occurrence_count), 0)
                FROM molecules
                """
            ).fetchone()[0]
        )

        rejected_count = int(
            out_conn.execute(
                "SELECT COUNT(*) FROM rejected_records"
            ).fetchone()[0]
        )

        duplicate_count = max(
            0,
            valid_occurrences - unique_count,
        )

        log("")
        log("=" * 72)
        log("四个数据库处理完成")
        log(f"有效输入记录：{valid_occurrences:,}")
        log(f"最终唯一分子：{unique_count:,}")
        log(f"去除重复记录：{duplicate_count:,}")
        log(f"解析失败记录：{rejected_count:,}")
        log(f"结果数据库：{OUTPUT_DB}")
        log("主表：molecules")
        log("结构字段：canonical_isomeric_smiles")
        log("=" * 72)

        if EXPORT_CSV_SMI:
            export_csv_and_smi(out_conn)
        else:
            log(
                "当前未导出 CSV/SMI。需要导出时，将 "
                "EXPORT_CSV_SMI 改为 True 后重新运行。"
            )


if __name__ == "__main__":
    main()
