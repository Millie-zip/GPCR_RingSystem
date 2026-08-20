import os
import sqlite3
import time

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit import RDLogger

from gpcr_ringgen.preprocessing.paths import data_root


# =====================================================
# 关闭 RDKit 大量 warning
# =====================================================

RDLogger.DisableLog("rdApp.*")


# =====================================================
# 输入和输出路径
# =====================================================

SDF_PATH = str(data_root() / "chembl" / "chembl_37.sdf")

OUTPUT_DIR = str(data_root() / "chembl")

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

DB_PATH = os.path.join(
    OUTPUT_DIR,
    "ChEMBL_Fused_Ring.db"
)

DONE_FILE = os.path.join(
    OUTPUT_DIR,
    "chembl_processed_files.txt"
)


# =====================================================
# 筛选参数
# 与 PubChem 代码保持一致
# =====================================================

MAX_MW = 500
MAX_HEAVY = 50
BATCH_SIZE = 500


# =====================================================
# 判断稠环
#
# 与 PubChem 代码逻辑完全相同：
# 1. 两个环共享至少两个原子
# 2. 共享原子中至少有一对直接成键
# =====================================================

def get_fused_ring_info(mol):
    rings = mol.GetRingInfo().AtomRings()

    fused_pairs = 0
    fused_atoms = set()

    for i in range(len(rings)):
        for j in range(i + 1, len(rings)):
            r1 = set(rings[i])
            r2 = set(rings[j])

            shared = r1.intersection(r2)

            # 共享原子少于两个，不是稠环
            if len(shared) < 2:
                continue

            shared = list(shared)
            adjacent = False

            # 共享原子中至少有一对直接相邻
            for a in range(len(shared)):
                for b in range(a + 1, len(shared)):
                    if mol.GetBondBetweenAtoms(
                        shared[a],
                        shared[b]
                    ) is not None:
                        adjacent = True
                        break

                if adjacent:
                    break

            if adjacent:
                fused_pairs += 1
                fused_atoms.update(shared)

    return fused_pairs, fused_atoms


# =====================================================
# 获取 ChEMBL ID
#
# 优先读取 chembl_id；
# 若该属性不存在，再尝试常见属性和 _Name。
#
# 最终仍写入数据库的 cid 字段，以保证表结构一致。
# =====================================================

def get_chembl_id(mol):
    property_names = set(mol.GetPropNames())

    possible_properties = (
        "chembl_id",
        "CHEMBL_ID",
        "molecule_chembl_id"
    )

    for property_name in possible_properties:
        if property_name in property_names:
            value = mol.GetProp(property_name).strip()

            if value:
                return value

    if mol.HasProp("_Name"):
        value = mol.GetProp("_Name").strip()

        if value:
            return value

    return ""


# =====================================================
# 批量写入数据库
# =====================================================

def insert_batch(cursor, conn, batch):
    if not batch:
        return

    cursor.executemany(
        """
        INSERT INTO fused_ring_molecules
        (
            cid,
            smiles,
            mw,
            heavy_atoms,
            ring_number,
            fused_pairs,
            fused_atoms
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        batch
    )

    conn.commit()


# =====================================================
# 主程序
# =====================================================

def main():
    start_time = time.time()

    if not os.path.isfile(SDF_PATH):
        raise FileNotFoundError(
            f"找不到 ChEMBL SDF 文件：\n{SDF_PATH}"
        )

    sdf_filename = os.path.basename(SDF_PATH)

    # -------------------------------------------------
    # 已完成检查
    # -------------------------------------------------

    processed = set()

    if os.path.exists(DONE_FILE):
        with open(
            DONE_FILE,
            "r",
            encoding="utf-8"
        ) as file:
            processed = {
                line.strip()
                for line in file
                if line.strip()
            }

    if sdf_filename in processed:
        print("该 ChEMBL 文件已经完整处理过：")
        print(sdf_filename)
        print()
        print("数据库：")
        print(DB_PATH)
        return

    # -------------------------------------------------
    # 创建数据库
    # -------------------------------------------------

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS fused_ring_molecules
        (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cid TEXT,
            smiles TEXT,
            mw REAL,
            heavy_atoms INTEGER,
            ring_number INTEGER,
            fused_pairs INTEGER,
            fused_atoms INTEGER
        )
        """
    )

    conn.commit()

    # -------------------------------------------------
    # 统计变量
    # -------------------------------------------------

    total_records = 0
    valid_molecules = 0
    invalid_molecules = 0

    size_pass = 0
    ring_pass = 0
    fused_count = 0

    smiles_failures = 0
    missing_chembl_id = 0

    batch = []

    print("开始处理：")
    print(SDF_PATH)
    print()

    # -------------------------------------------------
    # 流式读取本地 SDF
    #
    # ForwardSDMolSupplier 不会一次性将整个 SDF
    # 加载到内存中。
    # -------------------------------------------------

    with open(SDF_PATH, "rb") as sdf_file:
        supplier = Chem.ForwardSDMolSupplier(
            sdf_file,
            sanitize=True,
            removeHs=True,
            strictParsing=True
        )

        for mol in supplier:
            total_records += 1

            # 无法解析的记录
            if mol is None:
                invalid_molecules += 1

                if total_records % 50000 == 0:
                    elapsed = time.time() - start_time

                    print(
                        f"记录: {total_records:,} | "
                        f"有效: {valid_molecules:,} | "
                        f"稠环: {fused_count:,} | "
                        f"耗时: {elapsed / 60:.1f} 分钟"
                    )

                continue

            valid_molecules += 1

            # -----------------------------------------
            # Step 1：小分子过滤
            # 与 PubChem 代码一致
            # -----------------------------------------

            try:
                mw = Descriptors.MolWt(mol)
                heavy = mol.GetNumHeavyAtoms()
            except Exception:
                continue

            if mw > MAX_MW or heavy > MAX_HEAVY:
                continue

            size_pass += 1

            # -----------------------------------------
            # Step 2：多环过滤
            # 与 PubChem 代码一致
            # -----------------------------------------

            try:
                ring_num = len(
                    mol.GetRingInfo().AtomRings()
                )
            except Exception:
                continue

            if ring_num < 2:
                continue

            ring_pass += 1

            # -----------------------------------------
            # Step 3：稠环过滤
            # 与 PubChem 代码一致
            # -----------------------------------------

            try:
                fused_pairs, fused_atoms = (
                    get_fused_ring_info(mol)
                )
            except Exception:
                continue

            if fused_pairs < 1:
                continue

            # -----------------------------------------
            # Step 4：生成 SMILES
            # -----------------------------------------

            try:
                smiles = Chem.MolToSmiles(
                    mol,
                    canonical=True,
                    isomericSmiles=True
                )
            except Exception:
                smiles_failures += 1
                continue

            if not smiles:
                smiles_failures += 1
                continue

            # -----------------------------------------
            # Step 5：提取 ChEMBL ID
            #
            # 为保证和 PubChem 数据库格式一致，
            # ChEMBL ID 写入 cid 字段。
            # -----------------------------------------

            chembl_id = get_chembl_id(mol)

            if not chembl_id:
                missing_chembl_id += 1

            fused_count += 1

            batch.append(
                (
                    chembl_id,
                    smiles,
                    round(mw, 2),
                    heavy,
                    ring_num,
                    fused_pairs,
                    len(fused_atoms)
                )
            )

            # -----------------------------------------
            # 每 500 条写入一次
            # 与 PubChem 代码一致
            # -----------------------------------------

            if len(batch) >= BATCH_SIZE:
                insert_batch(
                    cursor,
                    conn,
                    batch
                )

                batch = []

            # -----------------------------------------
            # 输出进度
            # -----------------------------------------

            if total_records % 50000 == 0:
                elapsed = time.time() - start_time

                print(
                    f"记录: {total_records:,} | "
                    f"有效: {valid_molecules:,} | "
                    f"稠环: {fused_count:,} | "
                    f"耗时: {elapsed / 60:.1f} 分钟"
                )

    # -------------------------------------------------
    # 写入剩余记录
    # -------------------------------------------------

    insert_batch(
        cursor,
        conn,
        batch
    )

    # -------------------------------------------------
    # 建立索引
    #
    # 不改变表结构，方便后续按 ID 和 SMILES 查询。
    # -------------------------------------------------

    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS
        idx_fused_ring_molecules_cid
        ON fused_ring_molecules(cid)
        """
    )

    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS
        idx_fused_ring_molecules_smiles
        ON fused_ring_molecules(smiles)
        """
    )

    conn.commit()
    conn.close()

    # -------------------------------------------------
    # 完成后写入进度文件
    # -------------------------------------------------

    with open(
        DONE_FILE,
        "a",
        encoding="utf-8"
    ) as file:
        file.write(sdf_filename + "\n")

    elapsed = time.time() - start_time

    print()
    print("====================================")
    print("ChEMBL 筛选完成")
    print("====================================")
    print(f"SDF 总记录数:       {total_records:,}")
    print(f"成功解析分子:       {valid_molecules:,}")
    print(f"解析失败记录:       {invalid_molecules:,}")
    print(f"通过小分子过滤:     {size_pass:,}")
    print(f"通过多环过滤:       {ring_pass:,}")
    print(f"最终多元稠环分子:   {fused_count:,}")
    print(f"SMILES 生成失败:    {smiles_failures:,}")
    print(f"缺少 ChEMBL ID:     {missing_chembl_id:,}")
    print("------------------------------------")
    print("数据库：")
    print(DB_PATH)
    print("------------------------------------")
    print(
        f"总耗时: "
        f"{elapsed / 60:.2f} 分钟"
    )
    print("====================================")


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        print()
        print("程序被手动停止。")
        print(
            "此前已经提交到 SQLite 的批次会保留，"
            "但该 SDF 不会被标记为完整处理。"
        )

    except Exception as error:
        print()
        print("程序运行失败：")
        print(repr(error))
        raise
