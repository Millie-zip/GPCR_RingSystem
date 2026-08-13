import os
import sqlite3
import time


from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit import RDLogger

from gpcr_ringgen.preprocessing.paths import data_root



# ================================
# 关闭RDKit大量warning
# ================================

RDLogger.DisableLog('rdApp.*')



# ================================
# 输入文件
# ================================

INPUT_SDF = str(data_root() / "coconut" / "coconut.sdf")



# ================================
# 输出目录
# ================================

OUTPUT_DIR = str(data_root() / "coconut")


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


DB_PATH = os.path.join(
    OUTPUT_DIR,
    "COCONUT_Fused_Ring.db"
)


DONE_FILE = os.path.join(
    OUTPUT_DIR,
    "processed_files.txt"
)



# ================================
# 参数
# 与PubChem完全一致
# ================================

MAX_MW = 500

MAX_HEAVY = 50



# ================================
# 创建数据库
# 与PubChem数据模式完全一致
# ================================

conn = sqlite3.connect(DB_PATH)

cursor = conn.cursor()


cursor.execute("""
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
""")


conn.commit()



# ================================
# 已完成文件
# ================================

if os.path.exists(DONE_FILE):

    with open(
        DONE_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        processed = set(
            x.strip()
            for x in f.readlines()
        )

else:

    processed = set()



# ================================
# 判断稠环
# 与PubChem代码完全一致
# ================================

def get_fused_ring_info(mol):


    rings = mol.GetRingInfo().AtomRings()


    fused_pairs = 0

    fused_atoms = set()



    for i in range(len(rings)):


        for j in range(i + 1, len(rings)):


            r1 = set(rings[i])

            r2 = set(rings[j])


            shared = r1.intersection(r2)



            if len(shared) < 2:

                continue



            shared = list(shared)


            adjacent = False



            for a in range(len(shared)):

                for b in range(a + 1, len(shared)):

                    if mol.GetBondBetweenAtoms(
                        shared[a],
                        shared[b]
                    ):

                        adjacent = True



            if adjacent:

                fused_pairs += 1

                fused_atoms.update(shared)



    return fused_pairs, fused_atoms



# ================================
# 读取COCONUT编号
# 只影响cid字段，不影响任何筛选逻辑
# ================================

def get_coconut_id(mol):


    prop_names = set(
        mol.GetPropNames()
    )


    possible_fields = [

        "COCONUT_ID",

        "coconut_id",

        "Coconut_ID",

        "coconutId",

        "identifier",

        "ID",

        "id"

    ]


    for field in possible_fields:

        if field in prop_names:

            value = mol.GetProp(
                field
            ).strip()

            if value:

                return value


    # SDF标题行
    if mol.HasProp("_Name"):

        value = mol.GetProp(
            "_Name"
        ).strip()

        if value:

            return value


    return ""



# ================================
# 处理COCONUT文件
# ================================

def process_file(filename):


    print("\n开始:")

    print(filename)



    count = 0

    fused_count = 0


    batch = []



    # 与PubChem一样使用ForwardSDMolSupplier
    supplier = Chem.ForwardSDMolSupplier(
        filename
    )



    for mol in supplier:


        if mol is None:

            continue



        count += 1



        mw = Descriptors.MolWt(mol)

        heavy = mol.GetNumHeavyAtoms()



        if mw > MAX_MW or heavy > MAX_HEAVY:

            continue



        ring_num = len(
            mol.GetRingInfo().AtomRings()
        )


        if ring_num < 2:

            continue



        fused_pairs, fused_atoms = get_fused_ring_info(
            mol
        )



        if fused_pairs < 1:

            continue



        fused_count += 1



        cid = get_coconut_id(
            mol
        )



        batch.append(
            (
                cid,
                Chem.MolToSmiles(mol),
                round(mw, 2),
                heavy,
                ring_num,
                fused_pairs,
                len(fused_atoms)
            )
        )



        if len(batch) >= 500:


            cursor.executemany(
                """
                INSERT INTO fused_ring_molecules
                VALUES
                (NULL,?,?,?,?,?,?,?)
                """,
                batch
            )


            conn.commit()


            batch = []



    if batch:

        cursor.executemany(
            """
            INSERT INTO fused_ring_molecules
            VALUES
            (NULL,?,?,?,?,?,?,?)
            """,
            batch
        )

        conn.commit()



    with open(
        DONE_FILE,
        "a",
        encoding="utf-8"
    ) as f:

        f.write(
            os.path.basename(filename)
            + "\n"
        )



    print(
        "完成",
        count,
        "个分子",
        "得到",
        fused_count,
        "个稠环"
    )



# ================================
# 主程序
# ================================

start = time.time()



if not os.path.exists(INPUT_SDF):

    conn.close()

    raise FileNotFoundError(
        "没有找到COCONUT文件：\n"
        + INPUT_SDF
    )



filename = os.path.basename(
    INPUT_SDF
)



if filename in processed:

    print(
        "跳过:",
        filename
    )

else:

    retry = 0


    while retry < 5:


        try:

            process_file(
                INPUT_SDF
            )

            break


        except Exception as e:


            conn.rollback()


            retry += 1


            print(
                f"{filename} 失败，"
                f"第 {retry} 次重试"
            )

            print(e)



            time.sleep(10)



    else:

        print(
            f"{filename} 连续失败5次，跳过"
        )



conn.close()



print("===================")

print("全部完成")

print(
    "数据库:",
    DB_PATH
)

print(
    "耗时:",
    round(
        time.time() - start,
        2
    ),
    "秒"
)

print("===================")
