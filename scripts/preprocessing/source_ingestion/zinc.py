import os
import re
import gzip
import sqlite3
import requests
import tempfile
import time
import shutil

from urllib.parse import urlparse

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit import RDLogger

from gpcr_ringgen.preprocessing.paths import data_root


# ================================
# 关闭RDKit大量warning
# ================================

RDLogger.DisableLog("rdApp.*")


# ================================
# 输出目录
# ================================

OUTPUT_DIR = str(data_root() / "zinc")

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ZINC网页下载得到的Raw URLs文件
URI_FILE = str(data_root() / "zinc" / "ZINC-downloader-2D-smi.uri")


# 输出数据库
DB_PATH = os.path.join(
    OUTPUT_DIR,
    "ZINC_Fused_Ring.db"
)


# 已处理文件记录
DONE_FILE = os.path.join(
    OUTPUT_DIR,
    "processed_files.txt"
)


# ================================
# 参数
# 与PubChem版本一致
# ================================

MAX_MW = 500

MAX_HEAVY = 50

BATCH_SIZE = 500

MAX_RETRY = 5


# ================================
# 创建数据库
# 数据模式与PubChem版本完全一致
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
# 读取已经完成的文件
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
            if x.strip()
        )

else:

    processed = set()


# ================================
# 从.uri文件中提取ZINC下载地址
# ================================

def get_zinc_urls():

    if not os.path.exists(URI_FILE):

        raise FileNotFoundError(
            "找不到ZINC地址文件：\n"
            f"{URI_FILE}"
        )

    with open(
        URI_FILE,
        "r",
        encoding="utf-8",
        errors="ignore"
    ) as f:

        text = f.read()

    # 提取所有HTTP或HTTPS地址
    urls = re.findall(
        r'https?://[^\s\'"<>]+',
        text
    )

    cleaned_urls = []

    for url in urls:

        # 清除脚本或文本末尾可能附带的符号
        url = url.strip()

        url = url.rstrip(
            "\\);,]"
        )

        if url:

            cleaned_urls.append(url)

    # 保持原顺序，同时去除重复地址
    unique_urls = list(
        dict.fromkeys(cleaned_urls)
    )

    return unique_urls


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
# 从URL中获取文件名
# ================================

def get_filename_from_url(url):

    parsed = urlparse(url)

    filename = os.path.basename(
        parsed.path
    )

    if not filename:

        filename = (
            "zinc_"
            + str(abs(hash(url)))
            + ".smi"
        )

    return filename


# ================================
# 为每个远程地址建立唯一任务标识
# 防止不同服务器出现相同文件名
# ================================

def get_task_id(url):

    filename = get_filename_from_url(url)

    return (
        filename
        + "_"
        + str(abs(hash(url)))
    )


# ================================
# 下载一个ZINC分块
# ================================

def download_file(url, temp_file):

    headers = {
        "User-Agent":
        "Mozilla/5.0 ZINC molecular database processor"
    }

    with requests.get(
        url,
        stream=True,
        timeout=(60, 600),
        headers=headers,
        allow_redirects=True
    ) as response:

        response.raise_for_status()

        with open(
            temp_file,
            "wb"
        ) as f:

            for chunk in response.iter_content(
                chunk_size=1024 * 1024
            ):

                if chunk:

                    f.write(chunk)


# ================================
# 判断文件是否为gzip压缩文件
# 不完全依赖文件扩展名
# ================================

def is_gzip_file(file_path):

    with open(
        file_path,
        "rb"
    ) as f:

        magic = f.read(2)

    return magic == b"\x1f\x8b"


# ================================
# 打开普通或gzip压缩SMILES文件
# ================================

def open_zinc_file(file_path):

    if is_gzip_file(file_path):

        return gzip.open(
            file_path,
            "rt",
            encoding="utf-8",
            errors="ignore"
        )

    return open(
        file_path,
        "r",
        encoding="utf-8",
        errors="ignore"
    )


# ================================
# 判断是否为表头
# ================================

def is_header(parts):

    if not parts:

        return True

    first = parts[0].strip().lower()

    header_names = {
        "smiles",
        "smile",
        "canonical_smiles",
        "canonicalsmiles",
        "zinc_id",
        "zincid",
        "name"
    }

    return first in header_names


# ================================
# 读取一行ZINC数据
# 支持常见的两种排列方式：
#
# SMILES    ZINC_ID
# ZINC_ID   SMILES
# ================================

def parse_zinc_line(line):

    line = line.strip()

    if not line:

        return None, None

    if line.startswith("#"):

        return None, None

    parts = line.split()

    if not parts:

        return None, None

    if is_header(parts):

        return None, None

    first = parts[0].strip()

    smiles = first

    zinc_id = ""

    # 常见形式：
    # SMILES ZINC_ID
    if len(parts) >= 2:

        second = parts[1].strip()

        if second.upper().startswith("ZINC"):

            zinc_id = second

    # 兼容：
    # ZINC_ID SMILES
    if first.upper().startswith("ZINC"):

        zinc_id = first

        if len(parts) >= 2:

            smiles = parts[1].strip()

        else:

            return None, None

    return smiles, zinc_id


# ================================
# 处理一个ZINC远程文件
# ================================

def process_url(url):

    filename = get_filename_from_url(url)

    task_id = get_task_id(url)

    print("\n开始:")
    print(filename)

    print("地址:")
    print(url)

    temp_dir = tempfile.mkdtemp(
        prefix="zinc_"
    )

    temp_file = os.path.join(
        temp_dir,
        filename
    )

    try:

        # ================================
        # 下载
        # ================================

        download_file(
            url,
            temp_file
        )

        file_size_mb = (
            os.path.getsize(temp_file)
            / 1024
            / 1024
        )

        print(
            "下载完成:",
            round(file_size_mb, 2),
            "MB"
        )

        count = 0

        fused_count = 0

        invalid_count = 0

        batch = []

        # ================================
        # 读取SMILES
        # ================================

        with open_zinc_file(temp_file) as f:

            for line in f:

                smiles, zinc_id = parse_zinc_line(
                    line
                )

                if not smiles:

                    continue

                mol = Chem.MolFromSmiles(
                    smiles
                )

                if mol is None:

                    invalid_count += 1

                    continue

                count += 1

                # ================================
                # 以下筛选逻辑与PubChem完全一致
                # ================================

                mw = Descriptors.MolWt(mol)

                heavy = mol.GetNumHeavyAtoms()

                if mw > MAX_MW or heavy > MAX_HEAVY:

                    continue

                ring_num = len(
                    mol.GetRingInfo().AtomRings()
                )

                if ring_num < 2:

                    continue

                fused_pairs, fused_atoms = (
                    get_fused_ring_info(mol)
                )

                if fused_pairs < 1:

                    continue

                fused_count += 1

                batch.append(
                    (
                        zinc_id,
                        Chem.MolToSmiles(mol),
                        round(mw, 2),
                        heavy,
                        ring_num,
                        fused_pairs,
                        len(fused_atoms)
                    )
                )

                # 每500个结果写入一次
                if len(batch) >= BATCH_SIZE:

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

        # 写入剩余结果
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

        # 完整处理成功后记录
        with open(
            DONE_FILE,
            "a",
            encoding="utf-8"
        ) as f:

            f.write(task_id + "\n")

        processed.add(task_id)

        print(
            "完成",
            count,
            "个有效分子",
            "得到",
            fused_count,
            "个稠环",
            "无效SMILES",
            invalid_count,
            "个"
        )

    finally:

        # 删除临时下载文件
        shutil.rmtree(
            temp_dir,
            ignore_errors=True
        )


# ================================
# 主程序
# ================================

start = time.time()


try:

    urls = get_zinc_urls()

    print(
        "发现ZINC远程文件:",
        len(urls)
    )

    if len(urls) == 0:

        raise RuntimeError(
            "没有从URI文件中找到HTTP或HTTPS下载地址。\n"
            "请确认下载方式选择的是Raw URLs。"
        )

    for index, url in enumerate(
        urls,
        start=1
    ):

        filename = get_filename_from_url(
            url
        )

        task_id = get_task_id(
            url
        )

        print(
            f"\n进度: {index}/{len(urls)}"
        )

        if task_id in processed:

            print(
                "跳过:",
                filename
            )

            continue

        retry = 0

        while retry < MAX_RETRY:

            try:

                process_url(url)

                break

            except Exception as e:

                conn.rollback()

                retry += 1

                print(
                    f"{filename} 失败，"
                    f"第 {retry} 次重试"
                )

                print(
                    type(e).__name__,
                    ":",
                    e
                )

                if retry < MAX_RETRY:

                    time.sleep(10)

        else:

            print(
                f"{filename} 连续失败"
                f"{MAX_RETRY}次，跳过"
            )

finally:

    conn.close()


print("===================")

print("全部完成")

print(
    "数据库:",
    DB_PATH
)

print(
    "处理记录:",
    DONE_FILE
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
