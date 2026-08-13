import os
import gzip
import sqlite3
import requests
import tempfile
import time


from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit import RDLogger


from bs4 import BeautifulSoup



# ================================
# 关闭RDKit大量warning
# ================================

RDLogger.DisableLog('rdApp.*')



# ================================
# 输出目录
# ================================


OUTPUT_DIR = r"D:\1match\Pubchem"


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


DB_PATH=os.path.join(
    OUTPUT_DIR,
    "PubChem_Fused_Ring.db"
)


DONE_FILE=os.path.join(
    OUTPUT_DIR,
    "processed_files.txt"
)



# ================================
# PubChem目录
# ================================


FTP_URL = (
"https://ftp.ncbi.nlm.nih.gov/"
"pubchem/Compound/CURRENT-Full/SDF/"
)



# ================================
# 参数
# ================================


MAX_MW=500

MAX_HEAVY=50



# ================================
# 创建数据库
# ================================


conn=sqlite3.connect(DB_PATH)

cursor=conn.cursor()


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

    with open(DONE_FILE,"r") as f:

        processed=set(
            x.strip()
            for x in f.readlines()
        )

else:

    processed=set()



# ================================
# 获取PubChem文件列表
# ================================


def get_pubchem_files():


    html=requests.get(
        FTP_URL,
        timeout=60
    ).text


    soup=BeautifulSoup(
        html,
        "html.parser"
    )


    files=[]


    for a in soup.find_all("a"):

        name=a.text


        if name.endswith(".sdf.gz"):

            files.append(name)


    return files



# ================================
# 判断稠环
# ================================


def get_fused_ring_info(mol):


    rings=mol.GetRingInfo().AtomRings()


    fused_pairs=0

    fused_atoms=set()



    for i in range(len(rings)):


        for j in range(i+1,len(rings)):


            r1=set(rings[i])

            r2=set(rings[j])


            shared=r1.intersection(r2)



            if len(shared)<2:

                continue



            shared=list(shared)


            adjacent=False



            for a in range(len(shared)):

                for b in range(a+1,len(shared)):

                    if mol.GetBondBetweenAtoms(
                        shared[a],
                        shared[b]
                    ):

                        adjacent=True



            if adjacent:

                fused_pairs+=1

                fused_atoms.update(shared)



    return fused_pairs,fused_atoms



# ================================
# 处理一个文件
# ================================


def process_file(filename):


    url=FTP_URL+filename


    print("\n开始:")
    print(filename)



    temp_file=os.path.join(
        tempfile.gettempdir(),
        filename
    )



    # 下载

    with requests.get(
        url,
        stream=True,
        timeout=300
    ) as r:


        r.raise_for_status()


        with open(
            temp_file,
            "wb"
        ) as f:


            for chunk in r.iter_content(
                chunk_size=1024*1024
            ):

                f.write(chunk)



    print("下载完成")



    count=0

    fused_count=0


    batch=[]



    with gzip.open(
        temp_file,
        "rb"
    ) as gz:


        supplier=Chem.ForwardSDMolSupplier(
            gz
        )



        for mol in supplier:


            if mol is None:

                continue



            count+=1



            mw=Descriptors.MolWt(mol)

            heavy=mol.GetNumHeavyAtoms()



            if mw>MAX_MW or heavy>MAX_HEAVY:

                continue



            ring_num=len(
                mol.GetRingInfo().AtomRings()
            )


            if ring_num<2:

                continue



            fused_pairs,fused_atoms=get_fused_ring_info(mol)



            if fused_pairs<1:

                continue



            fused_count+=1



            cid=""

            if "PUBCHEM_COMPOUND_CID" in mol.GetPropNames():

                cid=mol.GetProp(
                    "PUBCHEM_COMPOUND_CID"
                )


            batch.append(
                (
                    cid,
                    Chem.MolToSmiles(mol),
                    round(mw,2),
                    heavy,
                    ring_num,
                    fused_pairs,
                    len(fused_atoms)
                )
            )



            if len(batch)>=500:


                cursor.executemany(
                """
                INSERT INTO fused_ring_molecules
                VALUES
                (NULL,?,?,?,?,?,?,?)
                """,
                batch
                )


                conn.commit()


                batch=[]



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



    os.remove(temp_file)



    with open(
        DONE_FILE,
        "a"
    ) as f:

        f.write(filename+"\n")



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


start=time.time()



files=get_pubchem_files()


print(
    "发现文件:",
    len(files)
)



for file in files:


    if file in processed:

        print(
            "跳过:",
            file
        )

        continue



    retry = 0


    while retry < 5:


        try:

            process_file(file)

            break


        except Exception as e:


            retry += 1


            print(
                f"{file} 失败，第 {retry} 次重试"
            )

            print(e)



            time.sleep(10)



    else:


        print(
            f"{file} 连续失败5次，跳过"
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
    round(time.time()-start,2),
    "秒"
)

print("===================")