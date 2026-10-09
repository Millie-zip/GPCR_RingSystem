import sqlite3
import csv
import os
import random
from collections import defaultdict, Counter

DB = "/share/home/weiliao02/lyx/zinc_project/5ht2c_3d_diffusion_library_v1/5ht2c_3d_diffusion_library_v1.db"
OUTDIR = "/share/home/weiliao02/lyx/zinc_project/rascore_pilot_v1"
OUTCSV = os.path.join(OUTDIR, "rascore_pilot_candidates_10000.csv")
SUMMARY = os.path.join(OUTDIR, "rascore_pilot_summary.txt")

TARGET = 10000
SEED = 20260831
rng = random.Random(SEED)

# chemical scaffold 数量区间、计划选多少 topology、每个 topology 取多少 molecule
PLAN = [
    ("chem_1",       1,   1,    900,  1),
    ("chem_2_5",     2,   5,    900,  2),
    ("chem_6_10",    6,   10,   400,  3),
    ("chem_11_50",   11,  50,   400,  5),
    ("chem_51_100",  51,  100,  200,  7),
    ("chem_101_500", 101, 500,  200, 10),
    ("chem_501plus", 501, None,   68, 10),
]

os.makedirs(OUTDIR, exist_ok=True)

con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
cur = con.cursor()

print("1. Loading topologies...")

topologies = {}
for uid, smi, chem_n, inst_n in cur.execute("""
SELECT
    topology_uid,
    topology_smiles,
    strict_chemical_count,
    strict_instance_count
FROM topologies
"""):
    topologies[uid] = {
        "topology_smiles": smi,
        "strict_chemical_count": int(chem_n),
        "strict_instance_count": int(inst_n),
    }

print("   topologies:", len(topologies))

print("2. Loading chemical scaffolds...")

scaffolds_by_topology = defaultdict(list)

for chem_uid, topo_uid, chem_smi, inst_n in cur.execute("""
SELECT
    chemical_uid,
    topology_uid,
    chemical_smiles,
    strict_instance_count
FROM selected_scaffolds
WHERE prefilter_status = 'PASS'
"""):
    if topo_uid in topologies:
        scaffolds_by_topology[topo_uid].append(
            (
                chem_uid,
                chem_smi,
                int(inst_n) if inst_n is not None else 0
            )
        )

print("   topologies with scaffolds:", len(scaffolds_by_topology))


def get_one_molecule(chemical_uid, used_molecules):
    rows = cur.execute("""
    SELECT
        i.source_part,
        i.molecule_uid,
        m.original_smiles,
        m.canonical_isomeric_smiles,
        m.canonical_nonisomeric_smiles,
        m.molecular_formula,
        m.heavy_atom_count,
        m.formal_charge,
        m.calculated_exact_mw
    FROM selected_scaffold_instances i
    JOIN selected_molecules m
      ON m.source_part = i.source_part
     AND m.molecule_uid = i.molecule_uid
    WHERE i.chemical_uid = ?
    LIMIT 100
    """, (chemical_uid,)).fetchall()

    rng.shuffle(rows)

    for r in rows:
        molecule_uid = r[1]
        iso_smiles = r[3]

        if molecule_uid in used_molecules:
            continue

        if iso_smiles is None or not iso_smiles.strip():
            continue

        return r

    return None


def sample_topology(topology_uid, wanted_n, used_molecules, used_pairs):
    scaffolds = list(scaffolds_by_topology.get(topology_uid, []))
    rng.shuffle(scaffolds)

    results = []

    for chem_uid, chem_smiles, chem_instances in scaffolds:

        pair = (topology_uid, chem_uid)

        if pair in used_pairs:
            continue

        mol = get_one_molecule(chem_uid, used_molecules)

        if mol is None:
            continue

        row = {
            "topology_uid": topology_uid,
            "topology_smiles": topologies[topology_uid]["topology_smiles"],
            "strict_chemical_count": topologies[topology_uid]["strict_chemical_count"],
            "strict_instance_count": topologies[topology_uid]["strict_instance_count"],

            "chemical_uid": chem_uid,
            "chemical_smiles": chem_smiles,
            "chemical_instance_count": chem_instances,

            "source_part": mol[0],
            "molecule_uid": mol[1],
            "original_smiles": mol[2],
            "canonical_isomeric_smiles": mol[3],
            "canonical_nonisomeric_smiles": mol[4],
            "molecular_formula": mol[5],
            "heavy_atom_count": mol[6],
            "formal_charge": mol[7],
            "calculated_exact_mw": mol[8],
        }

        results.append(row)

        used_molecules.add(mol[1])
        used_pairs.add(pair)

        if len(results) >= wanted_n:
            break

    return results


selected = []
used_molecules = set()
used_pairs = set()
selected_topologies = set()

print("3. Main stratified sampling...")

for stratum, lo, hi, n_topologies, depth in PLAN:

    pool = []

    for uid, info in topologies.items():

        if uid not in scaffolds_by_topology:
            continue

        nchem = info["strict_chemical_count"]

        if nchem < lo:
            continue

        if hi is not None and nchem > hi:
            continue

        pool.append(uid)

    rng.shuffle(pool)

    chosen = pool[:min(n_topologies, len(pool))]

    before = len(selected)

    for uid in chosen:

        selected_topologies.add(uid)

        rows = sample_topology(
            uid,
            depth,
            used_molecules,
            used_pairs
        )

        for row in rows:
            row["stratum"] = stratum
            row["selection_stage"] = "main"
            selected.append(row)

    print(
        f"   {stratum}: "
        f"topologies={len(chosen)}, "
        f"molecules={len(selected)-before}"
    )


print("4. Main sample size:", len(selected))

# 不足10000时，优先从已选择的高多样性 topology 补不同 chemical scaffold
if len(selected) < TARGET:

    print("5. Filling from high-diversity topologies...")

    fill_topologies = sorted(
        selected_topologies,
        key=lambda x: topologies[x]["strict_chemical_count"],
        reverse=True
    )

    progress = True

    while len(selected) < TARGET and progress:

        progress = False

        for uid in fill_topologies:

            if len(selected) >= TARGET:
                break

            rows = sample_topology(
                uid,
                1,
                used_molecules,
                used_pairs
            )

            if rows:
                rows[0]["stratum"] = "fill_high_diversity"
                rows[0]["selection_stage"] = "fill"
                selected.append(rows[0])
                progress = True


# 如果还不够，从未选择 topology 中补新 topology
if len(selected) < TARGET:

    print("6. Filling from new topologies...")

    remaining = [
        uid for uid in topologies
        if uid in scaffolds_by_topology
        and uid not in selected_topologies
    ]

    rng.shuffle(remaining)

    for uid in remaining:

        if len(selected) >= TARGET:
            break

        rows = sample_topology(
            uid,
            1,
            used_molecules,
            used_pairs
        )

        if rows:
            selected_topologies.add(uid)
            rows[0]["stratum"] = "fill_new_topology"
            rows[0]["selection_stage"] = "fill"
            selected.append(rows[0])


selected = selected[:TARGET]

print("7. Writing CSV...")

fields = [
    "sample_index",
    "selection_stage",
    "stratum",
    "topology_uid",
    "topology_smiles",
    "strict_chemical_count",
    "strict_instance_count",
    "chemical_uid",
    "chemical_smiles",
    "chemical_instance_count",
    "source_part",
    "molecule_uid",
    "original_smiles",
    "canonical_isomeric_smiles",
    "canonical_nonisomeric_smiles",
    "molecular_formula",
    "heavy_atom_count",
    "formal_charge",
    "calculated_exact_mw",
]

with open(OUTCSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()

    for i, row in enumerate(selected, 1):
        writer.writerow({
            "sample_index": i,
            **row
        })


unique_molecules = len(set(x["molecule_uid"] for x in selected))
unique_topologies = len(set(x["topology_uid"] for x in selected))
unique_chemicals = len(
    set((x["topology_uid"], x["chemical_uid"]) for x in selected)
)

counts = Counter(x["stratum"] for x in selected)

lines = [
    "RAscore pilot sampling summary",
    "=" * 60,
    f"seed: {SEED}",
    f"target: {TARGET}",
    f"rows_written: {len(selected)}",
    f"unique_molecules: {unique_molecules}",
    f"unique_topologies: {unique_topologies}",
    f"unique_chemical_scaffolds: {unique_chemicals}",
    f"duplicate_molecule_uid: {len(selected) - unique_molecules}",
    f"missing_isomeric_smiles: {sum(1 for x in selected if not x['canonical_isomeric_smiles'])}",
    "",
    "Molecules by stratum:"
]

for k, v in sorted(counts.items()):
    lines.append(f"  {k}: {v}")

with open(SUMMARY, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")

print()
print("\n".join(lines))
print()
print("CSV:")
print(OUTCSV)
print()
print("SUMMARY:")
print(SUMMARY)

con.close()
