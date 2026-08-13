#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
global_scaffold_merge_v4_4_no_weights.py

Corrected final merge for the 120 validated V4 part-result DBs.

Special handling:
- 118 normal parts: copy existing part_audit row exactly.
- part_0116 / part_0119:
    * MUST pass V4.1 topology graph-equivalence validation.
    * if part_audit is missing, reconstruct the audit counts ONLY from the
      already-produced part-result DB.
    * do NOT modify the part-result DB.
    * elapsed_seconds is stored as NULL because the old worker aborted before
      normal audit finalization, so an exact elapsed time is unavailable.

NO three-level weights are calculated.
"""

import argparse
import csv
import json
import os
import re
import shutil
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path

from rdkit import Chem

from gpcr_ringgen.preprocessing.paths import data_root


PROJECT = data_root()
RUN_DIR = PROJECT / "global_scaffold_analysis_strict_full_v4"
PART_ROOT = RUN_DIR / "reports" / "_part_results"
VALIDATION_ROOT = PART_ROOT / "validation"
EXPECTED_PARTS = [f"part_{i:04d}" for i in range(120)]
SPECIAL_PARTS = {"part_0116", "part_0119"}


def log(msg=""):
    print(msg, flush=True)


def q(name):
    return '"' + name.replace('"', '""') + '"'


# ---------------------------------------------------------------------------
# V4.1 validation
# ---------------------------------------------------------------------------

def normalize_topology(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None

    rw = Chem.RWMol(mol)

    for atom in rw.GetAtoms():
        atom.SetAtomicNum(6)
        atom.SetFormalCharge(0)
        atom.SetIsotope(0)
        atom.SetNumExplicitHs(0)
        atom.SetNoImplicit(False)
        atom.SetNumRadicalElectrons(0)
        atom.SetIsAromatic(False)

    for bond in rw.GetBonds():
        bond.SetBondType(Chem.BondType.SINGLE)
        bond.SetIsAromatic(False)

    nmol = rw.GetMol()

    try:
        Chem.SanitizeMol(nmol)
    except Exception:
        return None, None

    canonical = Chem.MolToSmiles(
        nmol,
        canonical=True,
        isomericSmiles=False,
    )
    return nmol, canonical


def graph_equivalent(a, b):
    ma, ca = normalize_topology(a)
    mb, cb = normalize_topology(b)

    if ma is None or mb is None:
        return False
    if ma.GetNumAtoms() != mb.GetNumAtoms():
        return False
    if ma.GetNumBonds() != mb.GetNumBonds():
        return False
    if ca == cb:
        return True

    return (
        ma.HasSubstructMatch(mb)
        and mb.HasSubstructMatch(ma)
    )


def validate_special_part(part):
    d = VALIDATION_ROOT / part
    hits = sorted(d.glob("*topology_fallback_validation.json"))

    if not hits:
        raise RuntimeError(f"No validation JSON found for {part}")

    with open(hits[0], "r", encoding="utf-8") as f:
        data = json.load(f)

    n_validated = None
    for key in ("validated", "n_validated", "validation_count"):
        if key in data:
            n_validated = int(data[key])
            break

    exact = int(data.get("exact_matches", 0))
    graph_nonexact = 0
    true_bad = []

    for item in data.get("mismatches", []):
        stored = item.get("stored_topology")
        rebuilt = item.get("rebuilt_topology")

        if stored and rebuilt and graph_equivalent(stored, rebuilt):
            graph_nonexact += 1
        else:
            true_bad.append(item)

    passed = (
        n_validated is not None
        and n_validated > 0
        and exact + graph_nonexact == n_validated
        and len(true_bad) == 0
    )

    result = {
        "validation_file": str(hits[0]),
        "validated": n_validated,
        "exact_matches": exact,
        "graph_equivalent_nonexact_matches": graph_nonexact,
        "true_graph_mismatches": len(true_bad),
        "pass": passed,
    }

    if not passed:
        raise RuntimeError(
            f"{part} failed V4.1 topology validation: {result}"
        )

    return result


# ---------------------------------------------------------------------------
# Resolve and validate part-result DBs
# ---------------------------------------------------------------------------

def discover_part_dbs():
    found = defaultdict(list)

    for p in PART_ROOT.rglob("*.db"):
        m = re.search(r"(part_\d{4})", p.name)
        if m and m.group(1) in EXPECTED_PARTS:
            found[m.group(1)].append(p.resolve())

    resolved = {}

    for part in EXPECTED_PARTS:
        candidates = found.get(part, [])

        exact = [
            p for p in candidates
            if p.name == f"{part}_result.db"
        ]

        if len(exact) == 1:
            resolved[part] = exact[0]
            continue

        candidates = [
            p for p in candidates
            if "validation" not in str(p).lower()
        ]

        if len(candidates) != 1:
            raise RuntimeError(
                f"Cannot uniquely resolve {part}: "
                + " | ".join(map(str, candidates))
            )

        resolved[part] = candidates[0]

    return resolved


EXPECTED_SCHEMA = {
    "topologies": [
        "topology_uid",
        "topology_smiles",
        "strict_chemical_count",
        "strict_instance_count",
        "unresolved_instance_count",
        "topology_weight",
    ],
    "chemical_scaffolds": [
        "chemical_uid",
        "topology_uid",
        "chemical_smiles",
        "strict_instance_count",
        "chemical_weight",
    ],
    "strict_scaffold_instances": [
        "source_part",
        "molecule_uid",
        "ring_system_uid",
        "system_index",
        "sssr_basis_ring_count",
        "cycle_rank",
        "ring_system_type",
        "topology_uid",
        "chemical_uid",
        "instance_weight",
    ],
    "unresolved_ring_systems": [
        "source_part",
        "molecule_uid",
        "ring_system_uid",
        "system_index",
        "sssr_basis_ring_count",
        "cycle_rank",
        "ring_system_type",
        "topology_uid",
        "topology_smiles",
        "chemical_status",
        "original_failure_stage",
        "original_error_type",
        "original_error_message",
    ],
    "part_audit": [
        "source_part",
        "eligible_molecules",
        "eligible_ring_systems",
        "strict_complete_pairs",
        "unresolved_pure_scaffolds",
        "unresolved_with_topology",
        "unresolved_without_topology",
        "elapsed_seconds",
    ],
    "run_metadata": [
        "key",
        "value",
    ],
}


def check_part_db(path):
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)

    try:
        qc = con.execute("PRAGMA quick_check").fetchall()
        if not (len(qc) == 1 and qc[0][0] == "ok"):
            raise RuntimeError(
                f"quick_check failed for {path}: {qc}"
            )

        for table, expected_cols in EXPECTED_SCHEMA.items():
            rows = con.execute(
                f"PRAGMA table_info({q(table)})"
            ).fetchall()

            cols = [r[1] for r in rows]

            if cols != expected_cols:
                raise RuntimeError(
                    f"Schema mismatch in {path}, table={table}\n"
                    f"expected={expected_cols}\n"
                    f"actual={cols}"
                )

    finally:
        con.close()


# ---------------------------------------------------------------------------
# Final global schema
# ---------------------------------------------------------------------------

def create_global_schema(con):
    # elapsed_seconds is nullable in the GLOBAL audit table because the two
    # interrupted workers did not finish their original audit write.
    con.executescript("""
    CREATE TABLE topologies (
        topology_uid TEXT PRIMARY KEY,
        topology_smiles TEXT NOT NULL UNIQUE,
        strict_chemical_count INTEGER NOT NULL DEFAULT 0,
        strict_instance_count INTEGER NOT NULL DEFAULT 0,
        unresolved_instance_count INTEGER NOT NULL DEFAULT 0,
        topology_weight REAL
    );

    CREATE TABLE chemical_scaffolds (
        chemical_uid TEXT PRIMARY KEY,
        topology_uid TEXT NOT NULL,
        chemical_smiles TEXT NOT NULL,
        strict_instance_count INTEGER NOT NULL DEFAULT 0,
        chemical_weight REAL,
        UNIQUE(topology_uid, chemical_smiles)
    );

    CREATE TABLE strict_scaffold_instances (
        source_part TEXT NOT NULL,
        molecule_uid TEXT NOT NULL,
        ring_system_uid TEXT NOT NULL,
        system_index INTEGER NOT NULL,
        sssr_basis_ring_count INTEGER NOT NULL,
        cycle_rank INTEGER NOT NULL,
        ring_system_type TEXT NOT NULL,
        topology_uid TEXT NOT NULL,
        chemical_uid TEXT NOT NULL,
        instance_weight REAL,
        PRIMARY KEY(source_part, ring_system_uid)
    );

    CREATE TABLE unresolved_ring_systems (
        source_part TEXT NOT NULL,
        molecule_uid TEXT NOT NULL,
        ring_system_uid TEXT NOT NULL,
        system_index INTEGER NOT NULL,
        sssr_basis_ring_count INTEGER NOT NULL,
        cycle_rank INTEGER NOT NULL,
        ring_system_type TEXT NOT NULL,
        topology_uid TEXT,
        topology_smiles TEXT,
        chemical_status TEXT NOT NULL,
        original_failure_stage TEXT,
        original_error_type TEXT,
        original_error_message TEXT,
        PRIMARY KEY(source_part, ring_system_uid)
    );

    CREATE TABLE part_audit (
        source_part TEXT PRIMARY KEY,
        eligible_molecules INTEGER NOT NULL,
        eligible_ring_systems INTEGER NOT NULL,
        strict_complete_pairs INTEGER NOT NULL,
        unresolved_pure_scaffolds INTEGER NOT NULL,
        unresolved_with_topology INTEGER NOT NULL,
        unresolved_without_topology INTEGER NOT NULL,
        elapsed_seconds REAL
    );

    CREATE TABLE run_metadata (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    );
    """)

    con.commit()


# ---------------------------------------------------------------------------
# Dictionary merge with collision checks
# ---------------------------------------------------------------------------

def ensure_topology(out, topology_uid, topology_smiles, source_part):
    row = out.execute(
        """
        SELECT topology_smiles
        FROM topologies
        WHERE topology_uid=?
        """,
        (topology_uid,),
    ).fetchone()

    if row is None:
        reverse = out.execute(
            """
            SELECT topology_uid
            FROM topologies
            WHERE topology_smiles=?
            """,
            (topology_smiles,),
        ).fetchone()

        if reverse is not None and reverse[0] != topology_uid:
            raise RuntimeError(
                "Topology identity conflict:\n"
                f"smiles={topology_smiles}\n"
                f"existing_uid={reverse[0]}\n"
                f"new_uid={topology_uid}\n"
                f"source={source_part}"
            )

        out.execute(
            """
            INSERT INTO topologies(
                topology_uid,
                topology_smiles,
                strict_chemical_count,
                strict_instance_count,
                unresolved_instance_count,
                topology_weight
            )
            VALUES (?, ?, 0, 0, 0, NULL)
            """,
            (topology_uid, topology_smiles),
        )

    elif row[0] != topology_smiles:
        raise RuntimeError(
            "Topology UID collision:\n"
            f"uid={topology_uid}\n"
            f"existing_smiles={row[0]}\n"
            f"new_smiles={topology_smiles}\n"
            f"source={source_part}"
        )


def ensure_chemical(
    out,
    chemical_uid,
    topology_uid,
    chemical_smiles,
    source_part,
):
    row = out.execute(
        """
        SELECT topology_uid, chemical_smiles
        FROM chemical_scaffolds
        WHERE chemical_uid=?
        """,
        (chemical_uid,),
    ).fetchone()

    if row is None:
        reverse = out.execute(
            """
            SELECT chemical_uid
            FROM chemical_scaffolds
            WHERE topology_uid=? AND chemical_smiles=?
            """,
            (topology_uid, chemical_smiles),
        ).fetchone()

        if reverse is not None and reverse[0] != chemical_uid:
            raise RuntimeError(
                "Chemical identity conflict:\n"
                f"topology_uid={topology_uid}\n"
                f"chemical_smiles={chemical_smiles}\n"
                f"existing_uid={reverse[0]}\n"
                f"new_uid={chemical_uid}\n"
                f"source={source_part}"
            )

        out.execute(
            """
            INSERT INTO chemical_scaffolds(
                chemical_uid,
                topology_uid,
                chemical_smiles,
                strict_instance_count,
                chemical_weight
            )
            VALUES (?, ?, ?, 0, NULL)
            """,
            (
                chemical_uid,
                topology_uid,
                chemical_smiles,
            ),
        )

    elif row != (topology_uid, chemical_smiles):
        raise RuntimeError(
            "Chemical UID collision:\n"
            f"uid={chemical_uid}\n"
            f"existing={row}\n"
            f"new={(topology_uid, chemical_smiles)}\n"
            f"source={source_part}"
        )


def merge_dictionary_tables(out, src, source_part):
    for (
        topology_uid,
        topology_smiles,
        _strict_chemical_count,
        _strict_instance_count,
        _unresolved_instance_count,
        _topology_weight,
    ) in src.execute("SELECT * FROM topologies"):
        ensure_topology(
            out,
            topology_uid,
            topology_smiles,
            source_part,
        )

    for (
        chemical_uid,
        topology_uid,
        chemical_smiles,
        _strict_instance_count,
        _chemical_weight,
    ) in src.execute("SELECT * FROM chemical_scaffolds"):

        if out.execute(
            """
            SELECT 1
            FROM topologies
            WHERE topology_uid=?
            """,
            (topology_uid,),
        ).fetchone() is None:
            raise RuntimeError(
                f"Chemical references missing topology {topology_uid} "
                f"in {source_part}"
            )

        ensure_chemical(
            out,
            chemical_uid,
            topology_uid,
            chemical_smiles,
            source_part,
        )


# ---------------------------------------------------------------------------
# Instance merge
# ---------------------------------------------------------------------------

def merge_strict_instances(out, src):
    sql = """
    INSERT INTO strict_scaffold_instances(
        source_part,
        molecule_uid,
        ring_system_uid,
        system_index,
        sssr_basis_ring_count,
        cycle_rank,
        ring_system_type,
        topology_uid,
        chemical_uid,
        instance_weight
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
    """

    cur = src.execute("""
        SELECT
            source_part,
            molecule_uid,
            ring_system_uid,
            system_index,
            sssr_basis_ring_count,
            cycle_rank,
            ring_system_type,
            topology_uid,
            chemical_uid
        FROM strict_scaffold_instances
    """)

    n = 0

    while True:
        batch = cur.fetchmany(25000)
        if not batch:
            break

        out.executemany(sql, batch)
        n += len(batch)

    return n


def merge_unresolved(out, src):
    sql = """
    INSERT INTO unresolved_ring_systems(
        source_part,
        molecule_uid,
        ring_system_uid,
        system_index,
        sssr_basis_ring_count,
        cycle_rank,
        ring_system_type,
        topology_uid,
        topology_smiles,
        chemical_status,
        original_failure_stage,
        original_error_type,
        original_error_message
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    cur = src.execute(
        "SELECT * FROM unresolved_ring_systems"
    )

    n = 0

    while True:
        batch = cur.fetchmany(25000)
        if not batch:
            break

        out.executemany(sql, batch)
        n += len(batch)

    return n


# ---------------------------------------------------------------------------
# part_audit handling
# ---------------------------------------------------------------------------

def reconstruct_part_audit(src, part):
    """
    Reconstruct the missing audit row from the ALREADY GENERATED result DB.

    No source DB is read.
    No scaffold is recomputed.
    No result DB is modified.
    """

    strict_complete_pairs = src.execute(
        """
        SELECT COUNT(*)
        FROM strict_scaffold_instances
        """
    ).fetchone()[0]

    unresolved_pure_scaffolds = src.execute(
        """
        SELECT COUNT(*)
        FROM unresolved_ring_systems
        """
    ).fetchone()[0]

    unresolved_with_topology = src.execute(
        """
        SELECT COUNT(*)
        FROM unresolved_ring_systems
        WHERE topology_uid IS NOT NULL
          AND topology_smiles IS NOT NULL
        """
    ).fetchone()[0]

    unresolved_without_topology = (
        unresolved_pure_scaffolds
        - unresolved_with_topology
    )

    eligible_ring_systems = (
        strict_complete_pairs
        + unresolved_pure_scaffolds
    )

    # DISTINCT eligible molecules across BOTH strict and unresolved tables.
    eligible_molecules = src.execute("""
        SELECT COUNT(DISTINCT molecule_uid)
        FROM (
            SELECT molecule_uid
            FROM strict_scaffold_instances

            UNION

            SELECT molecule_uid
            FROM unresolved_ring_systems
        )
    """).fetchone()[0]

    # Exact worker elapsed time is unavailable because the old worker aborted
    # before the normal audit-finalization step.
    elapsed_seconds = None

    return (
        f"{part}.db",
        eligible_molecules,
        eligible_ring_systems,
        strict_complete_pairs,
        unresolved_pure_scaffolds,
        unresolved_with_topology,
        unresolved_without_topology,
        elapsed_seconds,
    )


def copy_or_reconstruct_part_audit(
    out,
    src,
    part,
    audit_reconstruction_records,
):
    rows = src.execute(
        "SELECT * FROM part_audit"
    ).fetchall()

    if len(rows) == 1:
        out.execute(
            """
            INSERT INTO part_audit(
                source_part,
                eligible_molecules,
                eligible_ring_systems,
                strict_complete_pairs,
                unresolved_pure_scaffolds,
                unresolved_with_topology,
                unresolved_without_topology,
                elapsed_seconds
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows[0],
        )
        return "copied"

    if len(rows) > 1:
        raise RuntimeError(
            f"{part}: expected <=1 part_audit row, got {len(rows)}"
        )

    if part not in SPECIAL_PARTS:
        raise RuntimeError(
            f"{part}: missing part_audit row, but only "
            f"{sorted(SPECIAL_PARTS)} are allowed reconstruction"
        )

    rebuilt = reconstruct_part_audit(src, part)

    out.execute(
        """
        INSERT INTO part_audit(
            source_part,
            eligible_molecules,
            eligible_ring_systems,
            strict_complete_pairs,
            unresolved_pure_scaffolds,
            unresolved_with_topology,
            unresolved_without_topology,
            elapsed_seconds
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rebuilt,
    )

    audit_reconstruction_records[part] = {
        "source_part": rebuilt[0],
        "eligible_molecules": rebuilt[1],
        "eligible_ring_systems": rebuilt[2],
        "strict_complete_pairs": rebuilt[3],
        "unresolved_pure_scaffolds": rebuilt[4],
        "unresolved_with_topology": rebuilt[5],
        "unresolved_without_topology": rebuilt[6],
        "elapsed_seconds": None,
        "elapsed_seconds_reason": (
            "Unavailable because old worker aborted before audit finalization"
        ),
        "reconstructed_from": (
            "strict_scaffold_instances + unresolved_ring_systems"
        ),
    }

    return "reconstructed"


# ---------------------------------------------------------------------------
# Global counts / indexes / checks
# ---------------------------------------------------------------------------

def create_indexes_before_global_counts(out):
    out.executescript("""
    CREATE INDEX idx_chem_topology_uid
        ON chemical_scaffolds(topology_uid);

    CREATE INDEX idx_strict_topology_uid
        ON strict_scaffold_instances(topology_uid);

    CREATE INDEX idx_strict_chemical_uid
        ON strict_scaffold_instances(chemical_uid);

    CREATE INDEX idx_strict_source_molecule
        ON strict_scaffold_instances(source_part, molecule_uid);

    CREATE INDEX idx_unresolved_topology_uid
        ON unresolved_ring_systems(topology_uid);

    CREATE INDEX idx_unresolved_status
        ON unresolved_ring_systems(chemical_status);

    CREATE INDEX idx_unresolved_source_molecule
        ON unresolved_ring_systems(source_part, molecule_uid);
    """)

    out.commit()


def recompute_global_counts(out):
    # Explicitly preserve the "no weights yet" state.
    out.execute(
        "UPDATE topologies SET topology_weight=NULL"
    )
    out.execute(
        "UPDATE chemical_scaffolds SET chemical_weight=NULL"
    )
    out.execute(
        "UPDATE strict_scaffold_instances SET instance_weight=NULL"
    )

    # Temporary aggregation tables avoid repeatedly rescanning the 17.5M-row
    # instance table for every scaffold row.
    out.executescript("""
    DROP TABLE IF EXISTS temp.tmp_chem_counts;
    CREATE TEMP TABLE tmp_chem_counts AS
    SELECT chemical_uid, COUNT(*) AS n
    FROM strict_scaffold_instances
    GROUP BY chemical_uid;

    CREATE INDEX temp.idx_tmp_chem_counts_uid
        ON tmp_chem_counts(chemical_uid);

    DROP TABLE IF EXISTS temp.tmp_topo_strict_counts;
    CREATE TEMP TABLE tmp_topo_strict_counts AS
    SELECT topology_uid, COUNT(*) AS n
    FROM strict_scaffold_instances
    GROUP BY topology_uid;

    CREATE INDEX temp.idx_tmp_topo_strict_counts_uid
        ON tmp_topo_strict_counts(topology_uid);

    DROP TABLE IF EXISTS temp.tmp_topo_unresolved_counts;
    CREATE TEMP TABLE tmp_topo_unresolved_counts AS
    SELECT topology_uid, COUNT(*) AS n
    FROM unresolved_ring_systems
    WHERE topology_uid IS NOT NULL
    GROUP BY topology_uid;

    CREATE INDEX temp.idx_tmp_topo_unresolved_counts_uid
        ON tmp_topo_unresolved_counts(topology_uid);

    DROP TABLE IF EXISTS temp.tmp_topo_chemical_counts;
    CREATE TEMP TABLE tmp_topo_chemical_counts AS
    SELECT topology_uid, COUNT(*) AS n
    FROM chemical_scaffolds
    GROUP BY topology_uid;

    CREATE INDEX temp.idx_tmp_topo_chemical_counts_uid
        ON tmp_topo_chemical_counts(topology_uid);
    """)

    out.execute("""
        UPDATE chemical_scaffolds
        SET strict_instance_count = COALESCE(
            (
                SELECT n
                FROM tmp_chem_counts x
                WHERE x.chemical_uid =
                    chemical_scaffolds.chemical_uid
            ),
            0
        )
    """)

    out.execute("""
        UPDATE topologies
        SET strict_instance_count = COALESCE(
            (
                SELECT n
                FROM tmp_topo_strict_counts x
                WHERE x.topology_uid =
                    topologies.topology_uid
            ),
            0
        )
    """)

    out.execute("""
        UPDATE topologies
        SET unresolved_instance_count = COALESCE(
            (
                SELECT n
                FROM tmp_topo_unresolved_counts x
                WHERE x.topology_uid =
                    topologies.topology_uid
            ),
            0
        )
    """)

    out.execute("""
        UPDATE topologies
        SET strict_chemical_count = COALESCE(
            (
                SELECT n
                FROM tmp_topo_chemical_counts x
                WHERE x.topology_uid =
                    topologies.topology_uid
            ),
            0
        )
    """)

    out.commit()


def scalar(out, sql):
    return out.execute(sql).fetchone()[0]


def final_checks(out):
    checks = {}

    checks["strict_missing_topology"] = scalar(out, """
        SELECT COUNT(*)
        FROM strict_scaffold_instances i
        LEFT JOIN topologies t
          ON t.topology_uid=i.topology_uid
        WHERE t.topology_uid IS NULL
    """)

    checks["strict_missing_chemical"] = scalar(out, """
        SELECT COUNT(*)
        FROM strict_scaffold_instances i
        LEFT JOIN chemical_scaffolds c
          ON c.chemical_uid=i.chemical_uid
        WHERE c.chemical_uid IS NULL
    """)

    checks["chemical_missing_topology"] = scalar(out, """
        SELECT COUNT(*)
        FROM chemical_scaffolds c
        LEFT JOIN topologies t
          ON t.topology_uid=c.topology_uid
        WHERE t.topology_uid IS NULL
    """)


    checks["nonnull_topology_weights"] = scalar(out, """
        SELECT COUNT(*)
        FROM topologies
        WHERE topology_weight IS NOT NULL
    """)

    checks["nonnull_chemical_weights"] = scalar(out, """
        SELECT COUNT(*)
        FROM chemical_scaffolds
        WHERE chemical_weight IS NOT NULL
    """)

    checks["nonnull_instance_weights"] = scalar(out, """
        SELECT COUNT(*)
        FROM strict_scaffold_instances
        WHERE instance_weight IS NOT NULL
    """)

    for key in (
        "strict_missing_topology",
        "strict_missing_chemical",
        "chemical_missing_topology",
        "nonnull_topology_weights",
        "nonnull_chemical_weights",
        "nonnull_instance_weights",
    ):
        if checks[key] != 0:
            raise RuntimeError(
                f"Final consistency check failed: {key}={checks[key]}"
            )

    checks["part_audit_rows"] = scalar(
        out,
        "SELECT COUNT(*) FROM part_audit"
    )

    if checks["part_audit_rows"] != 120:
        raise RuntimeError(
            f"Expected 120 global part_audit rows, "
            f"got {checks['part_audit_rows']}"
        )

    checks["audit_strict_sum"] = scalar(out, """
        SELECT SUM(strict_complete_pairs)
        FROM part_audit
    """)

    checks["strict_instance_rows"] = scalar(out, """
        SELECT COUNT(*)
        FROM strict_scaffold_instances
    """)

    checks["audit_unresolved_sum"] = scalar(out, """
        SELECT SUM(unresolved_pure_scaffolds)
        FROM part_audit
    """)

    checks["unresolved_rows"] = scalar(out, """
        SELECT COUNT(*)
        FROM unresolved_ring_systems
    """)

    if (
        checks["audit_strict_sum"]
        != checks["strict_instance_rows"]
    ):
        raise RuntimeError(
            "Strict instance total does not match part_audit sum: "
            f"{checks['strict_instance_rows']} vs "
            f"{checks['audit_strict_sum']}"
        )

    if (
        checks["audit_unresolved_sum"]
        != checks["unresolved_rows"]
    ):
        raise RuntimeError(
            "Unresolved total does not match part_audit sum: "
            f"{checks['unresolved_rows']} vs "
            f"{checks['audit_unresolved_sum']}"
        )

    # V4.4: genuine topology-fallback failures are allowed, but they must
    # close exactly against the part-level audit. We preserve them as
    # unresolved_without_topology and never fabricate topology.
    checks["actual_unresolved_without_topology"] = scalar(out, """
        SELECT COUNT(*)
        FROM unresolved_ring_systems
        WHERE topology_uid IS NULL
           OR topology_smiles IS NULL
    """)

    checks["audit_unresolved_without_topology_sum"] = scalar(out, """
        SELECT SUM(unresolved_without_topology)
        FROM part_audit
    """)

    if (
        checks["actual_unresolved_without_topology"]
        != checks["audit_unresolved_without_topology_sum"]
    ):
        raise RuntimeError(
            "unresolved_without_topology audit mismatch: "
            f"actual={checks['actual_unresolved_without_topology']} vs "
            f"audit={checks['audit_unresolved_without_topology_sum']}"
        )

    checks["actual_unresolved_with_topology"] = scalar(out, """
        SELECT COUNT(*)
        FROM unresolved_ring_systems
        WHERE topology_uid IS NOT NULL
          AND topology_smiles IS NOT NULL
    """)

    checks["audit_unresolved_with_topology_sum"] = scalar(out, """
        SELECT SUM(unresolved_with_topology)
        FROM part_audit
    """)

    if (
        checks["actual_unresolved_with_topology"]
        != checks["audit_unresolved_with_topology_sum"]
    ):
        raise RuntimeError(
            "unresolved_with_topology audit mismatch: "
            f"actual={checks['actual_unresolved_with_topology']} vs "
            f"audit={checks['audit_unresolved_with_topology_sum']}"
        )

    return checks


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def export_query(out, query, path):
    cur = out.execute(query)
    headers = [d[0] for d in cur.description]

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        w = csv.writer(f)
        w.writerow(headers)

        while True:
            batch = cur.fetchmany(25000)
            if not batch:
                break
            w.writerows(batch)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--out-dir",
        default=str(RUN_DIR),
    )

    ap.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Only needed if a completed global_scaffold_index.db "
            "already exists and should be replaced."
        ),
    )

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    final_db = out_dir / "global_scaffold_index.db"
    staging_db = out_dir / "global_scaffold_index.db.staging"

    if final_db.exists():
        if not args.overwrite:
            raise RuntimeError(
                f"{final_db} already exists. "
                f"Do not overwrite unintentionally."
            )

        backup = (
            out_dir
            / f"global_scaffold_index.db.backup_{int(time.time())}"
        )
        shutil.move(final_db, backup)
        log(f"Existing final DB backed up to: {backup}")

    # Previous failed merge staging file is disposable.
    if staging_db.exists():
        log(f"Removing old incomplete staging DB: {staging_db}")
        staging_db.unlink()

    t0 = time.time()

    log("=" * 100)
    log("GLOBAL SCAFFOLD MERGE V4.4 — NO WEIGHTS")
    log("=" * 100)
    log(
        "Special audit reconstruction is allowed ONLY for "
        "part_0116 and part_0119."
    )
    log(
        "Original molecular source DBs will NOT be opened."
    )
    log(
        "Three-level weights will NOT be calculated."
    )
    log()

    part_dbs = discover_part_dbs()

    log(
        f"Resolved part-result DBs: "
        f"{len(part_dbs)}/120"
    )

    for idx, part in enumerate(EXPECTED_PARTS, 1):
        check_part_db(part_dbs[part])

        if idx % 20 == 0:
            log(
                f"Preflight integrity/schema: "
                f"{idx}/120"
            )

    special_validation = {
        part: validate_special_part(part)
        for part in sorted(SPECIAL_PARTS)
    }

    for part, result in special_validation.items():
        log(
            f"{part} V4.1 validation: {result}"
        )

    log()

    out = sqlite3.connect(staging_db)

    try:
        out.execute(
            "PRAGMA journal_mode=DELETE"
        )
        out.execute(
            "PRAGMA synchronous=NORMAL"
        )
        out.execute(
            "PRAGMA temp_store=FILE"
        )
        out.execute(
            "PRAGMA cache_size=-200000"
        )

        create_global_schema(out)

        merged_strict = 0
        merged_unresolved = 0
        audit_reconstruction_records = {}

        for idx, part in enumerate(EXPECTED_PARTS, 1):
            p0 = time.time()

            log(
                f"[{idx:03d}/120] merging {part} ..."
            )

            src = sqlite3.connect(
                f"file:{part_dbs[part]}?mode=ro",
                uri=True,
            )

            try:
                out.execute("BEGIN")

                merge_dictionary_tables(
                    out,
                    src,
                    part,
                )

                merged_strict += merge_strict_instances(
                    out,
                    src,
                )

                merged_unresolved += merge_unresolved(
                    out,
                    src,
                )

                audit_mode = copy_or_reconstruct_part_audit(
                    out,
                    src,
                    part,
                    audit_reconstruction_records,
                )

                out.commit()

            except Exception:
                out.rollback()
                raise

            finally:
                src.close()

            log(
                f"    audit={audit_mode}; "
                f"done in {(time.time()-p0)/60:.2f} min"
            )

        log()
        log("Creating indexes ...")
        create_indexes_before_global_counts(out)

        log(
            "Recomputing GLOBAL counts "
            "(weights remain NULL) ..."
        )
        recompute_global_counts(out)

        log(
            "Running final global consistency checks ..."
        )
        checks = final_checks(out)

        metadata = {
            "merge_script":
                "global_scaffold_merge_v4_4_no_weights.py",
            "source_part_result_count":
                "120",
            "source_master_dbs_opened":
                "false",
            "part_0116_v4_1_validation":
                "pass",
            "part_0119_v4_1_validation":
                "pass",
            "part_0116_part_audit":
                (
                    "reconstructed_from_existing_part_result_db"
                    if "part_0116"
                    in audit_reconstruction_records
                    else "copied"
                ),
            "part_0119_part_audit":
                (
                    "reconstructed_from_existing_part_result_db"
                    if "part_0119"
                    in audit_reconstruction_records
                    else "copied"
                ),
            "reconstructed_audit_elapsed_seconds":
                "NULL_when_original_worker_elapsed_unavailable",
            "unresolved_without_topology_policy":
                "preserve_genuine_fallback_failures_no_fabrication",
            "unresolved_without_topology_count":
                str(checks["actual_unresolved_without_topology"]),
            "three_level_weights_calculated":
                "false",
            "weight_status":
                "DEFERRED_BY_USER_REQUEST",
        }

        out.executemany(
            """
            INSERT INTO run_metadata(key, value)
            VALUES (?, ?)
            """,
            list(metadata.items()),
        )

        out.commit()

        qc = out.execute(
            "PRAGMA quick_check"
        ).fetchall()

        if not (
            len(qc) == 1
            and qc[0][0] == "ok"
        ):
            raise RuntimeError(
                f"Final quick_check failed: {qc}"
            )

        final_counts = {
            "topologies": scalar(
                out,
                "SELECT COUNT(*) FROM topologies",
            ),
            "chemical_scaffolds": scalar(
                out,
                "SELECT COUNT(*) FROM chemical_scaffolds",
            ),
            "strict_scaffold_instances": scalar(
                out,
                "SELECT COUNT(*) FROM strict_scaffold_instances",
            ),
            "unresolved_ring_systems": scalar(
                out,
                "SELECT COUNT(*) FROM unresolved_ring_systems",
            ),
            "unresolved_with_topology":
                checks["actual_unresolved_with_topology"],
            "unresolved_without_topology":
                checks["actual_unresolved_without_topology"],
            "part_audit": scalar(
                out,
                "SELECT COUNT(*) FROM part_audit",
            ),
            "run_metadata": scalar(
                out,
                "SELECT COUNT(*) FROM run_metadata",
            ),
        }

        log("Writing CSV summaries ...")

        export_query(
            out,
            """
            SELECT
                topology_uid,
                topology_smiles,
                strict_chemical_count,
                strict_instance_count,
                unresolved_instance_count,
                topology_weight
            FROM topologies
            ORDER BY
                strict_instance_count DESC,
                unresolved_instance_count DESC,
                topology_uid
            """,
            out_dir / "topology_summary.csv",
        )

        export_query(
            out,
            """
            SELECT
                chemical_uid,
                topology_uid,
                chemical_smiles,
                strict_instance_count,
                chemical_weight
            FROM chemical_scaffolds
            ORDER BY
                strict_instance_count DESC,
                chemical_uid
            """,
            out_dir / "chemical_scaffold_summary.csv",
        )

        export_query(
            out,
            """
            SELECT
                CASE
                    WHEN topology_uid IS NULL
                      OR topology_smiles IS NULL
                    THEN 'without_topology'
                    ELSE 'with_topology'
                END AS topology_resolution,
                chemical_status,
                COALESCE(
                    original_failure_stage,
                    ''
                ) AS original_failure_stage,
                COALESCE(
                    original_error_type,
                    ''
                ) AS original_error_type,
                COUNT(*) AS unresolved_count
            FROM unresolved_ring_systems
            GROUP BY
                CASE
                    WHEN topology_uid IS NULL
                      OR topology_smiles IS NULL
                    THEN 'without_topology'
                    ELSE 'with_topology'
                END,
                chemical_status,
                COALESCE(
                    original_failure_stage,
                    ''
                ),
                COALESCE(
                    original_error_type,
                    ''
                )
            ORDER BY unresolved_count DESC
            """,
            out_dir / "unresolved_summary.csv",
        )

        export_query(
            out,
            """
            SELECT *
            FROM unresolved_ring_systems
            WHERE topology_uid IS NULL
               OR topology_smiles IS NULL
            ORDER BY source_part, ring_system_uid
            """,
            out_dir / "unresolved_without_topology.csv",
        )

        export_query(
            out,
            """
            SELECT *
            FROM part_audit
            ORDER BY source_part
            """,
            out_dir / "part_audit.csv",
        )

        summary = {
            "script":
                "global_scaffold_merge_v4_4_no_weights.py",
            "elapsed_minutes":
                round((time.time() - t0) / 60, 3),
            "source_part_result_count":
                120,
            "source_master_dbs_opened":
                False,
            "special_v4_1_validation":
                special_validation,
            "part_audit_reconstruction":
                audit_reconstruction_records,
            "unresolved_without_topology_policy":
                "preserve genuine fallback failures; do not fabricate topology",
            "three_level_weights_calculated":
                False,
            "weight_status":
                "DEFERRED_BY_USER_REQUEST",
            "merged_input_instance_counts": {
                "strict_scaffold_instances":
                    merged_strict,
                "unresolved_ring_systems":
                    merged_unresolved,
            },
            "final_global_row_counts":
                final_counts,
            "global_consistency_checks":
                checks,
            "final_quick_check":
                "ok",
            "final_db":
                str(final_db),
        }

        with open(
            out_dir / "run_summary.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                summary,
                f,
                indent=2,
                ensure_ascii=False,
            )

        out.close()
        out = None

        os.replace(
            staging_db,
            final_db,
        )

        log()
        log("=" * 100)
        log("FINAL MERGE COMPLETE")
        log("=" * 100)

        for key, value in final_counts.items():
            log(
                f"{key:32s}: {value:,}"
            )

        log()
        log(
            "part_0116 audit: "
            + (
                "RECONSTRUCTED"
                if "part_0116"
                in audit_reconstruction_records
                else "COPIED"
            )
        )
        log(
            "part_0119 audit: "
            + (
                "RECONSTRUCTED"
                if "part_0119"
                in audit_reconstruction_records
                else "COPIED"
            )
        )

        log(
            "Genuine unresolved_without_topology: "
            f"{final_counts['unresolved_without_topology']:,}"
        )
        log(
            "These rows are preserved, not repaired or fabricated."
        )
        log(
            "Three-level weights: NOT CALCULATED"
        )
        log(
            "Final SQLite quick_check: ok"
        )
        log(
            f"Elapsed: "
            f"{(time.time()-t0)/60:.2f} min"
        )
        log(
            f"Final DB: {final_db}"
        )
        log()
        log("MERGE SUCCESS.")

    finally:
        try:
            if out is not None:
                out.close()
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
