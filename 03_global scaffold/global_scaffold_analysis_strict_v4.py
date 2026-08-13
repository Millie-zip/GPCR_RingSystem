#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
global_scaffold_analysis_strict_v4.py

Purpose
-------
Build a GLOBAL fused-ring scaffold index from:
  1) eligible_index.db  (the molecule-level filter index)
  2) the original 120 part_XXXX.db files (READ-ONLY)

Core rules
----------
1. Only molecules present in eligible_index.db are analyzed.
2. Once a molecule is eligible, ALL of its ring_systems are retained,
   including 2-ring systems.
3. Existing successful scaffold records are reused directly:
      pure_ring_topology.scaffold_smiles -> topology key
      pure_ring_chemical.scaffold_smiles -> chemical key
4. Missing pure scaffolds are NOT deleted. They enter a rescue path.
5. Source part databases are never modified.
6. No full molecule tables are copied.
7. Global grouping is:
      topology -> chemical scaffold -> molecule/ring_system instance
8. Weights are assigned after all requested parts finish:
      each topology has equal total weight;
      chemicals within a topology share that topology weight equally;
      instances within a chemical share that chemical weight equally.

IMPORTANT RESCUE SAFETY
-----------------------
The rescue builder first reproduces the EXACT pure-scaffold construction logic from the original master-library script and is validation-gated.
Before trusting rescue on failed ring systems, run with e.g.:

    --validate-rescue 200 --enable-rescue

The script reconstructs 200 known-successful pure scaffolds from original
molecule + ring_atom_indices_json and requires exact SMILES agreement with
the stored pure_ring_chemical and pure_ring_topology records.

If validation is not 100%, rescue is aborted. This prevents silently creating
a second, incompatible scaffold definition.

Recommended first test
----------------------
Run only part_0000.db first. After inspecting audit/summary, rerun with
--all-parts for the full 120-part analysis.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sqlite3
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

try:
    from rdkit import Chem
except Exception:
    Chem = None


PURE_CHEM = "pure_ring_chemical"
PURE_TOPO = "pure_ring_topology"


@dataclass
class RingRec:
    ring_system_uid: str
    molecule_uid: str
    system_index: int
    ring_atom_indices_json: str
    sssr_basis_ring_count: int
    cycle_rank: int
    ring_system_type: str
    chemical_smiles: Optional[str] = None
    topology_smiles: Optional[str] = None


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def connect_ro(path: Path) -> sqlite3.Connection:
    # URI mode=ro gives an additional hard read-only guard.
    uri = f"file:{path.resolve()}?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    con.execute("PRAGMA query_only=ON")
    con.execute("PRAGMA temp_store=MEMORY")
    con.execute("PRAGMA cache_size=-262144")  # ~256 MiB per read connection
    con.execute("PRAGMA mmap_size=1073741824")  # 1 GiB read-only mmap
    return con


def connect_out(path: Path, overwrite: bool) -> sqlite3.Connection:
    if path.exists():
        if not overwrite:
            raise SystemExit(
                f"Output DB already exists: {path}\n"
                f"Use --overwrite only if you intentionally want to rebuild it."
            )
        path.unlink()

    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    # Conservative settings for shared filesystems. Output is derived data,
    # but we still keep normal transaction safety.
    con.execute("PRAGMA journal_mode=DELETE")
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute("PRAGMA temp_store=MEMORY")
    con.execute("PRAGMA cache_size=-524288")  # ~512 MiB
    con.execute("PRAGMA mmap_size=1073741824")  # 1 GiB mmap
    con.execute("PRAGMA foreign_keys=ON")
    return con


def init_output_schema(con: sqlite3.Connection) -> None:
    con.executescript(
        """
        CREATE TABLE run_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

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
            UNIQUE(topology_uid, chemical_smiles),
            FOREIGN KEY(topology_uid) REFERENCES topologies(topology_uid)
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
            PRIMARY KEY(source_part, ring_system_uid),
            FOREIGN KEY(topology_uid) REFERENCES topologies(topology_uid),
            FOREIGN KEY(chemical_uid) REFERENCES chemical_scaffolds(chemical_uid)
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
            PRIMARY KEY(source_part, ring_system_uid),
            FOREIGN KEY(topology_uid) REFERENCES topologies(topology_uid)
        );

        CREATE TABLE part_audit (
            source_part TEXT PRIMARY KEY,
            eligible_molecules INTEGER NOT NULL,
            eligible_ring_systems INTEGER NOT NULL,
            strict_complete_pairs INTEGER NOT NULL,
            unresolved_pure_scaffolds INTEGER NOT NULL,
            unresolved_with_topology INTEGER NOT NULL,
            unresolved_without_topology INTEGER NOT NULL,
            elapsed_seconds REAL NOT NULL
        );

        CREATE INDEX idx_strict_instances_topology
            ON strict_scaffold_instances(topology_uid);
        CREATE INDEX idx_strict_instances_chemical
            ON strict_scaffold_instances(chemical_uid);
        CREATE INDEX idx_strict_instances_molecule
            ON strict_scaffold_instances(molecule_uid);
        CREATE INDEX idx_chemical_topology
            ON chemical_scaffolds(topology_uid);
        CREATE INDEX idx_unresolved_topology
            ON unresolved_ring_systems(topology_uid);
        CREATE INDEX idx_unresolved_status
            ON unresolved_ring_systems(chemical_status);
        """
    )
    con.commit()

def sha_uid(prefix: str, text: str, nhex: int = 20) -> str:
    return f"{prefix}_{hashlib.sha1(text.encode('utf-8')).hexdigest()[:nhex]}"


def topology_uid(topology_smiles: str) -> str:
    return sha_uid("T", topology_smiles)


def chemical_uid(top_uid: str, chemical_smiles: str) -> str:
    return sha_uid("C", top_uid + "|" + chemical_smiles)


def find_eligible_table(con: sqlite3.Connection) -> str:
    required = {"source_part", "molecule_uid"}
    tables = [
        r[0]
        for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
    ]
    hits = []
    for table in tables:
        cols = {r[1] for r in con.execute(f'PRAGMA table_info("{table}")')}
        if required.issubset(cols):
            hits.append(table)

    if not hits:
        raise RuntimeError(
            "Could not find a table in eligible_index.db containing both "
            "'source_part' and 'molecule_uid'."
        )
    if len(hits) > 1:
        # Prefer a semantically obvious table if present.
        for preferred in ("eligible_molecules", "eligible", "molecules"):
            if preferred in hits:
                return preferred
        raise RuntimeError(
            "Multiple eligible-like tables found: "
            + ", ".join(hits)
            + ". Use --eligible-table explicitly."
        )
    return hits[0]


def eligible_source_values(con: sqlite3.Connection, table: str) -> List[str]:
    return [r[0] for r in con.execute(f'SELECT DISTINCT source_part FROM "{table}"')]


def resolve_source_part_value(
    basename: str, stem: str, available_values: Sequence[str]
) -> str:
    candidates = [basename, stem]
    avail = set(available_values)
    for c in candidates:
        if c in avail:
            return c

    # Handle values containing paths.
    by_base = {Path(v).name: v for v in available_values}
    if basename in by_base:
        return by_base[basename]
    if stem in by_base:
        return by_base[stem]

    # Handle a value such as part_0000 without .db.
    for v in available_values:
        if Path(v).stem == stem:
            return v

    raise RuntimeError(
        f"Could not map source part {basename!r} to any source_part value in eligible index."
    )


def load_eligible_molecules(
    eligible_con: sqlite3.Connection, table: str, source_value: str
) -> Set[str]:
    log(f"Loading eligible molecule_uids for {source_value} ...")
    cur = eligible_con.execute(
        f'SELECT molecule_uid FROM "{table}" WHERE source_part=?', (source_value,)
    )
    out = {row[0] for row in cur}
    log(f"  eligible molecules: {len(out):,}")
    return out


def scan_eligible_ring_systems(
    part_con: sqlite3.Connection, eligible_mols: Set[str]
) -> Dict[str, RingRec]:
    log("Sequentially scanning ring_systems ...")
    started = time.time()
    rings: Dict[str, RingRec] = {}
    sql = """
        SELECT
            ring_system_uid,
            molecule_uid,
            system_index,
            ring_atom_indices_json,
            sssr_basis_ring_count,
            cycle_rank,
            ring_system_type
        FROM ring_systems
    """
    for row in part_con.execute(sql):
        mol_uid = row[1]
        if mol_uid not in eligible_mols:
            continue
        rr = RingRec(
            ring_system_uid=row[0],
            molecule_uid=mol_uid,
            system_index=int(row[2]),
            ring_atom_indices_json=row[3],
            sssr_basis_ring_count=int(row[4]),
            cycle_rank=int(row[5]),
            ring_system_type=row[6],
        )
        rings[rr.ring_system_uid] = rr
    log(f"  eligible ring systems: {len(rings):,} in {time.time()-started:.2f} s")
    return rings


def _find_ring_variant_unique_index(
    part_con: sqlite3.Connection,
) -> Optional[str]:
    """Discover UNIQUE(ring_system_uid, variant_type)."""
    for row in part_con.execute("PRAGMA index_list('scaffold_variants')"):
        index_name = row[1]
        if not bool(row[2]):
            continue
        cols = [
            r[2]
            for r in part_con.execute(
                f'PRAGMA index_info("{index_name}")'
            )
        ]
        if cols[:2] == ["ring_system_uid", "variant_type"]:
            return index_name
    return None


def scan_existing_pure_scaffolds(
    part_con: sqlite3.Connection,
    rings: Dict[str, RingRec],
    batch_size: int = 400,
) -> None:
    """
    Performance-only optimization.

    Instead of scanning every pure_ring_chemical and pure_ring_topology row in
    the whole part, query only eligible ring_system_uids through the existing
    UNIQUE(ring_system_uid, variant_type) index.
    """
    ring_uids = sorted(rings)
    if not ring_uids:
        return

    index_name = _find_ring_variant_unique_index(part_con)
    if index_name:
        log(f"Indexed batched pure-scaffold lookup using {index_name} ...")
        indexed = f' INDEXED BY "{index_name}" '
    else:
        log("WARNING: composite UNIQUE index not found; using SQLite planner.")
        indexed = " "

    matched_chem = 0
    matched_topo = 0
    total_batches = (len(ring_uids) + batch_size - 1) // batch_size
    started = time.time()

    for batch_no, pos in enumerate(range(0, len(ring_uids), batch_size), 1):
        batch = ring_uids[pos:pos + batch_size]
        placeholders = ",".join("?" for _ in batch)
        sql = (
            "SELECT ring_system_uid, variant_type, scaffold_smiles "
            "FROM scaffold_variants"
            + indexed
            + f"WHERE ring_system_uid IN ({placeholders}) "
              "AND variant_type IN (?, ?)"
        )
        params = [*batch, PURE_CHEM, PURE_TOPO]

        for ring_uid, variant, smi in part_con.execute(sql, params):
            rr = rings.get(ring_uid)
            if rr is None:
                continue
            if variant == PURE_CHEM:
                rr.chemical_smiles = smi
                matched_chem += 1
            elif variant == PURE_TOPO:
                rr.topology_smiles = smi
                matched_topo += 1

        if batch_no == 1 or batch_no == total_batches or batch_no % 50 == 0:
            log(
                f"  scaffold batch {batch_no}/{total_batches}: "
                f"chemical={matched_chem:,}, topology={matched_topo:,}"
            )

    log(
        f"  matched pure chemical={matched_chem:,}, topology={matched_topo:,} "
        f"in {time.time()-started:.2f} s"
    )

def scan_failure_map(
    part_con: sqlite3.Connection,
    missing_rings: Set[str],
    rings: Dict[str, RingRec],
) -> Dict[str, str]:
    """
    failure rows identify molecule_uid + system_index, not ring_system_uid.
    Convert to ring_system_uid for the missing eligible ring systems.
    """
    if not missing_rings:
        return {}

    key_to_ring = {
        (rings[r].molecule_uid, rings[r].system_index): r for r in missing_rings
    }
    out: Dict[str, str] = {}
    sql = """
        SELECT molecule_uid, system_index, failure_stage
        FROM failures
        WHERE failure_stage='build_pure_ring_scaffolds'
    """
    for mol_uid, system_index, stage in part_con.execute(sql):
        if mol_uid is None or system_index is None:
            continue
        ring_uid = key_to_ring.get((mol_uid, int(system_index)))
        if ring_uid is not None:
            out[ring_uid] = stage
    return out


def fetch_smiles_for_molecules(
    part_con: sqlite3.Connection,
    molecule_uids: Set[str],
    batch_size: int = 400,
) -> Dict[str, str]:
    """
    Fetch only the requested molecule SMILES through the molecules primary-key
    index, in small IN(...) batches.

    v2 used a full sequential scan of the molecules table. That is safe but
    wasteful when only ~10^3 rescue molecules are needed from a very large
    part database. The current implementation performs bounded indexed reads
    instead.

    batch_size=400 is deliberately below SQLite's common host-parameter limit
    and avoids building very large IN clauses.
    """
    if not molecule_uids:
        return {}

    requested = sorted(molecule_uids)
    out: Dict[str, str] = {}

    log(
        f"Indexed lookup of original_smiles for {len(requested):,} "
        f"rescue/validation molecules (batch_size={batch_size}) ..."
    )

    started = time.time()
    total_batches = (len(requested) + batch_size - 1) // batch_size

    for batch_no, start in enumerate(range(0, len(requested), batch_size), start=1):
        batch = requested[start:start + batch_size]
        placeholders = ",".join("?" for _ in batch)

        sql = (
            "SELECT molecule_uid, original_smiles "
            "FROM molecules "
            f"WHERE molecule_uid IN ({placeholders})"
        )

        for mol_uid, smi in part_con.execute(sql, batch):
            out[mol_uid] = smi

        if batch_no == 1 or batch_no == total_batches or batch_no % 10 == 0:
            log(
                f"  SMILES lookup batch {batch_no}/{total_batches}: "
                f"found {len(out):,}/{len(requested):,}"
            )

    elapsed = time.time() - started
    log(
        f"  molecule SMILES found: {len(out):,}/{len(requested):,} "
        f"in {elapsed:.2f} s"
    )

    return out

def clear_stereo_and_directions(mol: Chem.Mol) -> None:
    """Exact helper copied from the original master-library builder."""
    Chem.RemoveStereochemistry(mol)
    for atom in mol.GetAtoms():
        atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    for bond in mol.GetBonds():
        bond.SetStereo(Chem.BondStereo.STEREONONE)
        bond.SetBondDir(Chem.BondDir.NONE)


def build_induced_submol_exact(
    mol: Chem.Mol,
    atom_indices: Iterable[int],
    *,
    preserve_atom_maps: bool = True,
) -> Tuple[Chem.Mol, Dict[int, int], Dict[int, int]]:
    """
    Exact pure-scaffold subgraph construction logic from
    build_fused_ring_master_library_server.py.

    Key detail:
    it COPIES THE FULL MOLECULE, writes full-atom map numbers, then removes
    non-scaffold atoms in reverse index order. It does NOT use
    MolFragmentToSmiles() and it does NOT heuristically repair aromatic N/H.
    """
    keep_atoms = tuple(sorted(set(map(int, atom_indices))))
    if not keep_atoms:
        raise ValueError("subgraph atom set is empty")

    atom_count = mol.GetNumAtoms()
    if keep_atoms[0] < 0 or keep_atoms[-1] >= atom_count:
        raise IndexError("subgraph atom index outside original molecule")

    work = Chem.Mol(mol)
    clear_stereo_and_directions(work)

    if preserve_atom_maps:
        for atom in work.GetAtoms():
            atom.SetAtomMapNum(atom.GetIdx() + 1)

    rw = Chem.RWMol(work)
    keep_set = set(keep_atoms)

    for atom_idx in range(atom_count - 1, -1, -1):
        if atom_idx not in keep_set:
            rw.RemoveAtom(atom_idx)

    submol = rw.GetMol()
    clear_stereo_and_directions(submol)
    submol.UpdatePropertyCache(strict=False)

    submol_to_full: Dict[int, int] = {}
    full_to_submol: Dict[int, int] = {}

    for atom in submol.GetAtoms():
        full_idx = atom.GetAtomMapNum() - 1
        if full_idx < 0:
            raise RuntimeError("subgraph atom-map number lost")
        sub_idx = atom.GetIdx()
        submol_to_full[sub_idx] = full_idx
        full_to_submol[full_idx] = sub_idx

    return submol, full_to_submol, submol_to_full


def copy_without_atom_maps_exact(mol: Chem.Mol) -> Chem.Mol:
    result = Chem.Mol(mol)
    for atom in result.GetAtoms():
        atom.SetAtomMapNum(0)
    result.UpdatePropertyCache(strict=False)
    return result


def canonical_scaffold_smiles_exact(
    submol_with_maps: Chem.Mol,
) -> Tuple[str, str]:
    """
    Exact canonicalization logic from the original master-library builder.
    Returns:
      (unmapped canonical scaffold SMILES, mapped canonical scaffold SMILES)
    """
    mapped = Chem.Mol(submol_with_maps)
    mapped.UpdatePropertyCache(strict=False)

    Chem.SanitizeMol(mapped)

    mapped_smiles = Chem.MolToSmiles(
        mapped,
        canonical=True,
        isomericSmiles=False,
        kekuleSmiles=False,
    )
    if not mapped_smiles:
        raise ValueError("mapped scaffold SMILES is empty")

    unmapped = copy_without_atom_maps_exact(mapped)
    Chem.SanitizeMol(unmapped)

    smiles = Chem.MolToSmiles(
        unmapped,
        canonical=True,
        isomericSmiles=False,
        kekuleSmiles=False,
    )
    if not smiles:
        raise ValueError("canonical scaffold SMILES is empty")

    check = Chem.MolFromSmiles(smiles, sanitize=True)
    if check is None:
        raise ValueError(f"scaffold cannot be read back: {smiles}")

    return smiles, mapped_smiles


def build_topology_scaffold_exact(
    chemical_submol_with_maps: Chem.Mol,
) -> Tuple[Chem.Mol, str, str]:
    """
    Exact topology conversion logic from the original master-library builder.
    All atoms -> carbon; all bonds -> single; charge/aromaticity/stereo removed.
    """
    rw = Chem.RWMol(chemical_submol_with_maps)

    for atom in rw.GetAtoms():
        atom.SetAtomicNum(6)
        atom.SetFormalCharge(0)
        atom.SetIsAromatic(False)
        atom.SetIsotope(0)
        atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
        atom.SetNumExplicitHs(0)
        atom.SetNoImplicit(False)

    for bond in rw.GetBonds():
        bond.SetBondType(Chem.BondType.SINGLE)
        bond.SetIsAromatic(False)
        bond.SetStereo(Chem.BondStereo.STEREONONE)
        bond.SetBondDir(Chem.BondDir.NONE)

    topology = rw.GetMol()
    topology.UpdatePropertyCache(strict=False)
    Chem.SanitizeMol(topology)

    smiles, mapped_smiles = canonical_scaffold_smiles_exact(topology)
    return topology, smiles, mapped_smiles


def reconstruct_pair(original_smiles: str, ring_atom_indices_json: str) -> Tuple[str, str]:
    """
    Reconstruct a pure chemical/topology scaffold pair using the EXACT
    master_v1 builder logic.

    This function is used both for:
      1) validation on known-successful records;
      2) the first rescue attempt on missing records.

    Important:
    a ring system that failed the original strict sanitizer may fail again here.
    Such cases remain unresolved and are preserved for a later, explicitly
    defined fallback rescue step; they are never silently rewritten.
    """
    if Chem is None:
        raise RuntimeError("RDKit is not importable in the current Python environment")

    mol = Chem.MolFromSmiles(original_smiles, sanitize=True)
    if mol is None:
        raise ValueError("RDKit could not parse original_smiles")

    atom_indices = json.loads(ring_atom_indices_json)
    if not isinstance(atom_indices, list) or not atom_indices:
        raise ValueError("ring_atom_indices_json is empty or not a list")

    pure_sub, _f2s, _s2f = build_induced_submol_exact(mol, atom_indices)
    chemical_smiles, _chemical_mapped = canonical_scaffold_smiles_exact(pure_sub)
    _topology, topology_smiles, _topology_mapped = build_topology_scaffold_exact(pure_sub)

    return chemical_smiles, topology_smiles

def validate_rescue_builder(
    part_con: sqlite3.Connection,
    rings: Dict[str, RingRec],
    n_validate: int,
) -> Tuple[int, int, List[dict]]:
    if n_validate <= 0:
        return 0, 0, []

    candidates = [
        rr for rr in rings.values()
        if rr.chemical_smiles is not None and rr.topology_smiles is not None
    ][:n_validate]

    if not candidates:
        raise RuntimeError("No known-successful scaffold pairs available for rescue validation.")

    smiles_map = fetch_smiles_for_molecules(
        part_con, {rr.molecule_uid for rr in candidates}
    )

    ok = 0
    mismatches: List[dict] = []
    for rr in candidates:
        original = smiles_map.get(rr.molecule_uid)
        if original is None:
            mismatches.append(
                {
                    "ring_system_uid": rr.ring_system_uid,
                    "reason": "missing original_smiles",
                }
            )
            continue
        try:
            chem, topo = reconstruct_pair(original, rr.ring_atom_indices_json)
            chem_ok = (chem == rr.chemical_smiles)
            topo_ok = (topo == rr.topology_smiles)
            if chem_ok and topo_ok:
                ok += 1
            else:
                mismatches.append(
                    {
                        "ring_system_uid": rr.ring_system_uid,
                        "stored_chemical": rr.chemical_smiles,
                        "rebuilt_chemical": chem,
                        "stored_topology": rr.topology_smiles,
                        "rebuilt_topology": topo,
                    }
                )
        except Exception as e:
            mismatches.append(
                {
                    "ring_system_uid": rr.ring_system_uid,
                    "reason": f"{type(e).__name__}: {e}",
                }
            )

    return ok, len(candidates), mismatches


def ensure_topology_and_chemical(
    out_con: sqlite3.Connection,
    top_smi: str,
    chem_smi: str,
) -> Tuple[str, str]:
    t_uid = topology_uid(top_smi)
    c_uid = chemical_uid(t_uid, chem_smi)

    out_con.execute(
        """
        INSERT OR IGNORE INTO topologies(topology_uid, topology_smiles)
        VALUES (?, ?)
        """,
        (t_uid, top_smi),
    )
    out_con.execute(
        """
        INSERT OR IGNORE INTO chemical_scaffolds(
            chemical_uid, topology_uid, chemical_smiles
        ) VALUES (?, ?, ?)
        """,
        (c_uid, t_uid, chem_smi),
    )
    return t_uid, c_uid


def ensure_topology_only(
    out_con: sqlite3.Connection,
    top_smi: str,
) -> str:
    t_uid = topology_uid(top_smi)
    out_con.execute(
        """
        INSERT OR IGNORE INTO topologies(topology_uid, topology_smiles)
        VALUES (?, ?)
        """,
        (t_uid, top_smi),
    )
    return t_uid


def insert_strict_instance(
    out_con: sqlite3.Connection,
    source_part: str,
    rr: RingRec,
) -> None:
    if rr.topology_smiles is None or rr.chemical_smiles is None:
        raise ValueError("strict instance requires both topology and chemical scaffold")

    t_uid, c_uid = ensure_topology_and_chemical(
        out_con,
        rr.topology_smiles,
        rr.chemical_smiles,
    )
    out_con.execute(
        """
        INSERT OR REPLACE INTO strict_scaffold_instances(
            source_part, molecule_uid, ring_system_uid, system_index,
            sssr_basis_ring_count, cycle_rank, ring_system_type,
            topology_uid, chemical_uid
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            source_part,
            rr.molecule_uid,
            rr.ring_system_uid,
            rr.system_index,
            rr.sssr_basis_ring_count,
            rr.cycle_rank,
            rr.ring_system_type,
            t_uid,
            c_uid,
        ),
    )



def bulk_write_strict_instances(
    out_con: sqlite3.Connection,
    source_part: str,
    strict: Sequence[RingRec],
) -> None:
    """Batch-write the exact same Strict-v1 IDs and rows."""
    if not strict:
        return

    topology_rows: Dict[str, Tuple[str, str]] = {}
    chemical_rows: Dict[str, Tuple[str, str, str]] = {}
    instance_rows: List[Tuple] = []

    for rr in strict:
        if rr.topology_smiles is None or rr.chemical_smiles is None:
            raise ValueError("strict record missing scaffold")

        t_uid = topology_uid(rr.topology_smiles)
        c_uid = chemical_uid(t_uid, rr.chemical_smiles)

        topology_rows[t_uid] = (t_uid, rr.topology_smiles)
        chemical_rows[c_uid] = (c_uid, t_uid, rr.chemical_smiles)
        instance_rows.append(
            (
                source_part,
                rr.molecule_uid,
                rr.ring_system_uid,
                rr.system_index,
                rr.sssr_basis_ring_count,
                rr.cycle_rank,
                rr.ring_system_type,
                t_uid,
                c_uid,
            )
        )

    out_con.executemany(
        "INSERT OR IGNORE INTO topologies(topology_uid, topology_smiles) VALUES (?, ?)",
        list(topology_rows.values()),
    )
    out_con.executemany(
        """
        INSERT OR IGNORE INTO chemical_scaffolds(
            chemical_uid, topology_uid, chemical_smiles
        ) VALUES (?, ?, ?)
        """,
        list(chemical_rows.values()),
    )

    sql = """
        INSERT OR REPLACE INTO strict_scaffold_instances(
            source_part, molecule_uid, ring_system_uid, system_index,
            sssr_basis_ring_count, cycle_rank, ring_system_type,
            topology_uid, chemical_uid
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    for pos in range(0, len(instance_rows), 20000):
        out_con.executemany(sql, instance_rows[pos:pos + 20000])

def scan_failure_details(
    part_con: sqlite3.Connection,
    missing_rings: Set[str],
    rings: Dict[str, RingRec],
) -> Dict[str, Tuple[Optional[str], Optional[str], Optional[str]]]:
    if not missing_rings:
        return {}

    key_to_ring = {
        (rings[r].molecule_uid, rings[r].system_index): r
        for r in missing_rings
    }

    out: Dict[str, Tuple[Optional[str], Optional[str], Optional[str]]] = {}

    sql = """
        SELECT molecule_uid, system_index, failure_stage, error_type, error_message
        FROM failures
        WHERE failure_stage='build_pure_ring_scaffolds'
    """

    for mol_uid, system_index, stage, error_type, error_message in part_con.execute(sql):
        if mol_uid is None or system_index is None:
            continue
        ring_uid = key_to_ring.get((mol_uid, int(system_index)))
        if ring_uid is not None:
            out[ring_uid] = (stage, error_type, error_message)

    return out


def build_topology_from_source_ring(
    original_smiles: str,
    ring_atom_indices_json: str,
) -> str:
    """
    Reliable topology fallback for pure-scaffold failures.

    The source complete molecule is valid and sanitized. We keep exactly the
    ring-system atom set, preserve only connectivity, convert all atoms to C
    and all internal bonds to SINGLE, then canonicalize.

    This matches the project's topology definition and avoids aromatic
    Kekulization because no aromatic bond survives.
    """
    if Chem is None:
        raise RuntimeError("RDKit is required for topology fallback")

    mol = Chem.MolFromSmiles(original_smiles, sanitize=True)
    if mol is None:
        raise ValueError("RDKit could not parse original_smiles")

    atom_indices = sorted(set(map(int, json.loads(ring_atom_indices_json))))
    if not atom_indices:
        raise ValueError("ring_atom_indices_json is empty")

    keep = set(atom_indices)
    rw = Chem.RWMol()
    mapping: Dict[int, int] = {}

    for old_idx in atom_indices:
        atom = Chem.Atom(6)
        atom.SetFormalCharge(0)
        atom.SetIsAromatic(False)
        atom.SetNumExplicitHs(0)
        atom.SetNoImplicit(False)
        mapping[old_idx] = rw.AddAtom(atom)

    for bond in mol.GetBonds():
        a = int(bond.GetBeginAtomIdx())
        b = int(bond.GetEndAtomIdx())
        if a in keep and b in keep:
            rw.AddBond(mapping[a], mapping[b], Chem.BondType.SINGLE)

    topo = rw.GetMol()
    topo.UpdatePropertyCache(strict=False)
    Chem.SanitizeMol(topo)

    smi = Chem.MolToSmiles(
        topo,
        canonical=True,
        isomericSmiles=False,
        kekuleSmiles=False,
    )
    if not smi:
        raise ValueError("topology SMILES is empty")

    return smi


def process_part(
    part_path: Path,
    source_value: str,
    eligible_con: sqlite3.Connection,
    eligible_table: str,
    out_con: sqlite3.Connection,
    validate_topology_n: int,
    validation_dir: Path,
) -> dict:
    started = time.time()
    log(f"=== Processing {part_path.name} ===")

    eligible_mols = load_eligible_molecules(
        eligible_con, eligible_table, source_value
    )

    if not eligible_mols:
        elapsed = time.time() - started
        out_con.execute(
            """
            INSERT OR REPLACE INTO part_audit VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (part_path.name, 0, 0, 0, 0, 0, 0, elapsed),
        )
        out_con.commit()
        return {
            "source_part": part_path.name,
            "eligible_molecules": 0,
            "eligible_ring_systems": 0,
            "strict_complete_pairs": 0,
            "unresolved_pure_scaffolds": 0,
            "unresolved_with_topology": 0,
            "unresolved_without_topology": 0,
            "elapsed_seconds": elapsed,
        }

    part_con = connect_ro(part_path)

    try:
        rings = scan_eligible_ring_systems(part_con, eligible_mols)
        scan_existing_pure_scaffolds(part_con, rings)

        strict = [
            rr for rr in rings.values()
            if rr.chemical_smiles is not None and rr.topology_smiles is not None
        ]
        unresolved = [
            rr for rr in rings.values()
            if rr.chemical_smiles is None or rr.topology_smiles is None
        ]

        log(f"  strict complete pairs: {len(strict):,}")
        log(f"  unresolved pure scaffolds: {len(unresolved):,}")

        # Optional topology fallback validation on successful records.
        if validate_topology_n > 0 and strict:
            sample = strict[:validate_topology_n]
            sample_mols = {rr.molecule_uid for rr in sample}
            smiles_map = fetch_smiles_for_molecules(part_con, sample_mols)

            ok = 0
            mismatches = []
            for rr in sample:
                original = smiles_map.get(rr.molecule_uid)
                if original is None:
                    mismatches.append({
                        "ring_system_uid": rr.ring_system_uid,
                        "reason": "original_smiles not found",
                    })
                    continue

                try:
                    topo = build_topology_from_source_ring(
                        original,
                        rr.ring_atom_indices_json,
                    )
                    if topo == rr.topology_smiles:
                        ok += 1
                    else:
                        mismatches.append({
                            "ring_system_uid": rr.ring_system_uid,
                            "stored_topology": rr.topology_smiles,
                            "rebuilt_topology": topo,
                        })
                except Exception as e:
                    mismatches.append({
                        "ring_system_uid": rr.ring_system_uid,
                        "reason": f"{type(e).__name__}: {e}",
                    })

            validation_dir.mkdir(parents=True, exist_ok=True)
            p = validation_dir / f"{part_path.stem}_topology_fallback_validation.json"
            p.write_text(
                json.dumps(
                    {
                        "validated": len(sample),
                        "exact_matches": ok,
                        "exact_match_fraction": ok / len(sample),
                        "mismatches": mismatches[:100],
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            log(f"  topology fallback validation: {ok:,}/{len(sample):,}")

            if ok != len(sample):
                raise RuntimeError(
                    "Topology fallback does not reproduce stored topology exactly. "
                    f"See {p}"
                )

        log("Bulk-writing strict scaffold instances ...")
        write_started = time.time()
        bulk_write_strict_instances(out_con, part_path.name, strict)
        log(f"  strict bulk write complete in {time.time()-write_started:.2f} s")

        unresolved_with_topology = 0
        unresolved_without_topology = 0

        if unresolved:
            unresolved_ids = {rr.ring_system_uid for rr in unresolved}
            failure_details = scan_failure_details(
                part_con,
                unresolved_ids,
                rings,
            )

            unresolved_mols = {rr.molecule_uid for rr in unresolved}
            smiles_map = fetch_smiles_for_molecules(
                part_con,
                unresolved_mols,
            )

            log("Recording unresolved chemical scaffolds with topology fallback ...")
            unresolved_rows: List[Tuple] = []

            for rr in unresolved:
                stage, error_type, error_message = failure_details.get(
                    rr.ring_system_uid,
                    (None, None, None),
                )

                topology_smiles = rr.topology_smiles
                topology_uid_value = None

                if topology_smiles is None:
                    original = smiles_map.get(rr.molecule_uid)
                    if original is not None:
                        try:
                            topology_smiles = build_topology_from_source_ring(
                                original,
                                rr.ring_atom_indices_json,
                            )
                        except Exception as e:
                            if error_message:
                                error_message = (
                                    error_message
                                    + " | topology_fallback: "
                                    + f"{type(e).__name__}: {e}"
                                )
                            else:
                                error_message = (
                                    "topology_fallback: "
                                    + f"{type(e).__name__}: {e}"
                                )

                if topology_smiles is not None:
                    topology_uid_value = ensure_topology_only(
                        out_con,
                        topology_smiles,
                    )
                    unresolved_with_topology += 1
                else:
                    unresolved_without_topology += 1

                if error_type == "KekulizeException":
                    chemical_status = "unresolved_kekulize"
                elif stage == "build_pure_ring_scaffolds":
                    chemical_status = "unresolved_pure_scaffold"
                else:
                    chemical_status = "unresolved_unknown"

                unresolved_rows.append(
                    (
                        part_path.name,
                        rr.molecule_uid,
                        rr.ring_system_uid,
                        rr.system_index,
                        rr.sssr_basis_ring_count,
                        rr.cycle_rank,
                        rr.ring_system_type,
                        topology_uid_value,
                        topology_smiles,
                        chemical_status,
                        stage,
                        error_type,
                        error_message,
                    )
                )

            if unresolved_rows:
                out_con.executemany(
                    """
                    INSERT OR REPLACE INTO unresolved_ring_systems(
                        source_part, molecule_uid, ring_system_uid, system_index,
                        sssr_basis_ring_count, cycle_rank, ring_system_type,
                        topology_uid, topology_smiles, chemical_status,
                        original_failure_stage, original_error_type,
                        original_error_message
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    unresolved_rows,
                )

        elapsed = time.time() - started

        audit = {
            "source_part": part_path.name,
            "eligible_molecules": len(eligible_mols),
            "eligible_ring_systems": len(rings),
            "strict_complete_pairs": len(strict),
            "unresolved_pure_scaffolds": len(unresolved),
            "unresolved_with_topology": unresolved_with_topology,
            "unresolved_without_topology": unresolved_without_topology,
            "elapsed_seconds": elapsed,
        }

        out_con.execute(
            """
            INSERT OR REPLACE INTO part_audit(
                source_part, eligible_molecules, eligible_ring_systems,
                strict_complete_pairs, unresolved_pure_scaffolds,
                unresolved_with_topology, unresolved_without_topology,
                elapsed_seconds
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                audit["source_part"],
                audit["eligible_molecules"],
                audit["eligible_ring_systems"],
                audit["strict_complete_pairs"],
                audit["unresolved_pure_scaffolds"],
                audit["unresolved_with_topology"],
                audit["unresolved_without_topology"],
                audit["elapsed_seconds"],
            ),
        )
        out_con.commit()

        log(
            f"Finished {part_path.name}: strict={len(strict):,}, "
            f"unresolved={len(unresolved):,}, "
            f"unresolved_with_topology={unresolved_with_topology:,}, "
            f"unresolved_without_topology={unresolved_without_topology:,}, "
            f"elapsed={elapsed/60:.2f} min"
        )

        return audit

    finally:
        part_con.close()



def finalize_counts_and_weights(out_con: sqlite3.Connection) -> dict:
    log("Finalizing strict global scaffold counts and weights ...")

    out_con.execute(
        """
        UPDATE topologies
        SET strict_chemical_count=0,
            strict_instance_count=0,
            unresolved_instance_count=0,
            topology_weight=NULL
        """
    )
    out_con.execute(
        """
        UPDATE chemical_scaffolds
        SET strict_instance_count=0,
            chemical_weight=NULL
        """
    )
    out_con.execute(
        "UPDATE strict_scaffold_instances SET instance_weight=NULL"
    )
    out_con.commit()

    # Strict instance counts per chemical.
    out_con.execute(
        """
        UPDATE chemical_scaffolds
        SET strict_instance_count = (
            SELECT COUNT(*)
            FROM strict_scaffold_instances i
            WHERE i.chemical_uid = chemical_scaffolds.chemical_uid
        )
        """
    )
    out_con.commit()

    # Strict chemical + strict instance + unresolved counts per topology.
    out_con.execute(
        """
        UPDATE topologies
        SET strict_chemical_count = (
                SELECT COUNT(*)
                FROM chemical_scaffolds c
                WHERE c.topology_uid = topologies.topology_uid
            ),
            strict_instance_count = (
                SELECT COUNT(*)
                FROM strict_scaffold_instances i
                WHERE i.topology_uid = topologies.topology_uid
            ),
            unresolved_instance_count = (
                SELECT COUNT(*)
                FROM unresolved_ring_systems u
                WHERE u.topology_uid = topologies.topology_uid
            )
        """
    )
    out_con.commit()

    # Weighting universe: only topologies that have >=1 strict chemical scaffold.
    n_weighted_topologies = out_con.execute(
        """
        SELECT COUNT(*)
        FROM topologies
        WHERE strict_chemical_count > 0
        """
    ).fetchone()[0]

    if n_weighted_topologies > 0:
        top_weight = 1.0 / n_weighted_topologies

        out_con.execute(
            """
            UPDATE topologies
            SET topology_weight=?
            WHERE strict_chemical_count > 0
            """,
            (top_weight,),
        )

        out_con.execute(
            """
            UPDATE chemical_scaffolds
            SET chemical_weight = (
                SELECT t.topology_weight / t.strict_chemical_count
                FROM topologies t
                WHERE t.topology_uid = chemical_scaffolds.topology_uid
            )
            WHERE strict_instance_count > 0
            """
        )

        out_con.execute(
            """
            UPDATE strict_scaffold_instances
            SET instance_weight = (
                SELECT c.chemical_weight / c.strict_instance_count
                FROM chemical_scaffolds c
                WHERE c.chemical_uid = strict_scaffold_instances.chemical_uid
            )
            """
        )
        out_con.commit()

    stats = {
        "topology_count_total": out_con.execute(
            "SELECT COUNT(*) FROM topologies"
        ).fetchone()[0],
        "weighted_topology_count": n_weighted_topologies,
        "strict_chemical_count": out_con.execute(
            "SELECT COUNT(*) FROM chemical_scaffolds"
        ).fetchone()[0],
        "strict_instance_count": out_con.execute(
            "SELECT COUNT(*) FROM strict_scaffold_instances"
        ).fetchone()[0],
        "unresolved_instance_count": out_con.execute(
            "SELECT COUNT(*) FROM unresolved_ring_systems"
        ).fetchone()[0],
        "unresolved_with_topology_count": out_con.execute(
            """
            SELECT COUNT(*)
            FROM unresolved_ring_systems
            WHERE topology_uid IS NOT NULL
            """
        ).fetchone()[0],
        "unresolved_without_topology_count": out_con.execute(
            """
            SELECT COUNT(*)
            FROM unresolved_ring_systems
            WHERE topology_uid IS NULL
            """
        ).fetchone()[0],
        "strict_instance_weight_sum": out_con.execute(
            """
            SELECT COALESCE(SUM(instance_weight), 0.0)
            FROM strict_scaffold_instances
            """
        ).fetchone()[0],
    }

    return stats

def write_reports(
    out_con: sqlite3.Connection,
    out_dir: Path,
    final_stats: dict,
    args: argparse.Namespace,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "part_audit.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        w = csv.writer(f)
        w.writerow(
            [
                "source_part",
                "eligible_molecules",
                "eligible_ring_systems",
                "strict_complete_pairs",
                "unresolved_pure_scaffolds",
                "unresolved_with_topology",
                "unresolved_without_topology",
                "elapsed_seconds",
            ]
        )
        for row in out_con.execute(
            """
            SELECT source_part, eligible_molecules, eligible_ring_systems,
                   strict_complete_pairs, unresolved_pure_scaffolds,
                   unresolved_with_topology, unresolved_without_topology,
                   elapsed_seconds
            FROM part_audit
            ORDER BY source_part
            """
        ):
            w.writerow(row)

    with (out_dir / "topology_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        w = csv.writer(f)
        w.writerow(
            [
                "topology_uid",
                "topology_smiles",
                "strict_chemical_count",
                "strict_instance_count",
                "unresolved_instance_count",
                "topology_weight",
            ]
        )
        for row in out_con.execute(
            """
            SELECT topology_uid, topology_smiles,
                   strict_chemical_count, strict_instance_count,
                   unresolved_instance_count, topology_weight
            FROM topologies
            ORDER BY strict_instance_count DESC, topology_uid
            """
        ):
            w.writerow(row)

    with (out_dir / "chemical_scaffold_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        w = csv.writer(f)
        w.writerow(
            [
                "chemical_uid",
                "topology_uid",
                "chemical_smiles",
                "strict_instance_count",
                "chemical_weight",
            ]
        )
        for row in out_con.execute(
            """
            SELECT chemical_uid, topology_uid, chemical_smiles,
                   strict_instance_count, chemical_weight
            FROM chemical_scaffolds
            ORDER BY strict_instance_count DESC, chemical_uid
            """
        ):
            w.writerow(row)

    with (out_dir / "unresolved_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        w = csv.writer(f)
        w.writerow(
            [
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
            ]
        )
        for row in out_con.execute(
            """
            SELECT source_part, molecule_uid, ring_system_uid, system_index,
                   sssr_basis_ring_count, cycle_rank, ring_system_type,
                   topology_uid, topology_smiles, chemical_status,
                   original_failure_stage, original_error_type,
                   original_error_message
            FROM unresolved_ring_systems
            ORDER BY source_part, ring_system_uid
            """
        ):
            w.writerow(row)

    unresolved_status_counts = dict(
        out_con.execute(
            """
            SELECT chemical_status, COUNT(*)
            FROM unresolved_ring_systems
            GROUP BY chemical_status
            """
        ).fetchall()
    )

    summary = {
        "program": "global_scaffold_analysis_strict_v4.py",
        "eligible_index": str(Path(args.eligible_index).resolve()),
        "parts_dir": str(Path(args.parts_dir).resolve()),
        "output_db": str(Path(args.output_db).resolve()),
        "validate_topology": int(args.validate_topology),
        "final_stats": final_stats,
        "unresolved_status_counts": unresolved_status_counts,
        "weight_definition": {
            "scope": "strict chemical scaffolds only",
            "topology": "1 / N_topologies_with_at_least_one_strict_chemical",
            "chemical": "topology_weight / N_strict_chemicals_in_topology",
            "instance": "chemical_weight / N_strict_instances_in_chemical",
            "unresolved_chemical_instances": "preserved but excluded from chemical weight allocation",
        },
    }

    (out_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    logic = """GLOBAL FUSED-RING SCAFFOLD ANALYSIS AND WEIGHTING LOGIC
=================================================================

1. INPUT AND IMMUTABILITY
-------------------------
The analysis uses:
  - eligible_index.db as the molecule-level eligibility index;
  - the original part_XXXX.db files as the sole structural source.

All original part databases are opened read-only. They are never updated,
deleted from, merged, or rewritten. Full molecular records are not copied.

2. ELIGIBILITY SCOPE
--------------------
Only molecules listed in eligible_index.db are analyzed.

Eligibility was previously defined at the complete-molecule level:
  - heavy atom count 13-35 inclusive;
  - at least one continuous fused-ring system with
    sssr_basis_ring_count >= 3.

Once a molecule is eligible, ALL ring systems belonging to that molecule are
retained for scaffold analysis, including internal 2-ring systems.

Therefore the operational scaffold instance is:
    source_part + molecule_uid + ring_system_uid

3. STRICT SCAFFOLD DEFINITIONS
------------------------------
For ring systems where the original master-library extraction succeeded:

  pure_ring_topology.scaffold_smiles
      -> global topology scaffold key

  pure_ring_chemical.scaffold_smiles
      -> strict chemical scaffold key

The stored successful scaffolds are reused exactly. They are not recomputed or
recanonicalized.

Topology describes fused-ring connectivity after removal of chemical identity.
Chemical scaffold preserves the original strict scaffold definition, including
atom identity, heteroatom position, bond type and aromatic representation
accepted by the original strict RDKit sanitization path.

4. UNRESOLVED PURE-CHEMICAL SCAFFOLDS
-------------------------------------
A ring system that lacks a successful pure_ring_chemical / pure_ring_topology
pair is NOT deleted and is NOT assigned an invented chemical scaffold.

Such records are preserved in unresolved_ring_systems.

For known build_pure_ring_scaffolds failures:
  - original failure_stage, error_type and error_message are retained;
  - KekulizeException records receive:
        chemical_status = unresolved_kekulize

A reliable topology fallback is generated directly from the valid complete
molecule and ring-system atom set by:
  - keeping the ring-system connectivity;
  - converting every atom to carbon;
  - converting every internal bond to single;
  - canonicalizing the resulting topology.

This fallback must first reproduce stored pure_ring_topology on a validation
sample. Chemical rescue is deliberately NOT attempted in this strict version.

5. GLOBAL GROUPING
------------------
Across all requested source parts:

Identical topology scaffold SMILES
    -> one deterministic global topology_uid

Within each topology:
Identical strict pure_ring_chemical scaffold SMILES
    -> one deterministic global chemical_uid

The hierarchy for trusted data is:

    topology
        -> strict chemical scaffold
            -> molecule/ring-system instance

Unresolved chemical instances are attached to topology when topology is
available, but they do not receive chemical_uid.

6. FREQUENCY STATISTICS
-----------------------
For every topology the output records:
  - number of strict chemical scaffolds;
  - number of strict scaffold instances;
  - number of unresolved chemical instances.

For every strict chemical scaffold:
  - number of strict molecule/ring-system instances.

Unresolved records are reported separately and never silently discarded.

7. WEIGHT ALLOCATION
--------------------
Weights are calculated ONLY on the strict chemical-scaffold population.

Let:
  N_T = number of topologies containing at least one strict chemical scaffold.

Each such topology receives equal total weight:

  W_T = 1 / N_T

For topology T with N_C(T) strict chemical scaffolds:

  W_C = W_T / N_C(T)

For strict chemical scaffold C with N_I(C) strict instances:

  W_I = W_C / N_I(C)

Therefore:
  - frequent topologies do not receive more total weight;
  - frequent chemical scaffolds do not receive more total weight;
  - a topology is not amplified merely because it has many chemical variants.

Unresolved chemical instances are preserved but excluded from chemical-level
weight allocation because no strict chemical scaffold identity exists for them.

8. OUTPUT BOUNDARY
------------------
The output is a lightweight global scaffold-analysis index containing:
  - topologies;
  - strict chemical scaffolds;
  - strict scaffold instances;
  - unresolved ring systems;
  - per-part audit statistics;
  - global frequencies and weights.

The program does NOT copy:
  - complete molecule tables;
  - atom_facts_json / bond_facts_json;
  - expanded scaffolds;
  - attachment-site tables;
  - other TB-scale source content.

Future operations can return to the original source data by:
  source_part + molecule_uid + ring_system_uid

9. SCIENTIFIC BOUNDARY
----------------------
This version intentionally prioritizes definition consistency over forced
completeness. Strict v4 keeps the validated Strict v2 per-part chemistry/I/O logic, but runs
independent source parts concurrently. Each worker writes a separate derived
part-result DB; only the main process merges them into the final global DB.
Scaffold identities, unresolved policy, output schema and weight definitions
remain unchanged.

If the original strict chemical scaffold cannot be defined because the isolated
ring-system fragment fails RDKit strict sanitization/Kekulization, the record is
kept as chemically unresolved rather than being modified by heuristic hydrogen,
valence, aromaticity or bond-order repairs.

Any future alternative chemical-graph definition for these unresolved records
must be treated as a separately versioned scaffold definition, not silently
mixed into the strict scaffold namespace.
"""

    (out_dir / "GLOBAL_SCAFFOLD_LOGIC.txt").write_text(
        logic,
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Strict v4: part-level parallel execution
# ---------------------------------------------------------------------------

def create_part_result_schema(con: sqlite3.Connection) -> None:
    """
    Per-part result DB uses the same core tables as the final global DB.
    No global weights are finalized inside workers.
    """
    init_output_schema(con)


def worker_process_part_v4(job: Tuple[str, str, str, str, str, int]) -> dict:
    """
    One worker:
      - opens eligible_index.db read-only;
      - opens exactly one source part read-only;
      - writes exactly one small derived part-result DB.

    Workers NEVER write the shared global output DB.
    """
    (
        eligible_index,
        eligible_table,
        source_value,
        part_path_str,
        result_db_str,
        validate_topology_n,
    ) = job

    part_path = Path(part_path_str)
    result_db = Path(result_db_str)

    # Fresh derived worker output.
    if result_db.exists():
        result_db.unlink()
    result_db.parent.mkdir(parents=True, exist_ok=True)

    eligible_con = connect_ro(Path(eligible_index))
    out_con = connect_out(result_db, overwrite=True)
    init_output_schema(out_con)

    try:
        audit = process_part(
            part_path=part_path,
            source_value=source_value,
            eligible_con=eligible_con,
            eligible_table=eligible_table,
            out_con=out_con,
            validate_topology_n=validate_topology_n,
            validation_dir=result_db.parent / "validation" / part_path.stem,
        )

        # Store small worker metadata only.
        out_con.execute(
            "INSERT OR REPLACE INTO run_metadata(key,value) VALUES (?,?)",
            ("source_part", part_path.name),
        )
        out_con.execute(
            "INSERT OR REPLACE INTO run_metadata(key,value) VALUES (?,?)",
            ("worker_completed_at", time.strftime("%Y-%m-%d %H:%M:%S")),
        )
        out_con.commit()

        return {
            "source_part": part_path.name,
            "result_db": str(result_db),
            "audit": audit,
        }
    finally:
        eligible_con.close()
        out_con.close()


def merge_part_result_db(
    global_con: sqlite3.Connection,
    part_result_db: Path,
    alias_no: int,
) -> None:
    """
    Merge one completed derived part DB into the single global output DB.

    Deterministic IDs make INSERT OR IGNORE across parts safe:
      topology_uid = hash(topology_smiles)
      chemical_uid = hash(topology_uid | chemical_smiles)
    """
    alias = f"p{alias_no}"
    global_con.execute(f"ATTACH DATABASE ? AS {alias}", (str(part_result_db),))

    try:
        global_con.execute(
            f"""
            INSERT OR IGNORE INTO topologies(
                topology_uid, topology_smiles,
                strict_chemical_count, strict_instance_count,
                unresolved_instance_count, topology_weight
            )
            SELECT
                topology_uid, topology_smiles,
                0, 0, 0, NULL
            FROM {alias}.topologies
            """
        )

        global_con.execute(
            f"""
            INSERT OR IGNORE INTO chemical_scaffolds(
                chemical_uid, topology_uid, chemical_smiles,
                strict_instance_count, chemical_weight
            )
            SELECT
                chemical_uid, topology_uid, chemical_smiles,
                0, NULL
            FROM {alias}.chemical_scaffolds
            """
        )

        global_con.execute(
            f"""
            INSERT OR REPLACE INTO strict_scaffold_instances(
                source_part, molecule_uid, ring_system_uid, system_index,
                sssr_basis_ring_count, cycle_rank, ring_system_type,
                topology_uid, chemical_uid, instance_weight
            )
            SELECT
                source_part, molecule_uid, ring_system_uid, system_index,
                sssr_basis_ring_count, cycle_rank, ring_system_type,
                topology_uid, chemical_uid, NULL
            FROM {alias}.strict_scaffold_instances
            """
        )

        global_con.execute(
            f"""
            INSERT OR REPLACE INTO unresolved_ring_systems(
                source_part, molecule_uid, ring_system_uid, system_index,
                sssr_basis_ring_count, cycle_rank, ring_system_type,
                topology_uid, topology_smiles, chemical_status,
                original_failure_stage, original_error_type,
                original_error_message
            )
            SELECT
                source_part, molecule_uid, ring_system_uid, system_index,
                sssr_basis_ring_count, cycle_rank, ring_system_type,
                topology_uid, topology_smiles, chemical_status,
                original_failure_stage, original_error_type,
                original_error_message
            FROM {alias}.unresolved_ring_systems
            """
        )

        global_con.execute(
            f"""
            INSERT OR REPLACE INTO part_audit(
                source_part, eligible_molecules, eligible_ring_systems,
                strict_complete_pairs, unresolved_pure_scaffolds,
                unresolved_with_topology, unresolved_without_topology,
                elapsed_seconds
            )
            SELECT
                source_part, eligible_molecules, eligible_ring_systems,
                strict_complete_pairs, unresolved_pure_scaffolds,
                unresolved_with_topology, unresolved_without_topology,
                elapsed_seconds
            FROM {alias}.part_audit
            """
        )

        global_con.commit()
    finally:
        global_con.execute(f"DETACH DATABASE {alias}")


def merge_all_part_results(
    global_con: sqlite3.Connection,
    result_paths: Sequence[Path],
) -> None:
    log(f"Merging {len(result_paths):,} completed part-result DBs into global index ...")
    started = time.time()

    for i, result_db in enumerate(result_paths, start=1):
        merge_part_result_db(global_con, result_db, i)
        if i == 1 or i == len(result_paths) or i % 10 == 0:
            log(f"  merged part results: {i}/{len(result_paths)}")

    log(f"Part-result merge completed in {(time.time()-started)/60:.2f} min")

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument(
        "--eligible-index",
        required=True,
        help="Path to eligible_index.db",
    )
    p.add_argument(
        "--eligible-table",
        default=None,
        help="Eligible table name. If omitted, auto-detect by source_part+molecule_uid.",
    )
    p.add_argument(
        "--parts-dir",
        required=True,
        help="Directory containing part_XXXX.db files",
    )
    p.add_argument(
        "--part",
        default="part_0000.db",
        help="Single part filename for test mode",
    )
    p.add_argument(
        "--all-parts",
        action="store_true",
        help="Process all part_*.db files",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of source parts processed concurrently. Start with 4 on shared storage.",
    )
    p.add_argument(
        "--output-db",
        required=True,
        help="Final global scaffold SQLite DB",
    )
    p.add_argument(
        "--reports-dir",
        required=True,
        help="Directory for reports and temporary per-part result DBs",
    )
    p.add_argument(
        "--validate-topology",
        type=int,
        default=200,
        metavar="N",
        help="Validate topology fallback on N successful ring systems per part; 0 disables",
    )
    p.add_argument(
        "--keep-part-results",
        action="store_true",
        help="Keep temporary derived per-part result DBs after successful global merge",
    )
    p.add_argument(
        "--skip-finalize",
        action="store_true",
        help="Skip global count/weight finalization during debugging",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing final output DB and temporary part-result directory",
    )
    return p.parse_args()

def main() -> int:
    args = parse_args()

    eligible_path = Path(args.eligible_index)
    parts_dir = Path(args.parts_dir)
    out_db = Path(args.output_db)
    reports_dir = Path(args.reports_dir)
    part_results_dir = reports_dir / "_part_results"

    if not eligible_path.exists():
        raise SystemExit(f"eligible_index.db not found: {eligible_path}")
    if not parts_dir.is_dir():
        raise SystemExit(f"parts directory not found: {parts_dir}")
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")
    if args.validate_topology > 0 and Chem is None:
        raise SystemExit("RDKit is required for topology validation/fallback.")

    # Final global output is created only after worker results are ready.
    if out_db.exists():
        if not args.overwrite:
            raise SystemExit(
                f"Output DB already exists: {out_db}\n"
                "Use --overwrite only if you intentionally want to rebuild it."
            )
        out_db.unlink()

    if part_results_dir.exists() and args.overwrite:
        import shutil
        shutil.rmtree(part_results_dir)
    part_results_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    eligible_con = connect_ro(eligible_path)
    try:
        eligible_table = args.eligible_table or find_eligible_table(eligible_con)
        source_values = eligible_source_values(eligible_con, eligible_table)
    finally:
        eligible_con.close()

    log(f"Eligible table: {eligible_table}")
    log(f"Distinct source_part values in eligible index: {len(source_values):,}")

    if args.all_parts:
        part_paths = sorted(parts_dir.glob("part_*.db"))
        if not part_paths:
            raise RuntimeError(f"No part_*.db files found in {parts_dir}")
    else:
        pth = parts_dir / args.part
        if not pth.exists():
            raise RuntimeError(f"Part file not found: {pth}")
        part_paths = [pth]

    log(
        f"Strict v4 part-level parallel mode: "
        f"{len(part_paths)} parts, workers={args.workers}"
    )

    jobs = []
    for part_path in part_paths:
        source_value = resolve_source_part_value(
            part_path.name,
            part_path.stem,
            source_values,
        )
        result_db = part_results_dir / f"{part_path.stem}_result.db"
        jobs.append(
            (
                str(eligible_path),
                eligible_table,
                source_value,
                str(part_path),
                str(result_db),
                int(args.validate_topology),
            )
        )

    started = time.time()
    completed_result_paths: List[Path] = []

    # Each process writes only its own derived result DB.
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        future_map = {
            executor.submit(worker_process_part_v4, job): job[3]
            for job in jobs
        }

        for completed_no, future in enumerate(as_completed(future_map), start=1):
            part_name = Path(future_map[future]).name
            try:
                result = future.result()
            except Exception:
                log(f"ERROR: worker failed for {part_name}; aborting global merge.")
                raise

            completed_result_paths.append(Path(result["result_db"]))
            audit = result["audit"]

            elapsed = max(1e-9, time.time() - started)
            rate = completed_no / elapsed
            remaining = len(jobs) - completed_no
            eta_hours = remaining / rate / 3600 if rate > 0 else float("inf")

            log(
                f"[PART COMPLETE] {completed_no}/{len(jobs)} "
                f"{part_name} | strict={audit['strict_complete_pairs']:,} "
                f"| unresolved={audit['unresolved_pure_scaffolds']:,} "
                f"| part_elapsed={audit['elapsed_seconds']/60:.2f} min "
                f"| ETA~{eta_hours:.2f} h"
            )

    completed_result_paths.sort()

    # Merge only after every worker succeeded.
    global_con = connect_out(out_db, overwrite=True)
    init_output_schema(global_con)

    try:
        global_con.execute(
            "INSERT OR REPLACE INTO run_metadata(key,value) VALUES (?,?)",
            ("eligible_table", eligible_table),
        )
        global_con.execute(
            "INSERT OR REPLACE INTO run_metadata(key,value) VALUES (?,?)",
            ("started_at", time.strftime("%Y-%m-%d %H:%M:%S")),
        )
        global_con.execute(
            "INSERT OR REPLACE INTO run_metadata(key,value) VALUES (?,?)",
            (
                "analysis_policy",
                "strict_chemical_only; unresolved_preserved; no_chemical_rescue",
            ),
        )
        global_con.execute(
            "INSERT OR REPLACE INTO run_metadata(key,value) VALUES (?,?)",
            ("parallel_workers", str(args.workers)),
        )
        global_con.commit()

        merge_all_part_results(global_con, completed_result_paths)

        if args.skip_finalize:
            final_stats = {
                "topology_count_total": global_con.execute(
                    "SELECT COUNT(*) FROM topologies"
                ).fetchone()[0],
                "strict_chemical_count": global_con.execute(
                    "SELECT COUNT(*) FROM chemical_scaffolds"
                ).fetchone()[0],
                "strict_instance_count": global_con.execute(
                    "SELECT COUNT(*) FROM strict_scaffold_instances"
                ).fetchone()[0],
                "unresolved_instance_count": global_con.execute(
                    "SELECT COUNT(*) FROM unresolved_ring_systems"
                ).fetchone()[0],
                "finalization_skipped": True,
            }
        else:
            final_stats = finalize_counts_and_weights(global_con)

        global_con.execute(
            "INSERT OR REPLACE INTO run_metadata(key,value) VALUES (?,?)",
            ("finished_at", time.strftime("%Y-%m-%d %H:%M:%S")),
        )
        global_con.commit()

        write_reports(global_con, reports_dir, final_stats, args)

        log("=== COMPLETE ===")
        log(json.dumps(final_stats, ensure_ascii=False))
        log(f"Output DB: {out_db}")
        log(f"Reports: {reports_dir}")

    finally:
        global_con.close()

    if not args.keep_part_results:
        import shutil
        shutil.rmtree(part_results_dir, ignore_errors=True)
        log("Temporary per-part result DBs removed.")

    log(f"Total wall time: {(time.time()-started)/3600:.2f} h")
    return 0


if __name__ == "__main__":
    sys.exit(main())
