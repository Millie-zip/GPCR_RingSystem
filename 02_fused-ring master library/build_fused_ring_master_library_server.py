#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
构建模型无关、可追溯、可版本化的连续稠环骨架母库。

设计目标
--------
1. 不绑定 DiGress、CoCoGraph 或其他具体生成模型。
2. 完整保留原分子的节点、边、原子索引和来源信息。
3. 严格以“共享环边”定义稠合，避免仅凭共享两个原子将桥环误判为稠环。
4. 对每个连续稠环系统保存：
   - 纯环系
   - 扩展化学骨架
   - 纯环系拓扑骨架
   - 扩展拓扑骨架
   - 带连接位点标记的骨架
5. 保存稠合共享边、环系内部边、连接位点和完整分子到骨架的索引映射。
6. 每次提取均记录算法版本、参数、RDKit 版本和运行 ID。
7. 使用多进程分片数据库，适合服务器长期运行。
8. 母库仅保存“结构事实”和“提取结果”，不写死模型训练权重。

默认输入：
    脚本所在目录/merged_fused_ring_unique.db

默认输出：
    脚本所在目录/fused_ring_master_library/fused_ring_master_library.db

建议服务器运行：
    nohup python -u build_fused_ring_master_library_server.py \
      --source-db merged_fused_ring_unique.db \
      --output-db fused_ring_master_library/fused_ring_master_library.db \
      --workers 30 \
      > fused_ring_master_library.log 2>&1 &

依赖：
    Python 3.10+
    RDKit
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import sqlite3
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem import rdMolDescriptors


EXTRACTION_VERSION = "master_v1_shared_ring_edge"
FUSED_DEFINITION = "two rings are fused only when they share at least one ring bond"
DEFAULT_PROGRESS_EVERY = 100_000


# ---------------------------------------------------------------------------
# 通用工具
# ---------------------------------------------------------------------------

def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=False)


def quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def remove_sqlite_files(path: Path) -> None:
    for candidate in (path, Path(str(path) + "-wal"), Path(str(path) + "-shm")):
        if candidate.exists():
            candidate.unlink()


def sha1_text(text: str, length: int = 20) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:length]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def clear_stereo_and_directions(mol: Chem.Mol) -> None:
    Chem.RemoveStereochemistry(mol)
    for atom in mol.GetAtoms():
        atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    for bond in mol.GetBonds():
        bond.SetStereo(Chem.BondStereo.STEREONONE)
        bond.SetBondDir(Chem.BondDir.NONE)


def bond_type_label(bond: Chem.Bond) -> str:
    bond_type = bond.GetBondType()
    if bond_type == Chem.BondType.SINGLE:
        return "SINGLE"
    if bond_type == Chem.BondType.DOUBLE:
        return "DOUBLE"
    if bond_type == Chem.BondType.TRIPLE:
        return "TRIPLE"
    if bond_type == Chem.BondType.AROMATIC:
        return "AROMATIC"
    return str(bond_type)


def bond_order_value(bond: Chem.Bond) -> float:
    if bond.GetBondType() == Chem.BondType.AROMATIC:
        return 1.5
    return float(bond.GetBondTypeAsDouble())


def canonicalize_molecule(mol: Chem.Mol) -> tuple[str, str]:
    isomeric = Chem.MolToSmiles(
        mol, canonical=True, isomericSmiles=True, kekuleSmiles=False
    )
    nonisomeric = Chem.MolToSmiles(
        mol, canonical=True, isomericSmiles=False, kekuleSmiles=False
    )
    if not isomeric or not nonisomeric:
        raise ValueError("规范化 SMILES 为空。")
    return isomeric, nonisomeric


# ---------------------------------------------------------------------------
# 源数据库发现
# ---------------------------------------------------------------------------

def discover_source_schema(
    conn: sqlite3.Connection,
    requested_table: str | None = None,
) -> tuple[str, dict[str, str | None]]:
    tables = [
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
    ]
    if requested_table:
        if requested_table not in tables:
            raise RuntimeError(f"指定源表不存在：{requested_table}")
        tables = [requested_table]

    smiles_candidates = (
        "canonical_smiles",
        "smiles",
        "canonical_isomeric_smiles",
    )
    id_candidates = ("id", "record_id", "molecule_id", "mol_id")
    cid_candidates = ("cid", "coconut_id", "identifier", "name")
    mw_candidates = ("mw", "molecular_weight", "mol_weight", "exact_mass")
    heavy_candidates = ("heavy_atoms", "heavy_atom_count", "num_heavy_atoms")

    for table in tables:
        columns = [
            row[1]
            for row in conn.execute(
                f"PRAGMA table_info({quote_identifier(table)})"
            )
        ]
        lower_to_real = {column.lower(): column for column in columns}

        def choose(candidates: tuple[str, ...]) -> str | None:
            return next(
                (lower_to_real[c] for c in candidates if c in lower_to_real),
                None,
            )

        smiles_col = choose(smiles_candidates)
        if smiles_col:
            return table, {
                "smiles": smiles_col,
                "id": choose(id_candidates),
                "cid": choose(cid_candidates),
                "mw": choose(mw_candidates),
                "heavy_atoms": choose(heavy_candidates),
            }

    raise RuntimeError(
        "没有找到含 SMILES 字段的数据表。支持字段名："
        + ", ".join(smiles_candidates)
    )


def source_rowid_bounds(
    conn: sqlite3.Connection,
    table: str,
    limit: int | None,
) -> tuple[int, int] | None:
    qtable = quote_identifier(table)
    lo, hi = conn.execute(
        f"SELECT MIN(rowid), MAX(rowid) FROM {qtable}"
    ).fetchone()
    if lo is None or hi is None:
        return None

    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit 必须大于 0。")
        target = conn.execute(
            f"SELECT rowid FROM {qtable} ORDER BY rowid LIMIT 1 OFFSET ?",
            (limit - 1,),
        ).fetchone()
        if target is not None:
            hi = min(hi, int(target[0]))
    return int(lo), int(hi)


def make_rowid_ranges(
    lo: int,
    hi: int,
    workers: int,
    ranges_per_worker: int = 4,
) -> list[tuple[int, int]]:
    desired_ranges = max(1, workers * ranges_per_worker)
    span = max(1, (hi - lo + 1 + desired_ranges - 1) // desired_ranges)
    return [
        (start, min(hi, start + span - 1))
        for start in range(lo, hi + 1, span)
    ]


# ---------------------------------------------------------------------------
# 环系识别
# ---------------------------------------------------------------------------

class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a == root_b:
            return
        if self.rank[root_a] < self.rank[root_b]:
            root_a, root_b = root_b, root_a
        self.parent[root_b] = root_a
        if self.rank[root_a] == self.rank[root_b]:
            self.rank[root_a] += 1


@dataclass(frozen=True)
class RingBasisEntry:
    ring_index: int
    atom_indices: tuple[int, ...]
    bond_indices: tuple[int, ...]


@dataclass(frozen=True)
class FusedRingSystem:
    ring_indices: tuple[int, ...]
    atom_indices: tuple[int, ...]
    bond_indices: tuple[int, ...]
    ring_sizes: tuple[int, ...]
    fused_ring_pair_count: int
    shared_atom_indices: tuple[int, ...]
    shared_bond_indices: tuple[int, ...]
    shared_bond_atom_pairs: tuple[tuple[int, int], ...]
    bridge_contact_pair_count: int
    cycle_rank: int


def build_ring_basis(mol: Chem.Mol) -> list[RingBasisEntry]:
    ring_info = mol.GetRingInfo()
    atom_rings = list(ring_info.AtomRings())
    bond_rings = list(ring_info.BondRings())

    if len(atom_rings) != len(bond_rings):
        raise RuntimeError(
            "RDKit AtomRings 与 BondRings 数量不一致，无法建立环基。"
        )

    return [
        RingBasisEntry(
            ring_index=i,
            atom_indices=tuple(sorted(map(int, atom_ring))),
            bond_indices=tuple(sorted(map(int, bond_ring))),
        )
        for i, (atom_ring, bond_ring) in enumerate(
            zip(atom_rings, bond_rings)
        )
    ]


def induced_bond_indices(
    mol: Chem.Mol,
    atom_indices: Iterable[int],
) -> tuple[int, ...]:
    keep = set(map(int, atom_indices))
    return tuple(
        sorted(
            bond.GetIdx()
            for bond in mol.GetBonds()
            if bond.GetBeginAtomIdx() in keep
            and bond.GetEndAtomIdx() in keep
        )
    )


def find_strict_fused_ring_systems(
    mol: Chem.Mol,
) -> list[FusedRingSystem]:
    """
    严格定义：
    两个环只有在共享至少一条“环键”时才视为稠合。

    仅共享两个或更多原子、但不共享环键的环对被记录为桥接接触，
    不用于合并连续稠环系统。
    """
    basis = build_ring_basis(mol)
    ring_count = len(basis)
    if ring_count < 2:
        return []

    atom_sets = [set(entry.atom_indices) for entry in basis]
    bond_sets = [set(entry.bond_indices) for entry in basis]

    uf = UnionFind(ring_count)
    fused_relations: list[tuple[int, int, set[int], set[int]]] = []
    bridge_contacts: list[tuple[int, int, set[int]]] = []

    for i in range(ring_count):
        for j in range(i + 1, ring_count):
            shared_bonds = bond_sets[i] & bond_sets[j]
            shared_atoms = atom_sets[i] & atom_sets[j]

            if shared_bonds:
                uf.union(i, j)
                fused_relations.append(
                    (i, j, shared_atoms, shared_bonds)
                )
            elif len(shared_atoms) >= 2:
                bridge_contacts.append((i, j, shared_atoms))

    if not fused_relations:
        return []

    components: dict[int, list[int]] = {}
    for ring_idx in range(ring_count):
        root = uf.find(ring_idx)
        components.setdefault(root, []).append(ring_idx)

    systems: list[FusedRingSystem] = []

    for component_ring_indices in components.values():
        if len(component_ring_indices) < 2:
            continue

        component_set = set(component_ring_indices)
        atom_indices: set[int] = set()
        basis_bond_indices: set[int] = set()
        shared_atoms: set[int] = set()
        shared_bonds: set[int] = set()
        fused_pair_count = 0
        bridge_contact_count = 0

        for ring_idx in component_ring_indices:
            atom_indices.update(basis[ring_idx].atom_indices)
            basis_bond_indices.update(basis[ring_idx].bond_indices)

        for i, j, pair_shared_atoms, pair_shared_bonds in fused_relations:
            if i in component_set and j in component_set:
                fused_pair_count += 1
                shared_atoms.update(pair_shared_atoms)
                shared_bonds.update(pair_shared_bonds)

        for i, j, _pair_shared_atoms in bridge_contacts:
            if i in component_set and j in component_set:
                bridge_contact_count += 1

        # 环系内部所有诱导边用于图论 cycle rank。
        all_internal_bonds = induced_bond_indices(mol, atom_indices)
        vertex_count = len(atom_indices)
        edge_count = len(all_internal_bonds)
        cycle_rank = edge_count - vertex_count + 1

        shared_bond_atom_pairs: list[tuple[int, int]] = []
        for bond_idx in sorted(shared_bonds):
            bond = mol.GetBondWithIdx(bond_idx)
            shared_bond_atom_pairs.append(
                tuple(
                    sorted(
                        (
                            int(bond.GetBeginAtomIdx()),
                            int(bond.GetEndAtomIdx()),
                        )
                    )
                )
            )

        systems.append(
            FusedRingSystem(
                ring_indices=tuple(sorted(component_ring_indices)),
                atom_indices=tuple(sorted(atom_indices)),
                bond_indices=tuple(sorted(all_internal_bonds)),
                ring_sizes=tuple(
                    sorted(
                        len(basis[idx].atom_indices)
                        for idx in component_ring_indices
                    )
                ),
                fused_ring_pair_count=fused_pair_count,
                shared_atom_indices=tuple(sorted(shared_atoms)),
                shared_bond_indices=tuple(sorted(shared_bonds)),
                shared_bond_atom_pairs=tuple(shared_bond_atom_pairs),
                bridge_contact_pair_count=bridge_contact_count,
                cycle_rank=int(cycle_rank),
            )
        )

    systems.sort(
        key=lambda item: (
            item.atom_indices[0],
            len(item.ring_indices),
            item.atom_indices,
        )
    )
    return systems


# ---------------------------------------------------------------------------
# 分子事实与骨架构建
# ---------------------------------------------------------------------------

def atom_fact_records(mol: Chem.Mol) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for atom in mol.GetAtoms():
        bond_order_sum = sum(
            bond_order_value(bond) for bond in atom.GetBonds()
        )
        records.append(
            {
                "atom_index": atom.GetIdx(),
                "atomic_number": atom.GetAtomicNum(),
                "symbol": atom.GetSymbol(),
                "formal_charge": atom.GetFormalCharge(),
                "is_aromatic": bool(atom.GetIsAromatic()),
                "isotope": atom.GetIsotope(),
                "chiral_tag": str(atom.GetChiralTag()),
                "degree": atom.GetDegree(),
                "total_degree": atom.GetTotalDegree(),
                "explicit_h_count": atom.GetNumExplicitHs(),
                "implicit_h_count": atom.GetNumImplicitHs(),
                "total_h_count": atom.GetTotalNumHs(),
                "total_valence": atom.GetTotalValence(),
                "bond_order_sum": bond_order_sum,
            }
        )
    return records


def bond_fact_records(mol: Chem.Mol) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for bond in mol.GetBonds():
        begin = int(bond.GetBeginAtomIdx())
        end = int(bond.GetEndAtomIdx())
        records.append(
            {
                "bond_index": bond.GetIdx(),
                "begin_atom_index": begin,
                "end_atom_index": end,
                "atom_pair": sorted((begin, end)),
                "bond_type": bond_type_label(bond),
                "bond_order": bond_order_value(bond),
                "is_aromatic": bool(bond.GetIsAromatic()),
                "is_conjugated": bool(bond.GetIsConjugated()),
                "is_in_ring": bool(bond.IsInRing()),
                "stereo": str(bond.GetStereo()),
            }
        )
    return records


def degree_sequence(mol: Chem.Mol) -> list[int]:
    return [atom.GetDegree() for atom in mol.GetAtoms()]


def bond_order_sum_sequence(mol: Chem.Mol) -> list[float]:
    return [
        sum(bond_order_value(bond) for bond in atom.GetBonds())
        for atom in mol.GetAtoms()
    ]


def formal_charge_sequence(mol: Chem.Mol) -> list[int]:
    return [atom.GetFormalCharge() for atom in mol.GetAtoms()]


def expand_chemical_scaffold_atoms(
    mol: Chem.Mol,
    ring_atom_indices: Iterable[int],
) -> tuple[int, ...]:
    """
    在纯环系基础上纳入直接通过双键或三键连接的环外原子。
    这通常用于保留环外羰基、亚胺等对化学身份重要的多键原子。
    """
    scaffold_atoms = set(map(int, ring_atom_indices))
    for atom_idx in tuple(scaffold_atoms):
        atom = mol.GetAtomWithIdx(atom_idx)
        for bond in atom.GetBonds():
            other_idx = bond.GetOtherAtomIdx(atom_idx)
            if other_idx in scaffold_atoms:
                continue
            if bond.GetBondType() in (
                Chem.BondType.DOUBLE,
                Chem.BondType.TRIPLE,
            ):
                scaffold_atoms.add(int(other_idx))
    return tuple(sorted(scaffold_atoms))


def build_induced_submol(
    mol: Chem.Mol,
    atom_indices: Iterable[int],
    *,
    preserve_atom_maps: bool = True,
) -> tuple[Chem.Mol, dict[int, int], dict[int, int]]:
    """
    建立诱导子图。

    不对芳香 N 做经验性加氢修复，避免静默改变骨架化学身份。
    若 sanitize 失败，调用者可改用 dummy 封端版本或记录失败。

    返回：
        submol
        full_to_submol
        submol_to_full
    """
    keep_atoms = tuple(sorted(set(map(int, atom_indices))))
    if not keep_atoms:
        raise ValueError("子图原子集合为空。")

    atom_count = mol.GetNumAtoms()
    if keep_atoms[0] < 0 or keep_atoms[-1] >= atom_count:
        raise IndexError("子图原子索引超出原分子范围。")

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

    submol_to_full: dict[int, int] = {}
    full_to_submol: dict[int, int] = {}
    for atom in submol.GetAtoms():
        full_idx = atom.GetAtomMapNum() - 1
        if full_idx < 0:
            raise RuntimeError("子图原子映射号丢失。")
        sub_idx = atom.GetIdx()
        submol_to_full[sub_idx] = full_idx
        full_to_submol[full_idx] = sub_idx

    return submol, full_to_submol, submol_to_full


def copy_without_atom_maps(mol: Chem.Mol) -> Chem.Mol:
    result = Chem.Mol(mol)
    for atom in result.GetAtoms():
        atom.SetAtomMapNum(0)
    result.UpdatePropertyCache(strict=False)
    return result


def canonical_scaffold_smiles(
    submol_with_maps: Chem.Mol,
) -> tuple[str, str]:
    """
    返回：
        无 atom-map 的规范化骨架 SMILES（用于去重）
        带 full atom map 的规范化 SMILES（用于追踪）
    """
    mapped = Chem.Mol(submol_with_maps)
    mapped.UpdatePropertyCache(strict=False)

    # 先尝试严格 sanitize。
    Chem.SanitizeMol(mapped)

    mapped_smiles = Chem.MolToSmiles(
        mapped,
        canonical=True,
        isomericSmiles=False,
        kekuleSmiles=False,
    )
    if not mapped_smiles:
        raise ValueError("带映射骨架 SMILES 为空。")

    unmapped = copy_without_atom_maps(mapped)
    Chem.SanitizeMol(unmapped)
    smiles = Chem.MolToSmiles(
        unmapped,
        canonical=True,
        isomericSmiles=False,
        kekuleSmiles=False,
    )
    if not smiles:
        raise ValueError("规范化骨架 SMILES 为空。")

    check = Chem.MolFromSmiles(smiles, sanitize=True)
    if check is None:
        raise ValueError(f"骨架无法回读：{smiles}")

    return smiles, mapped_smiles


def build_topology_scaffold(
    chemical_submol_with_maps: Chem.Mol,
) -> tuple[Chem.Mol, str, str]:
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
    smiles, mapped_smiles = canonical_scaffold_smiles(topology)
    return topology, smiles, mapped_smiles


@dataclass(frozen=True)
class AttachmentRecord:
    scaffold_full_atom_index: int
    external_full_atom_index: int
    bond_index: int
    bond_type: str
    bond_order: float


def find_attachment_records(
    mol: Chem.Mol,
    scaffold_atom_indices: Iterable[int],
) -> list[AttachmentRecord]:
    keep = set(map(int, scaffold_atom_indices))
    records: list[AttachmentRecord] = []

    for bond in mol.GetBonds():
        begin = int(bond.GetBeginAtomIdx())
        end = int(bond.GetEndAtomIdx())
        begin_in = begin in keep
        end_in = end in keep

        if begin_in == end_in:
            continue

        scaffold_idx = begin if begin_in else end
        external_idx = end if begin_in else begin
        records.append(
            AttachmentRecord(
                scaffold_full_atom_index=scaffold_idx,
                external_full_atom_index=external_idx,
                bond_index=bond.GetIdx(),
                bond_type=bond_type_label(bond),
                bond_order=bond_order_value(bond),
            )
        )

    records.sort(
        key=lambda item: (
            item.scaffold_full_atom_index,
            item.external_full_atom_index,
            item.bond_index,
        )
    )
    return records


def build_marked_scaffold(
    mol: Chem.Mol,
    scaffold_atom_indices: Iterable[int],
    attachments: list[AttachmentRecord],
) -> tuple[str, str]:
    """
    在骨架边界添加 dummy 原子。

    dummy 的 atom-map 编号按：
        骨架 full atom index
        外部 full atom index
        bond index
    的稳定排序编号。

    真正的连接位点身份仍以 attachment_sites 表中的 full atom index 为准；
    marked SMILES 仅作为便于查看和分组的派生表示。
    """
    keep_atoms = tuple(sorted(set(map(int, scaffold_atom_indices))))
    keep_set = set(keep_atoms)

    work = Chem.Mol(mol)
    clear_stereo_and_directions(work)
    for atom in work.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)

    rw = Chem.RWMol(work)

    for map_number, attachment in enumerate(attachments, start=1):
        dummy = Chem.Atom(0)
        dummy.SetAtomMapNum(10_000_000 + map_number)
        dummy.SetNoImplicit(True)
        dummy_idx = rw.AddAtom(dummy)

        original_bond = mol.GetBondWithIdx(attachment.bond_index)
        marked_bond_type = (
            Chem.BondType.SINGLE
            if original_bond.GetBondType() == Chem.BondType.AROMATIC
            else original_bond.GetBondType()
        )
        rw.AddBond(
            attachment.scaffold_full_atom_index,
            dummy_idx,
            marked_bond_type,
        )

    original_atom_count = mol.GetNumAtoms()
    for atom_idx in range(original_atom_count - 1, -1, -1):
        if atom_idx not in keep_set:
            rw.RemoveAtom(atom_idx)

    marked = rw.GetMol()
    clear_stereo_and_directions(marked)
    marked.UpdatePropertyCache(strict=False)
    Chem.SanitizeMol(marked)

    mapped_smiles = Chem.MolToSmiles(
        marked,
        canonical=True,
        isomericSmiles=False,
        kekuleSmiles=False,
    )

    unmapped = copy_without_atom_maps(marked)
    Chem.SanitizeMol(unmapped)
    smiles = Chem.MolToSmiles(
        unmapped,
        canonical=True,
        isomericSmiles=False,
        kekuleSmiles=False,
    )
    return smiles, mapped_smiles


def scaffold_bond_records(
    mol: Chem.Mol,
    atom_indices: Iterable[int],
) -> list[dict[str, Any]]:
    keep = set(map(int, atom_indices))
    records: list[dict[str, Any]] = []
    for bond in mol.GetBonds():
        begin = int(bond.GetBeginAtomIdx())
        end = int(bond.GetEndAtomIdx())
        if begin in keep and end in keep:
            records.append(
                {
                    "bond_index": bond.GetIdx(),
                    "atom_pair": sorted((begin, end)),
                    "bond_type": bond_type_label(bond),
                    "bond_order": bond_order_value(bond),
                    "is_aromatic": bool(bond.GetIsAromatic()),
                }
            )
    return records


# ---------------------------------------------------------------------------
# 数据库定义
# ---------------------------------------------------------------------------

CORE_SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS extraction_runs (
    run_id TEXT PRIMARY KEY,
    extraction_version TEXT NOT NULL,
    fused_definition TEXT NOT NULL,
    parameters_json TEXT NOT NULL,
    rdkit_version TEXT NOT NULL,
    python_version TEXT NOT NULL,
    platform TEXT NOT NULL,
    source_database TEXT NOT NULL,
    source_table TEXT NOT NULL,
    started_at_utc TEXT NOT NULL,
    finished_at_utc TEXT,
    status TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS molecules (
    molecule_uid TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    source_rowid INTEGER NOT NULL,
    source_record_id TEXT,
    source_cid TEXT,
    original_smiles TEXT NOT NULL,
    canonical_isomeric_smiles TEXT NOT NULL,
    canonical_nonisomeric_smiles TEXT NOT NULL,
    molecular_formula TEXT NOT NULL,
    atom_count INTEGER NOT NULL,
    heavy_atom_count INTEGER NOT NULL,
    bond_count INTEGER NOT NULL,
    formal_charge INTEGER NOT NULL,
    source_mw REAL,
    calculated_exact_mw REAL,
    atom_facts_json TEXT NOT NULL,
    bond_facts_json TEXT NOT NULL,
    degree_sequence_json TEXT NOT NULL,
    bond_order_sum_sequence_json TEXT NOT NULL,
    formal_charge_sequence_json TEXT NOT NULL,
    created_at_utc TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES extraction_runs(run_id)
);

CREATE TABLE IF NOT EXISTS ring_systems (
    ring_system_uid TEXT PRIMARY KEY,
    molecule_uid TEXT NOT NULL,
    run_id TEXT NOT NULL,
    system_index INTEGER NOT NULL,
    system_count_in_molecule INTEGER NOT NULL,
    basis_ring_indices_json TEXT NOT NULL,
    basis_ring_sizes_json TEXT NOT NULL,
    ring_atom_indices_json TEXT NOT NULL,
    ring_bond_indices_json TEXT NOT NULL,
    shared_atom_indices_json TEXT NOT NULL,
    shared_bond_indices_json TEXT NOT NULL,
    shared_bond_atom_pairs_json TEXT NOT NULL,
    fused_ring_pair_count INTEGER NOT NULL,
    bridge_contact_pair_count INTEGER NOT NULL,
    sssr_basis_ring_count INTEGER NOT NULL,
    cycle_rank INTEGER NOT NULL,
    ring_system_type TEXT NOT NULL,
    UNIQUE (molecule_uid, run_id, system_index),
    FOREIGN KEY (molecule_uid) REFERENCES molecules(molecule_uid),
    FOREIGN KEY (run_id) REFERENCES extraction_runs(run_id)
);

CREATE TABLE IF NOT EXISTS scaffold_variants (
    scaffold_occurrence_uid TEXT PRIMARY KEY,
    ring_system_uid TEXT NOT NULL,
    molecule_uid TEXT NOT NULL,
    run_id TEXT NOT NULL,
    variant_type TEXT NOT NULL,
    scaffold_smiles TEXT NOT NULL,
    mapped_scaffold_smiles TEXT,
    full_atom_indices_json TEXT NOT NULL,
    full_bond_records_json TEXT NOT NULL,
    full_to_submol_map_json TEXT NOT NULL,
    submol_to_full_map_json TEXT NOT NULL,
    scaffold_atom_count INTEGER NOT NULL,
    scaffold_bond_count INTEGER NOT NULL,
    sanitize_status TEXT NOT NULL,
    UNIQUE (ring_system_uid, variant_type),
    FOREIGN KEY (ring_system_uid) REFERENCES ring_systems(ring_system_uid),
    FOREIGN KEY (molecule_uid) REFERENCES molecules(molecule_uid),
    FOREIGN KEY (run_id) REFERENCES extraction_runs(run_id)
);

CREATE TABLE IF NOT EXISTS attachment_sites (
    attachment_uid TEXT PRIMARY KEY,
    ring_system_uid TEXT NOT NULL,
    molecule_uid TEXT NOT NULL,
    run_id TEXT NOT NULL,
    scaffold_variant_type TEXT NOT NULL,
    attachment_order INTEGER NOT NULL,
    scaffold_full_atom_index INTEGER NOT NULL,
    external_full_atom_index INTEGER NOT NULL,
    original_bond_index INTEGER NOT NULL,
    original_bond_type TEXT NOT NULL,
    original_bond_order REAL NOT NULL,
    UNIQUE (
        ring_system_uid,
        scaffold_variant_type,
        scaffold_full_atom_index,
        external_full_atom_index,
        original_bond_index
    ),
    FOREIGN KEY (ring_system_uid) REFERENCES ring_systems(ring_system_uid),
    FOREIGN KEY (molecule_uid) REFERENCES molecules(molecule_uid),
    FOREIGN KEY (run_id) REFERENCES extraction_runs(run_id)
);

CREATE TABLE IF NOT EXISTS failures (
    failure_uid TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    source_rowid INTEGER,
    source_record_id TEXT,
    source_cid TEXT,
    smiles TEXT,
    molecule_uid TEXT,
    system_index INTEGER,
    failure_stage TEXT NOT NULL,
    error_type TEXT NOT NULL,
    error_message TEXT NOT NULL,
    created_at_utc TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES extraction_runs(run_id)
);

CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_molecules_canonical_iso
    ON molecules(canonical_isomeric_smiles);
CREATE INDEX IF NOT EXISTS idx_molecules_formula
    ON molecules(molecular_formula);
CREATE INDEX IF NOT EXISTS idx_ring_systems_cycle_rank
    ON ring_systems(cycle_rank);
CREATE INDEX IF NOT EXISTS idx_ring_systems_type
    ON ring_systems(ring_system_type);
CREATE INDEX IF NOT EXISTS idx_scaffold_variant_type
    ON scaffold_variants(variant_type);
CREATE INDEX IF NOT EXISTS idx_scaffold_smiles
    ON scaffold_variants(scaffold_smiles);
CREATE INDEX IF NOT EXISTS idx_attachment_ring_system
    ON attachment_sites(ring_system_uid);
"""


def initialize_database(
    conn: sqlite3.Connection,
    *,
    fast_part_mode: bool,
) -> None:
    if fast_part_mode:
        conn.executescript(
            """
            PRAGMA journal_mode=OFF;
            PRAGMA synchronous=OFF;
            PRAGMA temp_store=MEMORY;
            PRAGMA locking_mode=EXCLUSIVE;
            """
        )
    else:
        conn.executescript(
            """
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=NORMAL;
            PRAGMA temp_store=MEMORY;
            PRAGMA foreign_keys=ON;
            """
        )
    conn.executescript(CORE_SCHEMA)
    if fast_part_mode:
        conn.execute("PRAGMA foreign_keys=OFF")
    conn.commit()


def create_summary_views(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        DROP VIEW IF EXISTS scaffold_catalog_summary;
        CREATE VIEW scaffold_catalog_summary AS
        SELECT
            run_id,
            variant_type,
            scaffold_smiles,
            COUNT(*) AS occurrence_count,
            COUNT(DISTINCT molecule_uid) AS molecule_count,
            COUNT(DISTINCT ring_system_uid) AS ring_system_count
        FROM scaffold_variants
        GROUP BY run_id, variant_type, scaffold_smiles;

        DROP VIEW IF EXISTS topology_to_chemical_summary;
        CREATE VIEW topology_to_chemical_summary AS
        WITH chemical AS (
            SELECT
                ring_system_uid,
                scaffold_smiles AS chemical_scaffold_smiles
            FROM scaffold_variants
            WHERE variant_type = 'expanded_chemical'
        ),
        topology AS (
            SELECT
                ring_system_uid,
                scaffold_smiles AS topology_scaffold_smiles
            FROM scaffold_variants
            WHERE variant_type = 'expanded_topology'
        )
        SELECT
            t.topology_scaffold_smiles,
            COUNT(DISTINCT c.chemical_scaffold_smiles)
                AS chemical_scaffold_count,
            COUNT(DISTINCT c.ring_system_uid)
                AS ring_system_occurrence_count
        FROM topology AS t
        JOIN chemical AS c
          ON c.ring_system_uid = t.ring_system_uid
        GROUP BY t.topology_scaffold_smiles;

        DROP VIEW IF EXISTS ring_system_statistics;
        CREATE VIEW ring_system_statistics AS
        SELECT
            run_id,
            cycle_rank,
            sssr_basis_ring_count,
            ring_system_type,
            COUNT(*) AS ring_system_count,
            COUNT(DISTINCT molecule_uid) AS molecule_count
        FROM ring_systems
        GROUP BY
            run_id,
            cycle_rank,
            sssr_basis_ring_count,
            ring_system_type;
        """
    )
    conn.commit()


# ---------------------------------------------------------------------------
# 单分子处理
# ---------------------------------------------------------------------------

@dataclass
class ProcessResult:
    molecule_row: tuple[Any, ...] | None
    ring_system_rows: list[tuple[Any, ...]]
    scaffold_rows: list[tuple[Any, ...]]
    attachment_rows: list[tuple[Any, ...]]
    failure_rows: list[tuple[Any, ...]]
    parsed: bool
    has_fused_system: bool


def failure_row(
    *,
    run_id: str,
    source_rowid: int | None,
    source_record_id: Any,
    source_cid: Any,
    smiles: str | None,
    molecule_uid: str | None,
    system_index: int | None,
    failure_stage: str,
    exc: BaseException | str,
) -> tuple[Any, ...]:
    if isinstance(exc, BaseException):
        error_type = type(exc).__name__
        error_message = repr(exc)
    else:
        error_type = "RecordedFailure"
        error_message = str(exc)

    raw = "|".join(
        [
            run_id,
            str(source_rowid),
            str(system_index),
            failure_stage,
            error_message,
        ]
    )
    failure_uid = "FAIL_" + sha1_text(raw, 24)

    return (
        failure_uid,
        run_id,
        source_rowid,
        None if source_record_id is None else str(source_record_id),
        None if source_cid is None else str(source_cid),
        smiles,
        molecule_uid,
        system_index,
        failure_stage,
        error_type,
        error_message,
        utc_now_iso(),
    )


def make_scaffold_row(
    *,
    occurrence_uid: str,
    ring_system_uid: str,
    molecule_uid: str,
    run_id: str,
    variant_type: str,
    scaffold_smiles: str,
    mapped_scaffold_smiles: str | None,
    atom_indices: Iterable[int],
    mol: Chem.Mol,
    full_to_submol: dict[int, int],
    submol_to_full: dict[int, int],
    sanitize_status: str,
) -> tuple[Any, ...]:
    atom_indices_tuple = tuple(sorted(set(map(int, atom_indices))))
    bond_records = scaffold_bond_records(mol, atom_indices_tuple)
    return (
        occurrence_uid,
        ring_system_uid,
        molecule_uid,
        run_id,
        variant_type,
        scaffold_smiles,
        mapped_scaffold_smiles,
        json_dumps(list(atom_indices_tuple)),
        json_dumps(bond_records),
        json_dumps({str(k): v for k, v in full_to_submol.items()}),
        json_dumps({str(k): v for k, v in submol_to_full.items()}),
        len(atom_indices_tuple),
        len(bond_records),
        sanitize_status,
    )


def process_one_molecule(
    *,
    run_id: str,
    source_rowid: int,
    source_record_id: Any,
    source_cid: Any,
    original_smiles: str | None,
    source_mw: Any,
    source_heavy_atoms: Any,
) -> ProcessResult:
    ring_system_rows: list[tuple[Any, ...]] = []
    scaffold_rows: list[tuple[Any, ...]] = []
    attachment_rows: list[tuple[Any, ...]] = []
    failure_rows: list[tuple[Any, ...]] = []

    if original_smiles is None or not str(original_smiles).strip():
        failure_rows.append(
            failure_row(
                run_id=run_id,
                source_rowid=source_rowid,
                source_record_id=source_record_id,
                source_cid=source_cid,
                smiles=original_smiles,
                molecule_uid=None,
                system_index=None,
                failure_stage="missing_smiles",
                exc="SMILES 为空。",
            )
        )
        return ProcessResult(
            None, [], [], [], failure_rows, False, False
        )

    smiles = str(original_smiles).strip()
    mol = Chem.MolFromSmiles(smiles, sanitize=True)
    if mol is None:
        failure_rows.append(
            failure_row(
                run_id=run_id,
                source_rowid=source_rowid,
                source_record_id=source_record_id,
                source_cid=source_cid,
                smiles=smiles,
                molecule_uid=None,
                system_index=None,
                failure_stage="parse_smiles",
                exc="RDKit 无法解析并 sanitize 原始 SMILES。",
            )
        )
        return ProcessResult(
            None, [], [], [], failure_rows, False, False
        )

    try:
        canonical_iso, canonical_noniso = canonicalize_molecule(mol)
        molecule_uid = f"MOL_{source_rowid:014d}"
        formula = rdMolDescriptors.CalcMolFormula(mol)
        calculated_exact_mw = rdMolDescriptors.CalcExactMolWt(mol)
        total_formal_charge = sum(
            atom.GetFormalCharge() for atom in mol.GetAtoms()
        )

        molecule_row = (
            molecule_uid,
            run_id,
            source_rowid,
            None if source_record_id is None else str(source_record_id),
            None if source_cid is None else str(source_cid),
            smiles,
            canonical_iso,
            canonical_noniso,
            formula,
            mol.GetNumAtoms(),
            mol.GetNumHeavyAtoms(),
            mol.GetNumBonds(),
            total_formal_charge,
            None if source_mw is None else float(source_mw),
            float(calculated_exact_mw),
            json_dumps(atom_fact_records(mol)),
            json_dumps(bond_fact_records(mol)),
            json_dumps(degree_sequence(mol)),
            json_dumps(bond_order_sum_sequence(mol)),
            json_dumps(formal_charge_sequence(mol)),
            utc_now_iso(),
        )
    except Exception as exc:
        failure_rows.append(
            failure_row(
                run_id=run_id,
                source_rowid=source_rowid,
                source_record_id=source_record_id,
                source_cid=source_cid,
                smiles=smiles,
                molecule_uid=None,
                system_index=None,
                failure_stage="molecule_facts",
                exc=exc,
            )
        )
        return ProcessResult(
            None, [], [], [], failure_rows, True, False
        )

    try:
        systems = find_strict_fused_ring_systems(mol)
    except Exception as exc:
        failure_rows.append(
            failure_row(
                run_id=run_id,
                source_rowid=source_rowid,
                source_record_id=source_record_id,
                source_cid=source_cid,
                smiles=smiles,
                molecule_uid=molecule_uid,
                system_index=None,
                failure_stage="find_strict_fused_ring_systems",
                exc=exc,
            )
        )
        return ProcessResult(
            molecule_row, [], [], [], failure_rows, True, False
        )

    if not systems:
        failure_rows.append(
            failure_row(
                run_id=run_id,
                source_rowid=source_rowid,
                source_record_id=source_record_id,
                source_cid=source_cid,
                smiles=smiles,
                molecule_uid=molecule_uid,
                system_index=None,
                failure_stage="no_strict_fused_system",
                exc="未识别到共享环边定义下的连续稠环系统。",
            )
        )
        return ProcessResult(
            molecule_row, [], [], [], failure_rows, True, False
        )

    for system_index, system in enumerate(systems, start=1):
        ring_system_uid = (
            f"RS_{source_rowid:014d}_{system_index:03d}"
        )

        ring_system_rows.append(
            (
                ring_system_uid,
                molecule_uid,
                run_id,
                system_index,
                len(systems),
                json_dumps(list(system.ring_indices)),
                json_dumps(list(system.ring_sizes)),
                json_dumps(list(system.atom_indices)),
                json_dumps(list(system.bond_indices)),
                json_dumps(list(system.shared_atom_indices)),
                json_dumps(list(system.shared_bond_indices)),
                json_dumps(
                    [list(pair) for pair in system.shared_bond_atom_pairs]
                ),
                system.fused_ring_pair_count,
                system.bridge_contact_pair_count,
                len(system.ring_indices),
                system.cycle_rank,
                "strict_fused",
            )
        )

        pure_atoms = tuple(system.atom_indices)
        expanded_atoms = expand_chemical_scaffold_atoms(
            mol, pure_atoms
        )

        try:
            pure_sub, pure_f2s, pure_s2f = build_induced_submol(
                mol, pure_atoms
            )
            pure_smiles, pure_mapped = canonical_scaffold_smiles(
                pure_sub
            )
            scaffold_rows.append(
                make_scaffold_row(
                    occurrence_uid=ring_system_uid + "_PURE_CHEM",
                    ring_system_uid=ring_system_uid,
                    molecule_uid=molecule_uid,
                    run_id=run_id,
                    variant_type="pure_ring_chemical",
                    scaffold_smiles=pure_smiles,
                    mapped_scaffold_smiles=pure_mapped,
                    atom_indices=pure_atoms,
                    mol=mol,
                    full_to_submol=pure_f2s,
                    submol_to_full=pure_s2f,
                    sanitize_status="strict_sanitize_ok",
                )
            )

            _, pure_topo_smiles, pure_topo_mapped = (
                build_topology_scaffold(pure_sub)
            )
            scaffold_rows.append(
                make_scaffold_row(
                    occurrence_uid=ring_system_uid + "_PURE_TOPO",
                    ring_system_uid=ring_system_uid,
                    molecule_uid=molecule_uid,
                    run_id=run_id,
                    variant_type="pure_ring_topology",
                    scaffold_smiles=pure_topo_smiles,
                    mapped_scaffold_smiles=pure_topo_mapped,
                    atom_indices=pure_atoms,
                    mol=mol,
                    full_to_submol=pure_f2s,
                    submol_to_full=pure_s2f,
                    sanitize_status="strict_sanitize_ok",
                )
            )
        except Exception as exc:
            failure_rows.append(
                failure_row(
                    run_id=run_id,
                    source_rowid=source_rowid,
                    source_record_id=source_record_id,
                    source_cid=source_cid,
                    smiles=smiles,
                    molecule_uid=molecule_uid,
                    system_index=system_index,
                    failure_stage="build_pure_ring_scaffolds",
                    exc=exc,
                )
            )

        try:
            exp_sub, exp_f2s, exp_s2f = build_induced_submol(
                mol, expanded_atoms
            )
            exp_smiles, exp_mapped = canonical_scaffold_smiles(
                exp_sub
            )
            scaffold_rows.append(
                make_scaffold_row(
                    occurrence_uid=ring_system_uid + "_EXP_CHEM",
                    ring_system_uid=ring_system_uid,
                    molecule_uid=molecule_uid,
                    run_id=run_id,
                    variant_type="expanded_chemical",
                    scaffold_smiles=exp_smiles,
                    mapped_scaffold_smiles=exp_mapped,
                    atom_indices=expanded_atoms,
                    mol=mol,
                    full_to_submol=exp_f2s,
                    submol_to_full=exp_s2f,
                    sanitize_status="strict_sanitize_ok",
                )
            )

            _, exp_topo_smiles, exp_topo_mapped = (
                build_topology_scaffold(exp_sub)
            )
            scaffold_rows.append(
                make_scaffold_row(
                    occurrence_uid=ring_system_uid + "_EXP_TOPO",
                    ring_system_uid=ring_system_uid,
                    molecule_uid=molecule_uid,
                    run_id=run_id,
                    variant_type="expanded_topology",
                    scaffold_smiles=exp_topo_smiles,
                    mapped_scaffold_smiles=exp_topo_mapped,
                    atom_indices=expanded_atoms,
                    mol=mol,
                    full_to_submol=exp_f2s,
                    submol_to_full=exp_s2f,
                    sanitize_status="strict_sanitize_ok",
                )
            )

            attachments = find_attachment_records(
                mol, expanded_atoms
            )
            marked_smiles, marked_mapped = build_marked_scaffold(
                mol,
                expanded_atoms,
                attachments,
            )
            scaffold_rows.append(
                make_scaffold_row(
                    occurrence_uid=ring_system_uid + "_MARKED",
                    ring_system_uid=ring_system_uid,
                    molecule_uid=molecule_uid,
                    run_id=run_id,
                    variant_type="marked_attachment",
                    scaffold_smiles=marked_smiles,
                    mapped_scaffold_smiles=marked_mapped,
                    atom_indices=expanded_atoms,
                    mol=mol,
                    full_to_submol=exp_f2s,
                    submol_to_full=exp_s2f,
                    sanitize_status="strict_sanitize_ok",
                )
            )

            for attachment_order, attachment in enumerate(
                attachments, start=1
            ):
                attachment_uid = (
                    f"ATT_{source_rowid:014d}_{system_index:03d}_"
                    f"{attachment_order:03d}"
                )
                attachment_rows.append(
                    (
                        attachment_uid,
                        ring_system_uid,
                        molecule_uid,
                        run_id,
                        "expanded_chemical",
                        attachment_order,
                        attachment.scaffold_full_atom_index,
                        attachment.external_full_atom_index,
                        attachment.bond_index,
                        attachment.bond_type,
                        attachment.bond_order,
                    )
                )
        except Exception as exc:
            failure_rows.append(
                failure_row(
                    run_id=run_id,
                    source_rowid=source_rowid,
                    source_record_id=source_record_id,
                    source_cid=source_cid,
                    smiles=smiles,
                    molecule_uid=molecule_uid,
                    system_index=system_index,
                    failure_stage="build_expanded_scaffolds",
                    exc=exc,
                )
            )

    return ProcessResult(
        molecule_row=molecule_row,
        ring_system_rows=ring_system_rows,
        scaffold_rows=scaffold_rows,
        attachment_rows=attachment_rows,
        failure_rows=failure_rows,
        parsed=True,
        has_fused_system=True,
    )


# ---------------------------------------------------------------------------
# 分片工作进程
# ---------------------------------------------------------------------------

MOLECULE_INSERT = """
INSERT OR REPLACE INTO molecules VALUES (
    ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
)
"""

RING_SYSTEM_INSERT = """
INSERT OR REPLACE INTO ring_systems VALUES (
    ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
)
"""

SCAFFOLD_INSERT = """
INSERT OR REPLACE INTO scaffold_variants VALUES (
    ?,?,?,?,?,?,?,?,?,?,?,?,?,?
)
"""

ATTACHMENT_INSERT = """
INSERT OR REPLACE INTO attachment_sites VALUES (
    ?,?,?,?,?,?,?,?,?,?,?
)
"""

FAILURE_INSERT = """
INSERT OR REPLACE INTO failures VALUES (
    ?,?,?,?,?,?,?,?,?,?,?,?
)
"""


def worker_process(args: tuple[Any, ...]) -> dict[str, Any]:
    (
        worker_job_id,
        rowid_start,
        rowid_end,
        run_id,
        source_db,
        source_table,
        source_columns,
        part_db,
        batch_size,
        show_rdkit_logs,
    ) = args

    if not show_rdkit_logs:
        RDLogger.DisableLog("rdApp.error")
        RDLogger.DisableLog("rdApp.warning")

    part_path = Path(part_db)
    remove_sqlite_files(part_path)

    stats = {
        "job_id": worker_job_id,
        "read": 0,
        "parsed": 0,
        "molecules_written": 0,
        "with_fused_system": 0,
        "ring_systems": 0,
        "scaffolds": 0,
        "attachments": 0,
        "failures": 0,
        "part": str(part_path),
    }

    qtable = quote_identifier(source_table)
    smiles_expr = quote_identifier(source_columns["smiles"])
    id_expr = (
        quote_identifier(source_columns["id"])
        if source_columns["id"]
        else "rowid"
    )
    cid_expr = (
        quote_identifier(source_columns["cid"])
        if source_columns["cid"]
        else "NULL"
    )
    mw_expr = (
        quote_identifier(source_columns["mw"])
        if source_columns["mw"]
        else "NULL"
    )
    heavy_expr = (
        quote_identifier(source_columns["heavy_atoms"])
        if source_columns["heavy_atoms"]
        else "NULL"
    )

    query = (
        f"SELECT rowid, {id_expr}, {cid_expr}, {smiles_expr}, "
        f"{mw_expr}, {heavy_expr} "
        f"FROM {qtable} "
        f"WHERE rowid BETWEEN ? AND ? "
        f"ORDER BY rowid"
    )

    molecule_batch: list[tuple[Any, ...]] = []
    ring_batch: list[tuple[Any, ...]] = []
    scaffold_batch: list[tuple[Any, ...]] = []
    attachment_batch: list[tuple[Any, ...]] = []
    failure_batch: list[tuple[Any, ...]] = []

    def flush(out: sqlite3.Connection) -> None:
        if molecule_batch:
            out.executemany(MOLECULE_INSERT, molecule_batch)
            molecule_batch.clear()
        if ring_batch:
            out.executemany(RING_SYSTEM_INSERT, ring_batch)
            ring_batch.clear()
        if scaffold_batch:
            out.executemany(SCAFFOLD_INSERT, scaffold_batch)
            scaffold_batch.clear()
        if attachment_batch:
            out.executemany(ATTACHMENT_INSERT, attachment_batch)
            attachment_batch.clear()
        if failure_batch:
            out.executemany(FAILURE_INSERT, failure_batch)
            failure_batch.clear()
        out.commit()

    with sqlite3.connect(source_db) as source, sqlite3.connect(
        part_path
    ) as out:
        source.execute("PRAGMA query_only=ON")
        source.execute("PRAGMA mmap_size=536870912")
        initialize_database(out, fast_part_mode=True)

        for (
            source_rowid,
            source_record_id,
            source_cid,
            smiles,
            source_mw,
            source_heavy_atoms,
        ) in source.execute(query, (rowid_start, rowid_end)):
            stats["read"] += 1

            try:
                result = process_one_molecule(
                    run_id=run_id,
                    source_rowid=int(source_rowid),
                    source_record_id=source_record_id,
                    source_cid=source_cid,
                    original_smiles=smiles,
                    source_mw=source_mw,
                    source_heavy_atoms=source_heavy_atoms,
                )
            except Exception as exc:
                result = ProcessResult(
                    molecule_row=None,
                    ring_system_rows=[],
                    scaffold_rows=[],
                    attachment_rows=[],
                    failure_rows=[
                        failure_row(
                            run_id=run_id,
                            source_rowid=int(source_rowid),
                            source_record_id=source_record_id,
                            source_cid=source_cid,
                            smiles=None if smiles is None else str(smiles),
                            molecule_uid=None,
                            system_index=None,
                            failure_stage="unexpected_worker_error",
                            exc=exc,
                        )
                    ],
                    parsed=False,
                    has_fused_system=False,
                )

            stats["parsed"] += int(result.parsed)
            stats["with_fused_system"] += int(
                result.has_fused_system
            )

            if result.molecule_row is not None:
                molecule_batch.append(result.molecule_row)
                stats["molecules_written"] += 1

            ring_batch.extend(result.ring_system_rows)
            scaffold_batch.extend(result.scaffold_rows)
            attachment_batch.extend(result.attachment_rows)
            failure_batch.extend(result.failure_rows)

            stats["ring_systems"] += len(result.ring_system_rows)
            stats["scaffolds"] += len(result.scaffold_rows)
            stats["attachments"] += len(result.attachment_rows)
            stats["failures"] += len(result.failure_rows)

            if (
                len(molecule_batch)
                + len(ring_batch)
                + len(scaffold_batch)
                + len(attachment_batch)
                + len(failure_batch)
                >= batch_size
            ):
                flush(out)

        flush(out)

    return stats


# ---------------------------------------------------------------------------
# 合并与导出
# ---------------------------------------------------------------------------

def merge_part_databases(
    output_conn: sqlite3.Connection,
    part_paths: list[Path],
) -> None:
    output_conn.execute("PRAGMA synchronous=OFF")
    output_conn.execute("PRAGMA journal_mode=OFF")
    output_conn.execute("PRAGMA temp_store=MEMORY")
    output_conn.commit()

    tables = (
        "molecules",
        "ring_systems",
        "scaffold_variants",
        "attachment_sites",
        "failures",
    )

    for index, part_path in enumerate(part_paths, start=1):
        alias = f"partdb{index}"
        output_conn.execute(
            f"ATTACH DATABASE ? AS {alias}",
            (str(part_path),),
        )
        for table in tables:
            output_conn.execute(
                f"INSERT OR REPLACE INTO {table} "
                f"SELECT * FROM {alias}.{table}"
            )
        output_conn.commit()
        output_conn.execute(f"DETACH DATABASE {alias}")
        print(
            f"[MERGE] 已合并分片 {index}/{len(part_paths)}："
            f"{part_path.name}",
            flush=True,
        )


def export_table_or_view_to_csv(
    conn: sqlite3.Connection,
    name: str,
    output_path: Path,
) -> int:
    cursor = conn.execute(f"SELECT * FROM {quote_identifier(name)}")
    headers = [column[0] for column in cursor.description]
    row_count = 0
    with output_path.open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for row in cursor:
            writer.writerow(row)
            row_count += 1
    return row_count


def write_run_metadata(
    conn: sqlite3.Connection,
    metadata: dict[str, Any],
) -> None:
    conn.executemany(
        "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
        [(key, str(value)) for key, value in metadata.items()],
    )
    conn.commit()


# ---------------------------------------------------------------------------
# 命令行
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_source = script_dir / "merged_fused_ring_unique.db"
    default_output = (
        script_dir
        / "fused_ring_master_library"
        / "fused_ring_master_library.db"
    )

    parser = argparse.ArgumentParser(
        description="构建模型无关的连续稠环骨架母库。"
    )
    parser.add_argument(
        "--source-db",
        type=Path,
        default=default_source,
        help=f"源 SQLite 数据库，默认：{default_source}",
    )
    parser.add_argument(
        "--source-table",
        default=None,
        help="可选：指定源表；不指定时自动寻找含 SMILES 的表。",
    )
    parser.add_argument(
        "--output-db",
        type=Path,
        default=default_output,
        help=f"输出母库，默认：{default_output}",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 4) - 2),
        help="并行进程数。",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="仅处理前 N 条，适合测试。",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5000,
        help="每个分片累计多少行后写盘。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已有输出数据库和分片目录。",
    )
    parser.add_argument(
        "--keep-parts",
        action="store_true",
        help="合并后保留分片数据库，便于排错。",
    )
    parser.add_argument(
        "--export-csv",
        action="store_true",
        help="导出关键摘要 CSV；全量明细默认只保存在 SQLite。",
    )
    parser.add_argument(
        "--show-rdkit-logs",
        action="store_true",
        help="显示 RDKit warning/error 日志。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    source_db = args.source_db.resolve()
    output_db = args.output_db.resolve()
    output_dir = output_db.parent
    part_dir = output_dir / "_parts_master_v1"

    if not source_db.exists():
        raise FileNotFoundError(f"找不到源数据库：{source_db}")

    if args.workers < 1:
        raise ValueError("--workers 必须至少为 1。")
    if args.batch_size < 100:
        raise ValueError("--batch-size 建议至少为 100。")

    output_dir.mkdir(parents=True, exist_ok=True)

    if args.overwrite:
        remove_sqlite_files(output_db)
        if part_dir.exists():
            shutil.rmtree(part_dir)
    elif output_db.exists():
        raise FileExistsError(
            f"输出数据库已存在：{output_db}\n"
            "若确认覆盖，请增加 --overwrite。"
        )

    part_dir.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(source_db) as source_conn:
        source_table, source_columns = discover_source_schema(
            source_conn,
            args.source_table,
        )
        bounds = source_rowid_bounds(
            source_conn,
            source_table,
            args.limit,
        )
        if bounds is None:
            raise RuntimeError("源表为空。")
        lo, hi = bounds
        total_estimate = source_conn.execute(
            f"SELECT COUNT(*) FROM {quote_identifier(source_table)} "
            f"WHERE rowid BETWEEN ? AND ?",
            (lo, hi),
        ).fetchone()[0]

    run_id = (
        EXTRACTION_VERSION
        + "_"
        + datetime.now().strftime("%Y%m%d_%H%M%S")
    )

    parameters = {
        "strict_fused_by_shared_ring_edge": True,
        "include_exocyclic_double_triple_atoms": True,
        "save_pure_and_expanded_scaffolds": True,
        "save_full_graph_facts": True,
        "save_atom_maps": True,
        "aromatic_n_boundary_repair": False,
        "workers": args.workers,
        "limit": args.limit,
        "batch_size": args.batch_size,
    }

    remove_sqlite_files(output_db)
    with sqlite3.connect(output_db) as output_conn:
        initialize_database(output_conn, fast_part_mode=False)
        output_conn.execute(
            """
            INSERT INTO extraction_runs (
                run_id, extraction_version, fused_definition,
                parameters_json, rdkit_version, python_version,
                platform, source_database, source_table,
                started_at_utc, finished_at_utc, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                EXTRACTION_VERSION,
                FUSED_DEFINITION,
                json_dumps(parameters),
                rdBase.rdkitVersion,
                sys.version,
                platform.platform(),
                str(source_db),
                source_table,
                utc_now_iso(),
                None,
                "running",
            ),
        )
        output_conn.commit()

    ranges = make_rowid_ranges(
        lo,
        hi,
        workers=args.workers,
        ranges_per_worker=4,
    )

    jobs: list[tuple[Any, ...]] = []
    for job_id, (start, end) in enumerate(ranges):
        part_path = part_dir / f"part_{job_id:04d}.db"
        jobs.append(
            (
                job_id,
                start,
                end,
                run_id,
                str(source_db),
                source_table,
                source_columns,
                str(part_path),
                args.batch_size,
                args.show_rdkit_logs,
            )
        )

    print("=" * 88, flush=True)
    print("连续稠环骨架母库构建开始", flush=True)
    print("run_id:", run_id, flush=True)
    print("源数据库:", source_db, flush=True)
    print("源表:", source_table, flush=True)
    print("源字段:", source_columns, flush=True)
    print("估计处理记录数:", f"{total_estimate:,}", flush=True)
    print("并行进程数:", args.workers, flush=True)
    print("任务分片数:", len(jobs), flush=True)
    print("输出数据库:", output_db, flush=True)
    print("分片目录:", part_dir, flush=True)
    print("=" * 88, flush=True)

    totals = {
        "read": 0,
        "parsed": 0,
        "molecules_written": 0,
        "with_fused_system": 0,
        "ring_systems": 0,
        "scaffolds": 0,
        "attachments": 0,
        "failures": 0,
    }

    started = time.time()
    completed_parts: list[Path] = []

    with ProcessPoolExecutor(
        max_workers=args.workers
    ) as executor:
        futures = [
            executor.submit(worker_process, job) for job in jobs
        ]

        for completed_index, future in enumerate(
            as_completed(futures), start=1
        ):
            stats = future.result()
            completed_parts.append(Path(stats["part"]))

            for key in totals:
                totals[key] += int(stats[key])

            elapsed = max(1e-9, time.time() - started)
            rate = totals["read"] / elapsed
            remaining = max(0, total_estimate - totals["read"])
            eta_hours = (
                remaining / rate / 3600 if rate > 0 else float("inf")
            )

            print(
                "[PROGRESS] "
                f"分片 {completed_index}/{len(jobs)} | "
                f"读取 {totals['read']:,}/{total_estimate:,} | "
                f"速度 {rate:,.1f} 分子/秒 | "
                f"严格稠环分子 {totals['with_fused_system']:,} | "
                f"环系 {totals['ring_systems']:,} | "
                f"失败记录 {totals['failures']:,} | "
                f"ETA {eta_hours:.2f} 小时",
                flush=True,
            )

    completed_parts = sorted(completed_parts)

    with sqlite3.connect(output_db) as output_conn:
        merge_part_databases(output_conn, completed_parts)
        create_summary_views(output_conn)

        output_conn.execute(
            """
            UPDATE extraction_runs
            SET finished_at_utc = ?, status = ?
            WHERE run_id = ?
            """,
            (utc_now_iso(), "completed", run_id),
        )

        # 重新建立正常运行参数。
        output_conn.executescript(
            """
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=NORMAL;
            ANALYZE;
            """
        )

        counts = {
            "molecule_count": output_conn.execute(
                "SELECT COUNT(*) FROM molecules"
            ).fetchone()[0],
            "ring_system_count": output_conn.execute(
                "SELECT COUNT(*) FROM ring_systems"
            ).fetchone()[0],
            "scaffold_occurrence_count": output_conn.execute(
                "SELECT COUNT(*) FROM scaffold_variants"
            ).fetchone()[0],
            "attachment_site_count": output_conn.execute(
                "SELECT COUNT(*) FROM attachment_sites"
            ).fetchone()[0],
            "failure_count": output_conn.execute(
                "SELECT COUNT(*) FROM failures"
            ).fetchone()[0],
            "unique_expanded_chemical_scaffold_count":
                output_conn.execute(
                    """
                    SELECT COUNT(DISTINCT scaffold_smiles)
                    FROM scaffold_variants
                    WHERE variant_type='expanded_chemical'
                    """
                ).fetchone()[0],
            "unique_expanded_topology_scaffold_count":
                output_conn.execute(
                    """
                    SELECT COUNT(DISTINCT scaffold_smiles)
                    FROM scaffold_variants
                    WHERE variant_type='expanded_topology'
                    """
                ).fetchone()[0],
        }

        metadata = {
            "run_id": run_id,
            "extraction_version": EXTRACTION_VERSION,
            "fused_definition": FUSED_DEFINITION,
            "source_database": source_db,
            "source_table": source_table,
            "source_columns": json_dumps(source_columns),
            "rdkit_version": rdBase.rdkitVersion,
            "python_version": sys.version,
            "workers": args.workers,
            "limit": args.limit,
            "total_estimated_source_rows": total_estimate,
            **counts,
            "mother_library_policy":
                "store structural facts and versioned extraction results; "
                "do not bind to a specific diffusion model or weight scheme",
        }
        write_run_metadata(output_conn, metadata)

        exported: dict[str, int] = {}
        if args.export_csv:
            export_targets = (
                "ring_system_statistics",
                "scaffold_catalog_summary",
                "topology_to_chemical_summary",
                "metadata",
                "failures",
            )
            for name in export_targets:
                csv_path = output_dir / f"{name}.csv"
                exported[name] = export_table_or_view_to_csv(
                    output_conn, name, csv_path
                )
                print(
                    f"[CSV] {name}: {exported[name]:,} 行 -> "
                    f"{csv_path}",
                    flush=True,
                )

    if not args.keep_parts:
        shutil.rmtree(part_dir, ignore_errors=True)

    elapsed_hours = (time.time() - started) / 3600

    print("=" * 88, flush=True)
    print("连续稠环骨架母库构建完成", flush=True)
    print("run_id:", run_id, flush=True)
    print("总耗时:", f"{elapsed_hours:.2f} 小时", flush=True)
    for key, value in totals.items():
        print(f"{key}: {value:,}", flush=True)
    print("输出数据库:", output_db, flush=True)
    print(
        "分片数据库:",
        "已保留" if args.keep_parts else "已删除",
        flush=True,
    )
    print("=" * 88, flush=True)


if __name__ == "__main__":
    main()
