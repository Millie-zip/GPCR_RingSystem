#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GPCR Pocket Admissibility Prefilter - FINAL PARALLEL V1

Target: human 5-HT2C
Receptor: 8DPF active-state prepared receptor

V2 changes:
1. ligand-anchored connected free-space cavity grid
2. lorcaserin orthosteric anchor requirement
3. cavity-targeted rigid-body placement
4. per-scaffold hard timeout
5. conformer-generation fallbacks
6. conservative CONFIDENT_FAIL
7. source DB read-only; output written separately
"""

from pathlib import Path
import sqlite3
import argparse
import hashlib
import subprocess
import sys
import os
import csv
import json
import random
import time
import multiprocessing as mp
import queue as queue_module
from collections import Counter, deque

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdDistGeom, rdMolDescriptors
from rdkit import rdBase

from gpcr_ringgen.preprocessing.paths import data_root

PROJECT_DIR = data_root()
SOURCE_DB = PROJECT_DIR / "global_scaffold_analysis_strict_full_v4" / "global_scaffold_index.db"
RECEPTOR_PDB = PROJECT_DIR / "8DPF_V1_prepared" / "8DPF_receptor_V1_prepared.pdb"
LORCASERIN_PDB = PROJECT_DIR / "8DPF_V1_prepared" / "8DPF_lorcaserin_reference_raw.pdb"
OUTPUT_DIR = PROJECT_DIR / "gpcr_pocket_prefilter_final_parallel_v1"
OUTPUT_DB = OUTPUT_DIR / "gpcr_pocket_prefilter_final_parallel_v1.db"
OUTPUT_CSV = OUTPUT_DIR / "gpcr_pocket_prefilter_final_parallel_v1.csv"
DB_SCHEMA_REPORT = OUTPUT_DIR / "database_schema_detected.txt"
RUN_REPORT = OUTPUT_DIR / "run_report.txt"
CAVITY_GRID_PDB = OUTPUT_DIR / "8DPF_lorcaserin_anchored_cavity_grid.pdb"

MASTER_RANDOM_SEED = 20260810

# Parallel production execution.
# 16 workers gives an ideal ~16x speedup versus the validated serial test.
DEFAULT_PARALLEL_WORKERS = 16
SHARD_DIR = OUTPUT_DIR / "shards"
LOG_DIR = OUTPUT_DIR / "logs"
PER_SCAFFOLD_TIMEOUT_SEC = 60
MAX_CONFORMERS = 10
ETKDG_RANDOM_SEED = 20260810
UFF_MAX_ITERS = 250
POSES_PER_CONFORMER = 500
GRID_SPACING_A = 0.75
LIGAND_EXPANSION_A = 3.25
GRID_PROTEIN_VDW_SCALE = 0.85
LIGAND_SEED_DISTANCE_A = 1.25
POCKET_RECEPTOR_RADIUS_A = 13.0
PLACEMENT_TARGET_CENTER_RADIUS_A = 3.25
PLACEMENT_JITTER_A = 0.40
CAVITY_NEIGHBOR_CELL_RADIUS = 0
CORE_ANCHOR_DISTANCE_A = 2.5
PASS_ANCHOR_FRAC = 0.20
BORDERLINE_ANCHOR_FRAC = 0.10
SEVERE_CLASH_SCALE = 0.65
SOFT_CLASH_SCALE = 0.80
BASE_PASS_SEVERE_ATOM_FRAC = 0.05
BASE_PASS_SOFT_ATOM_FRAC = 0.35
NATIVE_SEVERE_MARGIN = 0.03
NATIVE_SOFT_MARGIN = 0.15
PASS_CAVITY_FRAC = 0.85
BORDERLINE_SEVERE_ATOM_FRAC = 0.18
BORDERLINE_SOFT_ATOM_FRAC = 0.60
BORDERLINE_CAVITY_FRAC = 0.60
CONFIDENT_FAIL_MAX_CAVITY_FRAC = 0.40
CONFIDENT_FAIL_MIN_SEVERE_FRAC = 0.30

# Whole-scaffold size compatibility relative to the actual connected cavity.
# Measured along the cavity principal axes.
PASS_MAX_SPAN_EXCESS_A = 1.00
BORDERLINE_MAX_SPAN_EXCESS_A = 2.00

CONFIDENT_FAIL_MIN_SPAN_EXCESS_A = 3.00

random.seed(MASTER_RANDOM_SEED)
np.random.seed(MASTER_RANDOM_SEED)

VDW_RADII = {
    "H": 1.20, "B": 1.92, "C": 1.70, "N": 1.55, "O": 1.52,
    "F": 1.47, "P": 1.80, "S": 1.80, "CL": 1.75, "BR": 1.85,
    "I": 1.98, "SI": 2.10, "SE": 1.90,
}

def get_vdw_radius(element):
    return VDW_RADII.get(element.strip().upper(), 1.80)

def infer_element_from_pdb_line(line):
    element = line[76:78].strip() if len(line) >= 78 else ""
    if element:
        return element.upper()
    atom_name = line[12:16].strip()
    letters = "".join(x for x in atom_name if x.isalpha())
    if not letters:
        return "C"
    if len(letters) >= 2 and letters[:2].upper() in VDW_RADII:
        return letters[:2].upper()
    return letters[0].upper()

def read_pdb_atoms(filename, include_atom=True, include_hetatm=True, heavy_only=True):
    atoms = []
    with open(filename, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            record = line[0:6].strip()
            if record == "ATOM" and not include_atom:
                continue
            if record == "HETATM" and not include_hetatm:
                continue
            if record not in {"ATOM", "HETATM"}:
                continue
            try:
                x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
            except Exception:
                continue
            element = infer_element_from_pdb_line(line)
            if heavy_only and element == "H":
                continue
            atoms.append({
                "record": record,
                "atom_name": line[12:16].strip(),
                "resname": line[17:20].strip(),
                "chain": line[21:22].strip(),
                "resseq": line[22:26].strip(),
                "element": element,
                "coord": np.array([x, y, z], dtype=float),
                "radius": get_vdw_radius(element),
            })
    return atoms

def get_table_names(conn):
    return [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()]

def get_table_columns(conn, table):
    rows = conn.execute(f'PRAGMA table_info("{table}")').fetchall()
    return [{"cid": r[0], "name": r[1], "type": r[2], "notnull": r[3], "default": r[4], "pk": r[5]} for r in rows]



def build_ligand_anchored_cavity_grid(receptor_atoms, ligand_atoms):
    ligand_coords = np.array([a["coord"] for a in ligand_atoms], dtype=float)
    receptor_coords = np.array([a["coord"] for a in receptor_atoms], dtype=float)
    receptor_radii = np.array([a["radius"] for a in receptor_atoms], dtype=float)
    grid_min = ligand_coords.min(axis=0) - LIGAND_EXPANSION_A
    grid_max = ligand_coords.max(axis=0) + LIGAND_EXPANSION_A
    xs = np.arange(grid_min[0], grid_max[0] + GRID_SPACING_A * 0.5, GRID_SPACING_A)
    ys = np.arange(grid_min[1], grid_max[1] + GRID_SPACING_A * 0.5, GRID_SPACING_A)
    zs = np.arange(grid_min[2], grid_max[2] + GRID_SPACING_A * 0.5, GRID_SPACING_A)
    free_indices = set(); seed_indices = set()
    ligand_expansion2 = LIGAND_EXPANSION_A ** 2
    seed2 = LIGAND_SEED_DISTANCE_A ** 2
    occupied_cutoff2 = (GRID_PROTEIN_VDW_SCALE * receptor_radii) ** 2
    for ix, x in enumerate(xs):
        for iy, y in enumerate(ys):
            for iz, z in enumerate(zs):
                p = np.array([x, y, z], dtype=float)
                dl2 = np.sum((ligand_coords - p) ** 2, axis=1)
                if float(dl2.min()) > ligand_expansion2:
                    continue
                dr2 = np.sum((receptor_coords - p) ** 2, axis=1)
                if np.any(dr2 < occupied_cutoff2):
                    continue
                idx = (ix, iy, iz)
                free_indices.add(idx)
                if float(dl2.min()) <= seed2:
                    seed_indices.add(idx)
    if not free_indices:
        raise RuntimeError("No free cavity-grid points generated.")
    if not seed_indices:
        free_list = list(free_indices)
        free_xyz = np.array([[xs[ix], ys[iy], zs[iz]] for ix, iy, iz in free_list], dtype=float)
        nearest = set()
        for lig in ligand_coords:
            d2 = np.sum((free_xyz - lig) ** 2, axis=1)
            nearest.add(int(np.argmin(d2)))
        seed_indices = {free_list[i] for i in nearest}
    offsets = [(dx,dy,dz) for dx in (-1,0,1) for dy in (-1,0,1) for dz in (-1,0,1) if (dx,dy,dz)!=(0,0,0)]
    cavity = set(); q = deque(seed_indices)
    while q:
        cur = q.popleft()
        if cur in cavity or cur not in free_indices:
            continue
        cavity.add(cur)
        cx, cy, cz = cur
        for dx,dy,dz in offsets:
            nxt = (cx+dx, cy+dy, cz+dz)
            if nxt in free_indices and nxt not in cavity:
                q.append(nxt)
    points = np.array([[xs[ix], ys[iy], zs[iz]] for ix,iy,iz in cavity], dtype=float)
    return {
        "indices": cavity,
        "points": points,
        "origin": np.array([xs[0], ys[0], zs[0]], dtype=float),
        "n_free": len(free_indices),
        "n_seed": len(seed_indices),
        "n_cavity": len(cavity),
    }

def compute_cavity_principal_frame(points):
    center = points.mean(axis=0)
    centered = points - center
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axes = eigvecs[:, np.argsort(eigvals)[::-1]]
    projected = centered @ axes
    spans = projected.max(axis=0) - projected.min(axis=0)
    return center, axes, spans

def write_cavity_grid_pdb(points, filename):
    with open(filename, "w", encoding="utf-8") as f:
        for i, p in enumerate(points, start=1):
            serial = ((i - 1) % 99999) + 1
            f.write(
                f"HETATM{serial:5d}  C   GRD Z{1:4d}    "
                f"{p[0]:8.3f}{p[1]:8.3f}{p[2]:8.3f}  1.00  0.00           C\n"
            )
        f.write("END\n")

def point_in_cavity(coord, cavity_indices, grid_origin):
    center = np.rint((coord - grid_origin) / GRID_SPACING_A).astype(int)
    r = CAVITY_NEIGHBOR_CELL_RADIUS
    for dx in range(-r, r+1):
        for dy in range(-r, r+1):
            for dz in range(-r, r+1):
                if (int(center[0]+dx), int(center[1]+dy), int(center[2]+dz)) in cavity_indices:
                    return True
    return False

def random_rotation_matrix(rng):
    q = rng.normal(size=4); q /= np.linalg.norm(q)
    w,x,y,z = q
    return np.array([
        [1-2*y*y-2*z*z, 2*x*y-2*z*w, 2*x*z+2*y*w],
        [2*x*y+2*z*w, 1-2*x*x-2*z*z, 2*y*z-2*x*w],
        [2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x*x-2*y*y],
    ], dtype=float)

def get_heavy_atom_data(mol, conf_id):
    conf = mol.GetConformer(conf_id)
    coords = []; radii = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() in (0,1):
            continue
        p = conf.GetAtomPosition(atom.GetIdx())
        coords.append([p.x,p.y,p.z])
        radii.append(get_vdw_radius(atom.GetSymbol()))
    return np.array(coords,dtype=float), np.array(radii,dtype=float)

def evaluate_pose(scaffold_coords, scaffold_radii, receptor_coords, receptor_radii,
                  ligand_coords, cavity_indices, grid_origin,
                  cavity_frame_center, cavity_axes, cavity_spans):
    delta = scaffold_coords[:,None,:] - receptor_coords[None,:,:]
    d2 = np.sum(delta*delta, axis=2)
    radius_sum = scaffold_radii[:,None] + receptor_radii[None,:]
    severe = np.any(d2 < (SEVERE_CLASH_SCALE * radius_sum)**2, axis=1)
    soft = np.any(d2 < (SOFT_CLASH_SCALE * radius_sum)**2, axis=1)
    n = len(scaffold_coords)
    severe_count = int(severe.sum()); soft_count = int(soft.sum())
    severe_frac = severe_count/n; soft_frac = soft_count/n
    cavity_frac = float(np.mean([point_in_cavity(c,cavity_indices,grid_origin) for c in scaffold_coords]))
    ligand_d2 = np.sum((scaffold_coords[:,None,:]-ligand_coords[None,:,:])**2, axis=2)
    anchor_frac = float(np.any(ligand_d2 <= CORE_ANCHOR_DISTANCE_A**2, axis=1).mean())
    min_distance = float(np.sqrt(d2.min()))

    scaffold_proj = (scaffold_coords - cavity_frame_center) @ cavity_axes
    scaffold_spans = scaffold_proj.max(axis=0) - scaffold_proj.min(axis=0)
    span_excess = np.maximum(scaffold_spans - cavity_spans, 0.0)
    max_span_excess = float(span_excess.max())
    span_fit = max_span_excess <= PASS_MAX_SPAN_EXCESS_A

    score = (
        severe_frac*100
        + soft_frac*15
        + (1-cavity_frac)*25
        + (1-anchor_frac)*10
        + max_span_excess*8
    )
    return {
        "score": score,
        "severe_atom_count": severe_count,
        "severe_atom_frac": severe_frac,
        "soft_atom_count": soft_count,
        "soft_atom_frac": soft_frac,
        "cavity_frac": cavity_frac,
        "anchor_frac": anchor_frac,
        "min_distance": min_distance,
        "max_span_excess_A": max_span_excess,
        "span_fit": span_fit,
    }

def _embed_multiple(mol3d, mode, seed):
    if mode == 1:
        params = rdDistGeom.ETKDGv3(); params.useRandomCoords=False; params.pruneRmsThresh=0.50
    elif mode == 2:
        params = rdDistGeom.ETKDGv3(); params.useRandomCoords=True; params.pruneRmsThresh=0.30
    else:
        params = rdDistGeom.ETKDGv2(); params.useRandomCoords=True; params.pruneRmsThresh=0.10; params.enforceChirality=True
    params.randomSeed = int(seed)
    params.numThreads = 1
    params.maxIterations = 2000
    return list(rdDistGeom.EmbedMultipleConfs(mol3d, numConfs=MAX_CONFORMERS, params=params))

def generate_conformers(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("RDKit MolFromSmiles failed")
    heavy_atoms = mol.GetNumHeavyAtoms()
    if heavy_atoms == 0:
        raise ValueError("No heavy atoms")
    dummy_count = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum()==0)
    mol3d = Chem.AddHs(mol)
    conf_ids = []; embed_mode = None
    for mode in (1,2,3):
        conf_ids = _embed_multiple(mol3d, mode, ETKDG_RANDOM_SEED + mode)
        if conf_ids:
            embed_mode = mode
            break
    if not conf_ids:
        raise RuntimeError("3D conformer generation failed after all fallbacks")
    uff_status = "NOT_RUN"
    try:
        if AllChem.UFFHasAllMoleculeParams(mol3d):
            AllChem.UFFOptimizeMoleculeConfs(mol3d, numThreads=1, maxIters=UFF_MAX_ITERS)
            uff_status = "OK"
        else:
            uff_status = "MISSING_UFF_PARAMETERS"
    except Exception as e:
        uff_status = f"UFF_ERROR:{type(e).__name__}:{e}"
    return {
        "mol": mol,
        "mol3d": mol3d,
        "conf_ids": conf_ids,
        "heavy_atom_count": heavy_atoms,
        "dummy_atom_count": dummy_count,
        "uff_status": uff_status,
        "embed_mode": embed_mode,
    }

def pose_is_pass(m, t):
    return (
        m["severe_atom_frac"] <= t["pass_severe_frac"] and
        m["soft_atom_frac"] <= t["pass_soft_frac"] and
        m["cavity_frac"] >= PASS_CAVITY_FRAC and
        m["anchor_frac"] >= PASS_ANCHOR_FRAC and
        m["max_span_excess_A"] <= PASS_MAX_SPAN_EXCESS_A
    )

def pose_is_borderline(m):
    return (
        m["severe_atom_frac"] <= BORDERLINE_SEVERE_ATOM_FRAC and
        m["soft_atom_frac"] <= BORDERLINE_SOFT_ATOM_FRAC and
        m["cavity_frac"] >= BORDERLINE_CAVITY_FRAC and
        m["anchor_frac"] >= BORDERLINE_ANCHOR_FRAC and
        m["max_span_excess_A"] <= BORDERLINE_MAX_SPAN_EXCESS_A
    )

def classify_search(search, thresholds):
    best = search["best"]
    if best is None:
        return "TECHNICAL_FAIL", "NO_VALID_POSE_EVALUATED"
    if search["n_pass_conformers"] > 0:
        return "PASS", "AT_LEAST_ONE_CAVITY_ADMISSIBLE_ANCHORED_CONFORMER"
    if search["n_borderline_conformers"] > 0 or pose_is_borderline(best):
        return "BORDERLINE", "TIGHT_OR_PARTIAL_CAVITY_FIT"
    strong_exclusion = (
        best["cavity_frac"] <= CONFIDENT_FAIL_MAX_CAVITY_FRAC or
        best["severe_atom_frac"] >= CONFIDENT_FAIL_MIN_SEVERE_FRAC or
        best["max_span_excess_A"] >= CONFIDENT_FAIL_MIN_SPAN_EXCESS_A
    )
    if strong_exclusion and search["total_poses"] >= 500 and search["n_conformers_tested"] >= 1:
        return "CONFIDENT_FAIL", "REPEATED_PLACEMENT_SHOWS_STRONG_STERIC_OR_CAVITY_EXCLUSION"
    return "BORDERLINE", "NO_CLEAR_FIT_BUT_NOT_ENOUGH_EVIDENCE_FOR_CONFIDENT_FAIL"

def search_best_placement(mol3d, conf_ids, receptor_coords, receptor_radii,
                          ligand_coords, pocket_center, cavity_indices,
                          cavity_points, grid_origin, cavity_frame_center,
                          cavity_axes, cavity_spans, pass_thresholds, rng_seed):
    rng = np.random.default_rng(rng_seed)
    target_dist = np.linalg.norm(cavity_points-pocket_center, axis=1)
    target_points = cavity_points[target_dist <= PLACEMENT_TARGET_CENTER_RADIUS_A]
    if len(target_points)==0:
        target_points = cavity_points
    best = None; total_poses=0; n_pass=0; n_border=0; n_tested=0
    for conf_id in conf_ids:
        coords, scaffold_radii = get_heavy_atom_data(mol3d, conf_id)
        if len(coords)==0:
            continue
        n_tested += 1
        centered = coords - coords.mean(axis=0)
        conf_best = None
        first = centered + pocket_center
        metrics = evaluate_pose(first, scaffold_radii, receptor_coords, receptor_radii,
                                ligand_coords, cavity_indices, grid_origin,
                                cavity_frame_center, cavity_axes, cavity_spans)
        metrics["conf_id"] = int(conf_id); metrics["pose_index"] = 0
        total_poses += 1; conf_best = metrics.copy()
        if best is None or metrics["score"] < best["score"]:
            best = metrics.copy()
        for pose_index in range(1, POSES_PER_CONFORMER+1):
            R = random_rotation_matrix(rng)
            rotated = centered @ R.T
            target = target_points[int(rng.integers(0,len(target_points)))]
            jitter = rng.uniform(-PLACEMENT_JITTER_A, PLACEMENT_JITTER_A, size=3)
            placed = rotated + target + jitter
            metrics = evaluate_pose(placed, scaffold_radii, receptor_coords, receptor_radii,
                                    ligand_coords, cavity_indices, grid_origin,
                                    cavity_frame_center, cavity_axes, cavity_spans)
            metrics["conf_id"] = int(conf_id); metrics["pose_index"] = int(pose_index)
            total_poses += 1
            if metrics["score"] < conf_best["score"]:
                conf_best = metrics.copy()
            if best is None or metrics["score"] < best["score"]:
                best = metrics.copy()
            if pose_is_pass(metrics, pass_thresholds):
                break
        if pose_is_pass(conf_best, pass_thresholds):
            n_pass += 1
        elif pose_is_borderline(conf_best):
            n_border += 1
    return {
        "best": best,
        "total_poses": total_poses,
        "n_pass_conformers": n_pass,
        "n_borderline_conformers": n_border,
        "n_conformers_tested": n_tested,
    }

RESULT_FIELDS = [
    "chemical_scaffold_id","scaffold_smiles","status","reason",
    "heavy_atom_count","ring_count","rotatable_bond_count","dummy_atom_count",
    "attachment_site_status","n_conformers_generated","n_conformers_tested",
    "n_conformers_pass","n_conformers_borderline","total_poses_tested",
    "best_score","best_severe_atom_count","best_severe_atom_frac",
    "best_soft_atom_count","best_soft_atom_frac","best_cavity_frac",
    "best_anchor_frac","best_min_distance","best_max_span_excess_A","best_span_fit",
    "best_conf_id","best_pose_index",
    "uff_status","embed_mode","error_message"
]

def scaffold_worker(scaffold_id, smiles, receptor_coords, receptor_radii,
                    ligand_coords, pocket_center, cavity_indices, cavity_points,
                    grid_origin, cavity_frame_center, cavity_axes, cavity_spans,
                    pass_thresholds, worker_seed, result_queue):
    result = {k: None for k in RESULT_FIELDS}
    result.update({
        "chemical_scaffold_id": str(scaffold_id),
        "scaffold_smiles": smiles,
        "n_conformers_generated": 0,
        "n_conformers_tested": 0,
        "n_conformers_pass": 0,
        "n_conformers_borderline": 0,
        "total_poses_tested": 0,
    })
    try:
        conf = generate_conformers(smiles)
        mol = conf["mol"]
        result["heavy_atom_count"] = conf["heavy_atom_count"]
        result["dummy_atom_count"] = conf["dummy_atom_count"]
        result["ring_count"] = int(rdMolDescriptors.CalcNumRings(mol))
        result["rotatable_bond_count"] = int(rdMolDescriptors.CalcNumRotatableBonds(mol))
        result["uff_status"] = conf["uff_status"]
        result["embed_mode"] = conf["embed_mode"]
        result["n_conformers_generated"] = len(conf["conf_ids"])
        result["attachment_site_status"] = (
            "DUMMY_SITES_PRESENT_DETAILED_ACCESSIBILITY_PENDING"
            if conf["dummy_atom_count"] > 0 else
            "NO_EXPLICIT_ATTACHMENT_SITE_IN_SCAFFOLD_SMILES"
        )
        search = search_best_placement(
            conf["mol3d"], conf["conf_ids"], receptor_coords, receptor_radii,
            ligand_coords, pocket_center, cavity_indices, cavity_points,
            grid_origin, cavity_frame_center, cavity_axes, cavity_spans,
            pass_thresholds, worker_seed
        )
        status, reason = classify_search(search, pass_thresholds)
        result["status"] = status; result["reason"] = reason
        result["n_conformers_tested"] = search["n_conformers_tested"]
        result["n_conformers_pass"] = search["n_pass_conformers"]
        result["n_conformers_borderline"] = search["n_borderline_conformers"]
        result["total_poses_tested"] = search["total_poses"]
        best = search["best"]
        if best is not None:
            for key in ["score","severe_atom_count","severe_atom_frac","soft_atom_count",
                        "soft_atom_frac","cavity_frac","anchor_frac","min_distance",
                        "max_span_excess_A","span_fit","conf_id","pose_index"]:
                result["best_" + key] = best[key]
    except Exception as e:
        result["status"] = "TECHNICAL_FAIL"
        result["reason"] = "EXCEPTION"
        result["error_message"] = f"{type(e).__name__}: {e}"
    result_queue.put(result)

def initialize_output_db(conn):
    conn.execute("CREATE TABLE IF NOT EXISTS run_metadata (key TEXT PRIMARY KEY, value TEXT)")
    conn.execute("""
    CREATE TABLE IF NOT EXISTS scaffold_results (
        chemical_scaffold_id TEXT PRIMARY KEY,
        scaffold_smiles TEXT, status TEXT, reason TEXT,
        heavy_atom_count INTEGER, ring_count INTEGER, rotatable_bond_count INTEGER,
        dummy_atom_count INTEGER, attachment_site_status TEXT,
        n_conformers_generated INTEGER, n_conformers_tested INTEGER,
        n_conformers_pass INTEGER, n_conformers_borderline INTEGER,
        total_poses_tested INTEGER, best_score REAL,
        best_severe_atom_count INTEGER, best_severe_atom_frac REAL,
        best_soft_atom_count INTEGER, best_soft_atom_frac REAL,
        best_cavity_frac REAL, best_anchor_frac REAL, best_min_distance REAL,
        best_max_span_excess_A REAL, best_span_fit INTEGER,
        best_conf_id INTEGER, best_pose_index INTEGER,
        uff_status TEXT, embed_mode INTEGER, error_message TEXT
    )""")
    conn.commit()

def write_result_to_db(conn, result):
    cols = ",".join(RESULT_FIELDS)
    ph = ",".join("?" for _ in RESULT_FIELDS)
    conn.execute(f"INSERT OR REPLACE INTO scaffold_results ({cols}) VALUES ({ph})",
                 tuple(result.get(k) for k in RESULT_FIELDS))
    conn.commit()


def run_worker_timeout(
    ctx,
    scaffold_id,
    smiles,
    receptor_coords,
    receptor_radii,
    ligand_coords,
    pocket_center,
    cavity_indices,
    cavity_points,
    grid_origin,
    cavity_frame_center,
    cavity_axes,
    cavity_spans,
    pass_thresholds,
    worker_seed,
):
    """Run one scaffold in an isolated process with hard timeout."""

    result_queue = ctx.Queue(maxsize=1)

    p = ctx.Process(
        target=scaffold_worker,
        args=(
            scaffold_id,
            smiles,
            receptor_coords,
            receptor_radii,
            ligand_coords,
            pocket_center,
            cavity_indices,
            cavity_points,
            grid_origin,
            cavity_frame_center,
            cavity_axes,
            cavity_spans,
            pass_thresholds,
            worker_seed,
            result_queue,
        ),
    )

    p.start()
    p.join(PER_SCAFFOLD_TIMEOUT_SEC)

    if p.is_alive():
        p.terminate()
        p.join(5)

        if p.is_alive():
            p.kill()
            p.join()

        result = {k: None for k in RESULT_FIELDS}
        result.update(
            {
                "chemical_scaffold_id": str(scaffold_id),
                "scaffold_smiles": smiles,
                "status": "TECHNICAL_FAIL",
                "reason": "TIMEOUT",
                "error_message": f"Exceeded {PER_SCAFFOLD_TIMEOUT_SEC} s",
                "n_conformers_generated": 0,
                "n_conformers_tested": 0,
                "n_conformers_pass": 0,
                "n_conformers_borderline": 0,
                "total_poses_tested": 0,
            }
        )

    else:
        try:
            result = result_queue.get(timeout=2)
        except queue_module.Empty:
            result = {k: None for k in RESULT_FIELDS}
            result.update(
                {
                    "chemical_scaffold_id": str(scaffold_id),
                    "scaffold_smiles": smiles,
                    "status": "TECHNICAL_FAIL",
                    "reason": "WORKER_EXITED_WITHOUT_RESULT",
                    "error_message": f"worker_exitcode={p.exitcode}",
                    "n_conformers_generated": 0,
                    "n_conformers_tested": 0,
                    "n_conformers_pass": 0,
                    "n_conformers_borderline": 0,
                    "total_poses_tested": 0,
                }
            )

    result_queue.close()
    return result



def stable_shard_for_id(scaffold_id, num_shards):
    """Stable, reproducible partition independent of Python hash randomization."""
    digest = hashlib.blake2b(
        str(scaffold_id).encode("utf-8"),
        digest_size=8,
    ).digest()
    value = int.from_bytes(digest, byteorder="big", signed=False)
    return value % num_shards


def configure_shard_outputs(shard_id):
    """Redirect one worker process to its own resumable SQLite database."""
    global OUTPUT_DB, OUTPUT_CSV, RUN_REPORT, DB_SCHEMA_REPORT, CAVITY_GRID_PDB

    SHARD_DIR.mkdir(parents=True, exist_ok=True)

    prefix = f"shard_{shard_id:02d}"
    OUTPUT_DB = SHARD_DIR / f"{prefix}.db"
    OUTPUT_CSV = SHARD_DIR / f"{prefix}.csv"
    RUN_REPORT = SHARD_DIR / f"{prefix}_run_report.txt"
    DB_SCHEMA_REPORT = SHARD_DIR / f"{prefix}_schema.txt"
    CAVITY_GRID_PDB = SHARD_DIR / f"{prefix}_cavity_grid.pdb"


def main(shard_id=0, num_shards=1):
    start_time = time.time()
    print("="*92)
    print("GPCR POCKET ADMISSIBILITY PREFILTER - FINAL PARALLEL V1")
    print("="*92)
    print("RDKit version:", rdBase.rdkitVersion)
    for path in (SOURCE_DB, RECEPTOR_PDB, LORCASERIN_PDB):
        if not path.exists():
            raise FileNotFoundError(f"Missing input:\n{path}")
    print("[1] Inputs verified.")

    receptor_atoms_all = read_pdb_atoms(RECEPTOR_PDB, True, False, True)
    ligand_atoms = read_pdb_atoms(LORCASERIN_PDB, False, True, True)
    if not ligand_atoms:
        raise RuntimeError("No lorcaserin heavy atoms found.")
    ligand_coords = np.array([a["coord"] for a in ligand_atoms], dtype=float)
    pocket_center = ligand_coords.mean(axis=0)
    pocket_receptor_atoms = [a for a in receptor_atoms_all if np.linalg.norm(a["coord"]-pocket_center) <= POCKET_RECEPTOR_RADIUS_A]
    receptor_coords = np.array([a["coord"] for a in pocket_receptor_atoms], dtype=float)
    receptor_radii = np.array([a["radius"] for a in pocket_receptor_atoms], dtype=float)

    print("[2] Building lorcaserin-anchored connected cavity grid...")
    cavity = build_ligand_anchored_cavity_grid(pocket_receptor_atoms, ligand_atoms)
    cavity_indices = cavity["indices"]; cavity_points = cavity["points"]; grid_origin = cavity["origin"]
    cavity_frame_center, cavity_axes, cavity_spans = compute_cavity_principal_frame(cavity_points)
    write_cavity_grid_pdb(cavity_points, CAVITY_GRID_PDB)
    print("    Grid spacing               :", GRID_SPACING_A, "A")
    print("    Candidate free grid points :", cavity["n_free"])
    print("    Ligand seed grid points    :", cavity["n_seed"])
    print("    Connected cavity points    :", cavity["n_cavity"])
    print("    Cavity PCA spans (A)       :", np.round(cavity_spans, 3))
    print("    Cavity grid PDB            :", CAVITY_GRID_PDB)

    native_radii = np.array([a["radius"] for a in ligand_atoms], dtype=float)
    native = evaluate_pose(ligand_coords, native_radii, receptor_coords, receptor_radii,
                           ligand_coords, cavity_indices, grid_origin,
                           cavity_frame_center, cavity_axes, cavity_spans)
    print("[3] Native lorcaserin positive control:")
    print(f"    Severe atom fraction : {native['severe_atom_frac']:.4f}")
    print(f"    Soft atom fraction   : {native['soft_atom_frac']:.4f}")
    print(f"    Cavity fraction      : {native['cavity_frac']:.4f}")
    print(f"    Anchor fraction      : {native['anchor_frac']:.4f}")
    print(f"    Minimum distance     : {native['min_distance']:.3f} A")
    print(f"    Max span excess      : {native['max_span_excess_A']:.3f} A")
    pass_thresholds = {
        "pass_severe_frac": max(BASE_PASS_SEVERE_ATOM_FRAC, native["severe_atom_frac"] + NATIVE_SEVERE_MARGIN),
        "pass_soft_frac": max(BASE_PASS_SOFT_ATOM_FRAC, native["soft_atom_frac"] + NATIVE_SOFT_MARGIN),
    }
    print("    Effective PASS severe cutoff:", round(pass_thresholds["pass_severe_frac"],4))
    print("    Effective PASS soft cutoff  :", round(pass_thresholds["pass_soft_frac"],4))
    if native["severe_atom_frac"] > 0.20:
        raise RuntimeError("Native lorcaserin severe-clash fraction is too high. STOP.")
    if native["cavity_frac"] < 0.80:
        raise RuntimeError("Native lorcaserin is not sufficiently represented by the cavity grid. STOP.")

    ctx = mp.get_context("fork")

    source_conn = sqlite3.connect(f"file:{SOURCE_DB}?mode=ro", uri=True)
    tables = get_table_names(source_conn)
    print("[4] Database tables:")
    for t in tables: print("   ", t)
    if "chemical_scaffolds" not in tables:
        raise RuntimeError("chemical_scaffolds table not found.")
    columns = get_table_columns(source_conn, "chemical_scaffolds")

    # Fixed from the actual schema of global_scaffold_index.db.
    # topology_uid is an identifier, NOT a SMILES field.
    id_column = "chemical_uid"
    smiles_column = "chemical_smiles"

    actual_column_names = {c["name"] for c in columns}

    if id_column not in actual_column_names:
        raise RuntimeError(
            f"Expected column {id_column!r} not found in chemical_scaffolds."
        )

    if smiles_column not in actual_column_names:
        raise RuntimeError(
            f"Expected column {smiles_column!r} not found in chemical_scaffolds."
        )

    print("[5] Database schema FIXED:")
    print("    ID column     :", id_column)
    print("    SMILES column :", smiles_column)

    with open(DB_SCHEMA_REPORT, "w", encoding="utf-8") as f:
        f.write("chemical_scaffolds columns\n" + "="*70 + "\n")
        for c in columns:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
        f.write(f"\nFixed ID column: {id_column}\n")
        f.write(f"Fixed SMILES column: {smiles_column}\n")
    total_available = source_conn.execute(
        f'SELECT COUNT(*) FROM chemical_scaffolds '
        f'WHERE "{smiles_column}" IS NOT NULL AND TRIM("{smiles_column}") != ""'
    ).fetchone()[0]

    print()
    print(f"[6] FULL LIBRARY MODE: {total_available:,} chemical scaffolds available.")
    print(f"    Shard                    : {shard_id + 1}/{num_shards}")

    # Resume-safe production output. Existing results are preserved.
    out_conn = sqlite3.connect(OUTPUT_DB)
    initialize_output_db(out_conn)

    already_done = out_conn.execute(
        "SELECT COUNT(*) FROM scaffold_results"
    ).fetchone()[0]

    print(f"    Existing completed results : {already_done:,}")
    print(f"    Remaining (approx.)        : {max(total_available - already_done, 0):,}")
    metadata = {
        "source_db": str(SOURCE_DB), "receptor_pdb": str(RECEPTOR_PDB),
        "lorcaserin_pdb": str(LORCASERIN_PDB), "mode": "FULL_LIBRARY_RESUMABLE",
        "sampling_seed": MASTER_RANDOM_SEED, "max_conformers": MAX_CONFORMERS,
        "poses_per_conformer": POSES_PER_CONFORMER,
        "per_scaffold_timeout_sec": PER_SCAFFOLD_TIMEOUT_SEC,
        "grid_spacing_A": GRID_SPACING_A, "ligand_expansion_A": LIGAND_EXPANSION_A,
        "cavity_neighbor_cell_radius": CAVITY_NEIGHBOR_CELL_RADIUS,
        "cavity_pca_spans_A": cavity_spans.tolist(),
        "pass_max_span_excess_A": PASS_MAX_SPAN_EXCESS_A,
        "borderline_max_span_excess_A": BORDERLINE_MAX_SPAN_EXCESS_A,
        "cavity_grid_points": cavity["n_cavity"], "native_lorcaserin": native,
        "pass_thresholds": pass_thresholds, "rdkit_version": rdBase.rdkitVersion,
    }
    for k,v in metadata.items():
        out_conn.execute("INSERT OR REPLACE INTO run_metadata (key,value) VALUES (?,?)",
                         (k, v if isinstance(v,str) else json.dumps(v,ensure_ascii=False)))
    out_conn.commit()

    print()
    print("[7] Starting/resuming full-library timeout-protected screening...")

    status_counter = Counter()

    completed_ids = {
        row[0]
        for row in out_conn.execute(
            "SELECT chemical_scaffold_id FROM scaffold_results"
        )
    }

    cursor = source_conn.execute(
        f'SELECT "{id_column}", "{smiles_column}" FROM chemical_scaffolds '
        f'WHERE "{smiles_column}" IS NOT NULL AND TRIM("{smiles_column}") != "" '
        f'ORDER BY "{id_column}"'
    )

    processed_this_run = 0

    for scaffold_id, smiles in cursor:
        sid = str(scaffold_id)

        if stable_shard_for_id(sid, num_shards) != shard_id:
            continue

        if sid in completed_ids:
            continue

        processed_this_run += 1
        idx = already_done + processed_this_run

        print(
            f"[shard {shard_id:02d} | done {idx:7d}] START ID={scaffold_id}",
            flush=True,
        )
        result = run_worker_timeout(
            ctx=ctx,
            scaffold_id=scaffold_id,
            smiles=smiles,
            receptor_coords=receptor_coords,
            receptor_radii=receptor_radii,
            ligand_coords=ligand_coords,
            pocket_center=pocket_center,
            cavity_indices=cavity_indices,
            cavity_points=cavity_points,
            grid_origin=grid_origin,
            cavity_frame_center=cavity_frame_center,
            cavity_axes=cavity_axes,
            cavity_spans=cavity_spans,
            pass_thresholds=pass_thresholds,
            worker_seed=MASTER_RANDOM_SEED + idx,
        )
        status_counter[result["status"]] += 1
        write_result_to_db(out_conn,result)
        print(
            f"[shard {shard_id:02d} | done {idx:7d}] {result['status']:15s} ID={scaffold_id} "
            f"HA={result.get('heavy_atom_count')} confs={result.get('n_conformers_generated')} "
            f"cavity={result.get('best_cavity_frac')} anchor={result.get('best_anchor_frac')} "
            f"severe={result.get('best_severe_atom_frac')} "
            f"spanEx={result.get('best_max_span_excess_A')}", flush=True
        )

    # Export complete accumulated result table to CSV.
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(RESULT_FIELDS)
        sql = (
            "SELECT " + ",".join(RESULT_FIELDS)
            + " FROM scaffold_results ORDER BY chemical_scaffold_id"
        )
        for row in out_conn.execute(sql):
            writer.writerow(row)

    elapsed = time.time() - start_time

    accumulated_counts = dict(
        out_conn.execute(
            "SELECT status, COUNT(*) FROM scaffold_results GROUP BY status"
        ).fetchall()
    )
    accumulated_total = sum(accumulated_counts.values())

    with open(RUN_REPORT, "w", encoding="utf-8") as f:
        f.write("GPCR POCKET ADMISSIBILITY PREFILTER FINAL PARALLEL V1\n" + "="*80 + "\n\n")
        f.write(f"Source DB:\n{SOURCE_DB}\n\n")
        f.write(f"Receptor:\n{RECEPTOR_PDB}\n\n")
        f.write(f"Lorcaserin:\n{LORCASERIN_PDB}\n\n")
        f.write("Frozen high-recall geometric admissibility model.\n")
        f.write("Retention rule: PASS + BORDERLINE retained; CONFIDENT_FAIL excluded; "
                "TECHNICAL_FAIL retained for retry/review.\n\n")
        f.write(f"Total source scaffolds: {total_available}\n")
        f.write(f"Shard: {shard_id + 1}/{num_shards}\n")
        f.write(f"Processed this run: {processed_this_run}\n")
        f.write(f"Accumulated results: {accumulated_total}\n")
        f.write(f"Per-scaffold timeout: {PER_SCAFFOLD_TIMEOUT_SEC} s\n")
        f.write(f"Max conformers: {MAX_CONFORMERS}\n")
        f.write(f"Poses per conformer: {POSES_PER_CONFORMER}\n")
        f.write(f"Cavity grid points: {cavity['n_cavity']}\n")
        f.write(f"Cavity PCA spans A: {cavity_spans.tolist()}\n\n")
        f.write("Native lorcaserin control:\n")
        f.write(json.dumps(native, indent=2) + "\n\n")
        f.write("Accumulated classification counts:\n")
        for status in ["PASS","BORDERLINE","CONFIDENT_FAIL","TECHNICAL_FAIL"]:
            f.write(f"{status}: {accumulated_counts.get(status,0)}\n")
        f.write(f"\nElapsed seconds this run: {elapsed:.1f}\n")

    source_conn.close()

    print("\n" + "="*92)
    print("FINAL PARALLEL V1 SHARD COMPLETE")
    print("="*92)
    print("Accumulated classification:")
    for status in ["PASS","BORDERLINE","CONFIDENT_FAIL","TECHNICAL_FAIL"]:
        print(f"    {status:16s}: {accumulated_counts.get(status,0)}")
    print(f"Accumulated total: {accumulated_total:,} / {total_available:,}")
    print(f"Elapsed this run: {elapsed/60:.2f} min")
    print("Output:", OUTPUT_DIR)
    print("Original global_scaffold_index.db was opened READ-ONLY.")
    out_conn.close()

def merge_shards(num_shards):
    """Merge all completed shard DBs into one final DB/CSV."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    final_db = OUTPUT_DIR / "gpcr_pocket_prefilter_final_parallel_v1.db"
    final_csv = OUTPUT_DIR / "gpcr_pocket_prefilter_final_parallel_v1.csv"
    final_report = OUTPUT_DIR / "run_report.txt"

    # Rebuild only the derived merged DB; shard DBs remain the resume source.
    if final_db.exists():
        final_db.unlink()

    conn = sqlite3.connect(final_db)
    initialize_output_db(conn)

    merged_from = []
    for shard_id in range(num_shards):
        shard_db = SHARD_DIR / f"shard_{shard_id:02d}.db"
        if not shard_db.exists():
            raise FileNotFoundError(f"Missing shard DB: {shard_db}")

        alias = f"s{shard_id}"
        conn.execute(f"ATTACH DATABASE ? AS {alias}", (str(shard_db),))
        conn.execute(
            f"""
            INSERT OR REPLACE INTO scaffold_results
            SELECT * FROM {alias}.scaffold_results
            """
        )
        conn.commit()
        conn.execute(f"DETACH DATABASE {alias}")
        merged_from.append(str(shard_db))

    counts = dict(
        conn.execute(
            "SELECT status, COUNT(*) FROM scaffold_results GROUP BY status"
        ).fetchall()
    )
    total = sum(counts.values())

    with open(final_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(RESULT_FIELDS)
        sql = (
            "SELECT " + ",".join(RESULT_FIELDS)
            + " FROM scaffold_results ORDER BY chemical_scaffold_id"
        )
        for row in conn.execute(sql):
            writer.writerow(row)

    with open(final_report, "w", encoding="utf-8") as f:
        f.write("GPCR POCKET ADMISSIBILITY PREFILTER FINAL PARALLEL V1\\n")
        f.write("=" * 80 + "\\n\\n")
        f.write(f"Parallel shards: {num_shards}\\n")
        f.write(f"Merged results: {total}\\n")
        for status in ["PASS", "BORDERLINE", "CONFIDENT_FAIL", "TECHNICAL_FAIL"]:
            f.write(f"{status}: {counts.get(status, 0)}\\n")
        f.write("\\nShard databases:\\n")
        for path in merged_from:
            f.write(path + "\\n")
        f.write("\\nRetention semantics:\\n")
        f.write("PASS: retain\\n")
        f.write("BORDERLINE: retain\\n")
        f.write("CONFIDENT_FAIL: exclude as geometry-incompatible\\n")
        f.write("TECHNICAL_FAIL: retain for retry/review\\n")

    conn.close()

    print("=" * 92)
    print("MERGE COMPLETE")
    print("=" * 92)
    print(f"Merged results: {total:,}")
    for status in ["PASS", "BORDERLINE", "CONFIDENT_FAIL", "TECHNICAL_FAIL"]:
        print(f"    {status:16s}: {counts.get(status,0):,}")
    print("Final DB :", final_db)
    print("Final CSV:", final_csv)
    print("Report   :", final_report)


def master_parallel(num_workers):
    """
    Launch N independent resumable shard processes.
    Each shard preserves the validated hard per-scaffold timeout design.
    """
    if num_workers < 1:
        raise ValueError("num_workers must be >= 1")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    script = Path(__file__).resolve()

    print("=" * 92)
    print("GPCR POCKET ADMISSIBILITY PREFILTER - FINAL PARALLEL V1")
    print("=" * 92)
    print(f"Parallel workers: {num_workers}")
    print("Each worker has its own resumable shard DB.")
    print("The source global_scaffold_index.db remains READ-ONLY.")
    print()

    processes = []
    log_handles = []

    for shard_id in range(num_workers):
        log_path = LOG_DIR / f"shard_{shard_id:02d}.log"
        log_handle = open(log_path, "a", encoding="utf-8")
        log_handles.append(log_handle)

        cmd = [
            sys.executable,
            str(script),
            "--worker-shard",
            str(shard_id),
            "--num-shards",
            str(num_workers),
        ]

        p = subprocess.Popen(
            cmd,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            cwd=str(script.parent),
            env=os.environ.copy(),
        )
        processes.append((shard_id, p))
        print(
            f"Started shard {shard_id:02d} PID={p.pid} "
            f"log={log_path}"
        )

    failed = []

    try:
        for shard_id, p in processes:
            rc = p.wait()
            print(f"Shard {shard_id:02d} finished with exit code {rc}")
            if rc != 0:
                failed.append((shard_id, rc))
    except KeyboardInterrupt:
        print("\\nMaster interrupted. Terminating shard parent processes...")
        for _, p in processes:
            if p.poll() is None:
                p.terminate()
        for _, p in processes:
            try:
                p.wait(timeout=10)
            except Exception:
                if p.poll() is None:
                    p.kill()
        raise
    finally:
        for handle in log_handles:
            handle.close()

    if failed:
        print("Some shards failed:", failed)
        print(
            "Do not delete shard DBs. Rerun the same command; "
            "completed scaffold IDs will be skipped."
        )
        raise SystemExit(2)

    merge_shards(num_workers)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Parallel resumable 5-HT2C scaffold geometric prefilter"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_PARALLEL_WORKERS,
        help=f"parallel shard count (default {DEFAULT_PARALLEL_WORKERS})",
    )
    parser.add_argument("--worker-shard", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument("--merge-only", action="store_true")
    return parser.parse_args()


def cli():
    """Run the command-line pocket prefilter entry point."""

    args = parse_args()

    if args.worker_shard is not None:
        if args.num_shards is None:
            raise SystemExit("--num-shards is required with --worker-shard")
        if not (0 <= args.worker_shard < args.num_shards):
            raise SystemExit("invalid shard id")

        configure_shard_outputs(args.worker_shard)
        main(
            shard_id=args.worker_shard,
            num_shards=args.num_shards,
        )

    elif args.merge_only:
        merge_shards(args.workers)

    else:
        master_parallel(args.workers)


if __name__ == "__main__":
    cli()
