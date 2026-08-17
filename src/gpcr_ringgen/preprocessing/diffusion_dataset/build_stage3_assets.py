"""Build the 8DPF pocket JSON and an atom-aligned docked-pose database.

RDKit creates starting conformers; a user-configured external docking command
produces pocket poses. Only poses passing the existing 8DPF QC are stored.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

DEFAULT_SOURCE_DB = Path(
    "/share/home/weiliao02/lyx/zinc_project/5ht2c_3d_diffusion_library_v1/"
    "5ht2c_3d_diffusion_library_v1.db"
)
DEFAULT_RECEPTOR_PDB = Path(
    "/share/home/weiliao02/lyx/zinc_project/8DPF_V1_prepared/"
    "8DPF_receptor_V1_prepared.pdb"
)
DEFAULT_REFERENCE_PDB = Path(
    "/share/home/weiliao02/lyx/zinc_project/8DPF_V1_prepared/"
    "8DPF_lorcaserin_reference_raw.pdb"
)


def _read_only_uri(path: Path) -> str:
    return f"file:{path.resolve().as_posix()}?mode=ro"


def _load_command(path: Path) -> list[str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise TypeError("docking config root must be an object")
    command = raw.get("command")
    if (
        not isinstance(command, list)
        or not command
        or any(not isinstance(token, str) or not token for token in command)
    ):
        raise ValueError("docking config command must be a non-empty string list")
    required = {"{ligand_sdf}", "{output_sdf}", "{receptor_pdb}"}
    joined = " ".join(command)
    missing = sorted(token for token in required if token not in joined)
    if missing:
        raise ValueError(
            "docking command is missing placeholders: " + ", ".join(missing)
        )
    return command


def _format_command(
    template: Sequence[str],
    *,
    ligand_sdf: Path,
    output_sdf: Path,
    receptor_pdb: Path,
    reference_pdb: Path,
) -> list[str]:
    values = {
        "ligand_sdf": str(ligand_sdf),
        "output_sdf": str(output_sdf),
        "receptor_pdb": str(receptor_pdb),
        "reference_pdb": str(reference_pdb),
    }
    return [token.format(**values) for token in template]


def _pocket_payload(
    pocket_atoms: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "coordinates": [atom["coord"].tolist() for atom in pocket_atoms],
        "features": [[0.0, 0.0] for _ in pocket_atoms],
        "mask": [True for _ in pocket_atoms],
        "atom_types": [str(atom["atom_name"]) for atom in pocket_atoms],
        "elements": [str(atom["element"]).title() for atom in pocket_atoms],
        "residue_names": [str(atom["resname"]) for atom in pocket_atoms],
        "residue_ids": [str(atom["resseq"]) for atom in pocket_atoms],
        "chain_ids": [str(atom["chain"]) for atom in pocket_atoms],
    }


def _write_pocket_json(
    path: Path,
    pocket_atoms: Sequence[Mapping[str, Any]],
    *,
    receptor_pdb: Path,
    reference_pdb: Path,
) -> None:
    payload = {
        "pocket_id": "8DPF_lorcaserin_center_13A",
        "selection": {
            "receptor": str(receptor_pdb),
            "reference_ligand": str(reference_pdb),
            "heavy_atoms_only": True,
            "center": "mean_of_lorcaserin_heavy_atoms",
            "radius_A": 13.0,
        },
        "pocket": _pocket_payload(pocket_atoms),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _create_pose_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS build_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS docked_poses (
            source_part TEXT NOT NULL,
            molecule_uid TEXT NOT NULL,
            pose_id TEXT NOT NULL,
            coordinates_json TEXT,
            status TEXT NOT NULL,
            cavity_frac REAL,
            anchor_frac REAL,
            span_excess_A REAL,
            collision_frac REAL,
            soft_collision_frac REAL,
            min_distance_A REAL,
            docking_score REAL,
            error_text TEXT,
            PRIMARY KEY (source_part, molecule_uid)
        );
        CREATE INDEX IF NOT EXISTS idx_docked_poses_status
        ON docked_poses(status);
        """
    )


def _source_rows(
    source: sqlite3.Connection,
    *,
    limit: int | None,
) -> sqlite3.Cursor:
    query = """
        SELECT source_part, molecule_uid, original_smiles, atom_facts_json
        FROM selected_molecules
        ORDER BY source_part, molecule_uid
    """
    if limit is not None:
        return source.execute(query + " LIMIT ?", (limit,))
    return source.execute(query)


def _expected_symbols(atom_facts_json: str) -> list[str]:
    atoms = json.loads(atom_facts_json)
    if not isinstance(atoms, list) or not atoms:
        raise ValueError("atom_facts_json must be a non-empty list")
    result: list[str] = []
    for index, atom in enumerate(atoms):
        if not isinstance(atom, Mapping) or atom.get("atom_index") != index:
            raise ValueError("atom facts are not in contiguous atom-index order")
        symbol = atom.get("symbol")
        if not isinstance(symbol, str):
            raise TypeError("atom symbol must be a string")
        result.append(symbol)
    return result


def _write_starting_conformers(smiles: str, output_sdf: Path) -> None:
    from rdkit import Chem

    from gpcr_ringgen.preprocessing.pocket_prefilter.pipeline import (
        generate_conformers,
    )

    generated = generate_conformers(smiles)
    molecule = generated["mol3d"]
    writer = Chem.SDWriter(str(output_sdf))
    try:
        for conformer_id in generated["conf_ids"]:
            writer.write(molecule, confId=int(conformer_id))
    finally:
        writer.close()


def _heavy_coordinates(
    molecule: Any,
    expected_symbols: Sequence[str],
) -> list[list[float]]:
    observed = [
        atom.GetSymbol()
        for atom in molecule.GetAtoms()
        if atom.GetAtomicNum() not in (0, 1)
    ]
    if observed != list(expected_symbols):
        raise ValueError("docking output changed heavy-atom identity or order")
    conformer = molecule.GetConformer()
    coordinates: list[list[float]] = []
    for atom in molecule.GetAtoms():
        if atom.GetAtomicNum() in (0, 1):
            continue
        point = conformer.GetAtomPosition(atom.GetIdx())
        coordinates.append([float(point.x), float(point.y), float(point.z)])
    return coordinates


def _docking_score(molecule: Any) -> float | None:
    for name in ("minimizedAffinity", "CNNaffinity", "docking_score", "SCORE"):
        if molecule.HasProp(name):
            try:
                return float(molecule.GetProp(name))
            except ValueError:
                continue
    return None


def _best_passing_pose(
    output_sdf: Path,
    expected_symbols: Sequence[str],
    context: Mapping[str, Any],
) -> tuple[list[list[float]], dict[str, Any], float | None]:
    import numpy as np
    from rdkit import Chem

    from gpcr_ringgen.preprocessing.pocket_prefilter.pipeline import (
        evaluate_pose,
        get_vdw_radius,
        pose_is_pass,
    )

    best: tuple[list[list[float]], dict[str, Any], float | None] | None = None
    supplier = Chem.SDMolSupplier(str(output_sdf), removeHs=False)
    for molecule in supplier:
        if molecule is None:
            continue
        coordinates = _heavy_coordinates(molecule, expected_symbols)
        radii = np.asarray([get_vdw_radius(symbol) for symbol in expected_symbols])
        metrics = evaluate_pose(
            np.asarray(coordinates),
            radii,
            context["receptor_coords"],
            context["receptor_radii"],
            context["ligand_coords"],
            context["cavity_indices"],
            context["grid_origin"],
            context["cavity_frame_center"],
            context["cavity_axes"],
            context["cavity_spans"],
        )
        if not pose_is_pass(metrics, context["pass_thresholds"]):
            continue
        score = _docking_score(molecule)
        candidate = (coordinates, metrics, score)
        if best is None or score is not None and (best[2] is None or score < best[2]) or score is None and best[2] is None and metrics["score"] < best[1]["score"]:
            best = candidate
    if best is None:
        raise ValueError("no docked pose passed calibrated pocket QC")
    return best


def _prepare_pocket_context(
    receptor_pdb: Path,
    reference_pdb: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import numpy as np

    from gpcr_ringgen.preprocessing.pocket_prefilter.pipeline import (
        BASE_PASS_SEVERE_ATOM_FRAC,
        BASE_PASS_SOFT_ATOM_FRAC,
        NATIVE_SEVERE_MARGIN,
        NATIVE_SOFT_MARGIN,
        POCKET_RECEPTOR_RADIUS_A,
        build_ligand_anchored_cavity_grid,
        compute_cavity_principal_frame,
        evaluate_pose,
        read_pdb_atoms,
    )

    receptor_atoms = read_pdb_atoms(receptor_pdb, True, False, True)
    ligand_atoms = read_pdb_atoms(reference_pdb, False, True, True)
    if not receptor_atoms or not ligand_atoms:
        raise RuntimeError("receptor/reference PDB did not yield heavy atoms")
    ligand_coords = np.asarray([atom["coord"] for atom in ligand_atoms])
    pocket_center = ligand_coords.mean(axis=0)
    pocket_atoms = [
        atom
        for atom in receptor_atoms
        if np.linalg.norm(atom["coord"] - pocket_center) <= POCKET_RECEPTOR_RADIUS_A
    ]
    receptor_coords = np.asarray([atom["coord"] for atom in pocket_atoms])
    receptor_radii = np.asarray([atom["radius"] for atom in pocket_atoms])
    cavity = build_ligand_anchored_cavity_grid(pocket_atoms, ligand_atoms)
    frame_center, axes, spans = compute_cavity_principal_frame(cavity["points"])
    native_radii = np.asarray([atom["radius"] for atom in ligand_atoms])
    native = evaluate_pose(
        ligand_coords,
        native_radii,
        receptor_coords,
        receptor_radii,
        ligand_coords,
        cavity["indices"],
        cavity["origin"],
        frame_center,
        axes,
        spans,
    )
    thresholds = {
        "pass_severe_frac": max(
            BASE_PASS_SEVERE_ATOM_FRAC,
            native["severe_atom_frac"] + NATIVE_SEVERE_MARGIN,
        ),
        "pass_soft_frac": max(
            BASE_PASS_SOFT_ATOM_FRAC,
            native["soft_atom_frac"] + NATIVE_SOFT_MARGIN,
        ),
    }
    context = {
        "receptor_coords": receptor_coords,
        "receptor_radii": receptor_radii,
        "ligand_coords": ligand_coords,
        "cavity_indices": cavity["indices"],
        "grid_origin": cavity["origin"],
        "cavity_frame_center": frame_center,
        "cavity_axes": axes,
        "cavity_spans": spans,
        "pass_thresholds": thresholds,
    }
    return pocket_atoms, context


def build_stage3_assets(
    source_db: Path,
    receptor_pdb: Path,
    reference_pdb: Path,
    docking_config: Path,
    *,
    output_dir: Path | None = None,
    limit: int | None = None,
    resume: bool = False,
) -> tuple[Path, Path]:
    """Generate sibling pose DB/pocket JSON and return their paths."""

    for path in (source_db, receptor_pdb, reference_pdb, docking_config):
        if not path.is_file():
            raise FileNotFoundError(path)
    if limit is not None and (isinstance(limit, bool) or limit < 1):
        raise ValueError("limit must be a positive integer or None")
    destination = source_db.parent if output_dir is None else output_dir
    destination.mkdir(parents=True, exist_ok=True)
    poses_db = destination / "8dpf_docked_poses.db"
    pocket_json = destination / "8dpf_13A_pocket.json"
    if not resume and (poses_db.exists() or pocket_json.exists()):
        raise FileExistsError("output exists; use --resume or choose another directory")

    command_template = _load_command(docking_config)
    executable = command_template[0]
    if not Path(executable).is_file() and shutil.which(executable) is None:
        raise FileNotFoundError(f"docking executable not found: {executable}")
    pocket_atoms, context = _prepare_pocket_context(receptor_pdb, reference_pdb)
    if not pocket_json.exists():
        _write_pocket_json(
            pocket_json,
            pocket_atoms,
            receptor_pdb=receptor_pdb,
            reference_pdb=reference_pdb,
        )

    source = sqlite3.connect(_read_only_uri(source_db), uri=True)
    output = sqlite3.connect(poses_db)
    try:
        _create_pose_schema(output)
        output.executemany(
            "INSERT OR REPLACE INTO build_metadata VALUES (?, ?)",
            (
                ("source_db", str(source_db)),
                ("receptor_pdb", str(receptor_pdb)),
                ("reference_pdb", str(reference_pdb)),
                ("pocket_json", str(pocket_json)),
                ("coordinate_contract", "heavy atoms in atom_facts_json order"),
                ("rdkit_conformer_role", "docking initialization only"),
            ),
        )
        processed = 0
        for source_part, molecule_uid, smiles, atom_facts in _source_rows(
            source,
            limit=limit,
        ):
            if resume and output.execute(
                "SELECT 1 FROM docked_poses WHERE source_part=? AND molecule_uid=?",
                (source_part, molecule_uid),
            ).fetchone():
                continue
            status = "FAIL"
            coordinates_json = None
            metrics: dict[str, Any] = {}
            docking_score = None
            error_text = None
            try:
                symbols = _expected_symbols(atom_facts)
                with tempfile.TemporaryDirectory(prefix="gpcr_stage3_") as temp_name:
                    temp = Path(temp_name)
                    ligand_sdf = temp / "starting_conformers.sdf"
                    docked_sdf = temp / "docked.sdf"
                    _write_starting_conformers(smiles, ligand_sdf)
                    command = _format_command(
                        command_template,
                        ligand_sdf=ligand_sdf,
                        output_sdf=docked_sdf,
                        receptor_pdb=receptor_pdb,
                        reference_pdb=reference_pdb,
                    )
                    subprocess.run(command, check=True)
                    if not docked_sdf.is_file():
                        raise RuntimeError("docking command did not create output_sdf")
                    coordinates, metrics, docking_score = _best_passing_pose(
                        docked_sdf,
                        symbols,
                        context,
                    )
                status = "PASS"
                coordinates_json = json.dumps(coordinates, separators=(",", ":"))
            except Exception as exc:  # noqa: BLE001 - record per-molecule failures
                error_text = f"{type(exc).__name__}: {exc}"

            output.execute(
                """
                INSERT OR REPLACE INTO docked_poses VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                (
                    source_part,
                    molecule_uid,
                    "best_pose_v1",
                    coordinates_json,
                    status,
                    metrics.get("cavity_frac"),
                    metrics.get("anchor_frac"),
                    metrics.get("max_span_excess_A"),
                    metrics.get("severe_atom_frac"),
                    metrics.get("soft_atom_frac"),
                    metrics.get("min_distance"),
                    docking_score,
                    error_text,
                ),
            )
            processed += 1
            if processed % 100 == 0:
                output.commit()
                print(f"processed {processed:,} molecules", flush=True)
        output.commit()
        if output.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise RuntimeError("pose database quick_check failed")
    finally:
        output.close()
        source.close()
    return poses_db, pocket_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build docked x0 coordinates and the fixed 8DPF pocket JSON."
    )
    parser.add_argument("--source-db", type=Path, default=DEFAULT_SOURCE_DB)
    parser.add_argument("--receptor-pdb", type=Path, default=DEFAULT_RECEPTOR_PDB)
    parser.add_argument("--reference-pdb", type=Path, default=DEFAULT_REFERENCE_PDB)
    parser.add_argument("--docking-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser


def cli(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    poses_db, pocket_json = build_stage3_assets(
        args.source_db,
        args.receptor_pdb,
        args.reference_pdb,
        args.docking_config,
        output_dir=args.output_dir,
        limit=args.limit,
        resume=args.resume,
    )
    print(f"Pose database: {poses_db}")
    print(f"Pocket JSON: {pocket_json}")


__all__ = ["build_parser", "build_stage3_assets", "cli"]
