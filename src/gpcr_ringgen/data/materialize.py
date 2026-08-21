"""Materialize audited 3D assets into the JSONL training contract."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

from gpcr_ringgen.data.manifest import FeatureManifest
from gpcr_ringgen.data.records import TrainingRecord
from gpcr_ringgen.data.vocabulary import ATOM_TYPE_TO_ID, BOND_TYPE_TO_ID, normalize_bond_type, normalize_element


def _facts(value: str) -> list[dict[str, Any]]:
    parsed = json.loads(value)
    if isinstance(parsed, dict):
        parsed = list(parsed.values())
    if not isinstance(parsed, list) or any(not isinstance(item, dict) for item in parsed):
        raise ValueError("asset facts must be a JSON list of objects")
    return parsed


def _index(row: dict[str, Any], names: tuple[str, ...]) -> int:
    for name in names:
        if name in row:
            return int(row[name])
    raise ValueError("asset row has no index")


def _bond_type(row: dict[str, Any]) -> str:
    for name in ("bond_type", "type", "bond_kind"):
        if name in row and row[name] is not None:
            return normalize_bond_type(row[name])
    return normalize_bond_type(str(row.get("bond_order", row.get("order"))))


def _manifest_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def materialize_training_jsonl(
    asset_db: str | Path,
    output_jsonl: str | Path,
    manifest_path: str | Path,
    *,
    limit: int | None = None,
    overwrite: bool = False,
    summary_path: str | Path | None = None,
) -> dict[str, Any]:
    """Write one ``(molecule, ring_system)`` record per PASS asset."""

    asset_path = Path(asset_db)
    output_path = Path(output_jsonl)
    manifest_file = Path(manifest_path)
    manifest = FeatureManifest.from_file(manifest_file)
    if not asset_path.is_file():
        raise FileNotFoundError(f"asset DB not found: {asset_path}")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"output JSONL exists; pass overwrite=True: {output_path}")
    if limit is not None and (not isinstance(limit, int) or limit <= 0):
        raise ValueError("limit must be a positive integer or None")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_suffix(output_path.suffix + ".partial")
    if partial.exists():
        partial.unlink()
    connection = sqlite3.connect(str(asset_path))
    written = failed = 0
    manifest_hash = _manifest_hash(manifest_file)
    failure_path = output_path.with_suffix(output_path.suffix + ".failures.jsonl")
    try:
        asset_columns = {
            str(row[1])
            for row in connection.execute('PRAGMA table_info("molecule_3d")')
        }
        sample_id_column = "m.sample_id" if "sample_id" in asset_columns else "NULL"
        query = """
            SELECT m.source_part,m.molecule_uid,m.coordinates_json,m.atom_facts_json,m.bond_facts_json,
                   m.generation_method,{sample_id_column},r.ring_system_uid,
                   r.ring_atom_mask_json,r.noise_mask_json,r.ring_bond_indices_json
            FROM molecule_3d m JOIN ring_system_3d r
              ON r.source_part=m.source_part AND r.molecule_uid=m.molecule_uid
            WHERE m.status='PASS' AND r.status='PASS'
            ORDER BY m.source_part,m.molecule_uid,r.ring_system_uid
        """.format(sample_id_column=sample_id_column)
        with partial.open("w", encoding="utf-8") as output, failure_path.open("w", encoding="utf-8") as failures:
            for row in connection.execute(query):
                if limit is not None and written >= limit:
                    break
                try:
                    (
                        source_part,
                        molecule_uid,
                        coordinates_json,
                        atom_json,
                        bond_json,
                        method,
                        source_sample_id,
                        ring_uid,
                        ring_mask_json,
                        noise_mask_json,
                        ring_bond_indices_json,
                    ) = row
                    atom_facts = sorted(_facts(atom_json), key=lambda item: _index(item, ("atom_index", "index")))
                    bond_facts = _facts(bond_json)
                    coordinates = json.loads(coordinates_json)
                    ring_mask = json.loads(ring_mask_json)
                    noise_mask = json.loads(noise_mask_json)
                    ring_bond_indices = set(json.loads(ring_bond_indices_json or "[]"))
                    edge_pairs: list[tuple[int, int]] = []
                    edge_features: list[list[float]] = []
                    bond_types: list[int] = []
                    for bond_index, bond in enumerate(bond_facts):
                        source = _index(bond, ("begin_atom_index", "begin", "start", "source", "u"))
                        target = _index(bond, ("end_atom_index", "end", "target", "v"))
                        pair = tuple(sorted((source, target)))
                        if pair in edge_pairs:
                            raise ValueError("duplicate undirected source bond")
                        kind = _bond_type(bond)
                        if kind not in BOND_TYPE_TO_ID:
                            raise ValueError(f"bond type outside vocabulary: {kind}")
                        edge_pairs.append(pair)
                        edge_features.append(manifest.edge_vector(bond, ring_bond=(bond_index in ring_bond_indices or (ring_mask[source] and ring_mask[target]))))
                        bond_types.append(BOND_TYPE_TO_ID[kind])
                    atom_types = []
                    atom_features = []
                    for index, atom in enumerate(atom_facts):
                        element = normalize_element(atom.get("symbol", atom.get("element", atom.get("element_symbol"))))
                        if element not in ATOM_TYPE_TO_ID:
                            raise ValueError(f"element outside vocabulary: {element}")
                        atom_types.append(ATOM_TYPE_TO_ID[element])
                        atom_features.append(manifest.atom_vector(atom, ring_atom=bool(ring_mask[index])))
                    edge_index = [[pair[0] for pair in edge_pairs], [pair[1] for pair in edge_pairs]]
                    record = TrainingRecord(
                        sample_id=(
                            str(source_sample_id)
                            if isinstance(source_sample_id, str) and source_sample_id.strip()
                            else f"{source_part}:{molecule_uid}:{ring_uid}"
                        ),
                        atom_features=atom_features,
                        edge_index=edge_index,
                        edge_features=edge_features,
                        coordinates=coordinates,
                        atom_types=atom_types,
                        bond_types=bond_types,
                        ring_atom_mask=[bool(item) for item in ring_mask],
                        noise_mask=[bool(item) for item in noise_mask],
                        metadata={
                            "source_part": str(source_part),
                            "molecule_uid": str(molecule_uid),
                            "ring_system_uid": str(ring_uid),
                            "split_group": f"{source_part}:{molecule_uid}",
                            "generation_method": method,
                            "feature_manifest": manifest.schema_version,
                            "feature_manifest_sha256": _manifest_hash(manifest_file),
                            "training_constraints": "unconstrained_coordinate_diffusion",
                        },
                    )
                    output.write(json.dumps(record.to_mapping(), ensure_ascii=False, separators=(",", ":")) + "\n")
                    written += 1
                except Exception as exc:
                    failed += 1
                    failures.write(json.dumps({"row": written + failed, "error_type": type(exc).__name__, "error": str(exc)}, ensure_ascii=False) + "\n")
        partial.replace(output_path)
    finally:
        connection.close()
        if partial.exists():
            partial.unlink()
    summary = {"written": written, "failed": failed, "output_jsonl": str(output_path), "failure_report": str(failure_path), "manifest": str(manifest_file), "manifest_sha256": manifest_hash}
    if summary_path is not None:
        target = Path(summary_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


__all__ = ["materialize_training_jsonl"]
