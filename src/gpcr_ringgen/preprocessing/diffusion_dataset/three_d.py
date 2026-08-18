"""Read-only SQLite to independent 3D training-asset conversion.

This module creates an intermediate asset database.  It never changes the
large source database and it deliberately does not invent atom/edge feature
semantics.  A later materialization step can turn the validated 3D assets into
the JSONL records consumed by :class:`JsonlMolecularDataset` once a feature
manifest has been frozen.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _readonly_uri(path: Path) -> str:
    return f"file:{path.resolve()}?mode=ro"


def _json(value: Any, name: str) -> Any:
    try:
        return json.loads(value) if isinstance(value, str) else value
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} is not valid JSON") from exc


def _list_rows(value: Any, name: str) -> list[Mapping[str, Any]]:
    parsed = _json(value, name)
    if isinstance(parsed, Mapping):
        parsed = list(parsed.values())
    if not isinstance(parsed, list) or any(not isinstance(row, Mapping) for row in parsed):
        raise ValueError(f"{name} must be a JSON list of objects")
    return list(parsed)


def _index(row: Mapping[str, Any], candidates: tuple[str, ...], name: str) -> int:
    for key in candidates:
        if key in row:
            value = row[key]
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name}.{key} must be an integer")
            return int(value)
    raise ValueError(f"{name} has no index field; tried {candidates}")


def _element(row: Mapping[str, Any]) -> str:
    for key in ("symbol", "element", "element_symbol", "atom_symbol"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError("atom fact has no element symbol")


def _bond_endpoints(row: Mapping[str, Any]) -> tuple[int, int]:
    source = _index(row, ("begin_atom_index", "begin", "start", "source", "u"), "bond")
    target = _index(row, ("end_atom_index", "end", "target", "v"), "bond")
    return source, target


def _bond_kind(row: Mapping[str, Any]) -> str:
    for key in ("bond_type", "type", "bond_kind"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().upper()
    order = row.get("bond_order")
    if order == 1:
        return "SINGLE"
    if order == 2:
        return "DOUBLE"
    if order == 3:
        return "TRIPLE"
    if order == 1.5:
        return "AROMATIC"
    raise ValueError("bond fact has no supported bond type")


def _ring_indices(value: Any, atom_count: int) -> list[int]:
    parsed = _json(value, "ring_atom_indices_json")
    if not isinstance(parsed, list) or any(isinstance(item, bool) or not isinstance(item, int) for item in parsed):
        raise ValueError("ring atom indices must be an integer JSON list")
    result = sorted(set(int(item) for item in parsed))
    if any(item < 0 or item >= atom_count for item in result):
        raise ValueError("ring atom index is outside the molecule")
    if not result:
        raise ValueError("ring system contains no atoms")
    return result


def _table_columns(connection: sqlite3.Connection, table: str) -> set[str]:
    rows = connection.execute(f'PRAGMA table_info("{table.replace(chr(34), chr(34) * 2)}")').fetchall()
    if not rows:
        raise ValueError(f"required table not found: {table}")
    return {str(row[1]) for row in rows}


def _rdkit_version() -> str:
    try:
        from rdkit import rdBase

        return str(rdBase.rdkitVersion)
    except ImportError:
        return "unavailable"


@dataclass(frozen=True)
class ThreeDAssetBuilderConfig:
    source_db: Path
    output_db: Path
    molecule_table: str = "selected_molecules"
    ring_table: str = "selected_ring_systems"
    seed: int = 0xF00D
    limit: int | None = None
    resume: bool = False

    def validate(self) -> None:
        if not self.source_db.is_file():
            raise FileNotFoundError(f"source database not found: {self.source_db}")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        if self.limit is not None and (
            isinstance(self.limit, bool) or not isinstance(self.limit, int) or self.limit <= 0
        ):
            raise ValueError("limit must be a positive integer or None")


def _generate_coordinates(smiles: str, seed: int) -> tuple[list[list[float]], list[int], str]:
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError as exc:
        raise RuntimeError("RDKit is required for 3D asset generation") from exc
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("SMILES could not be parsed by RDKit")
    atom_count = molecule.GetNumAtoms()
    molecule_with_h = Chem.AddHs(molecule)
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    conformer_id = AllChem.EmbedMolecule(molecule_with_h, params)
    method = "ETKDGv3"
    if conformer_id < 0:
        fallback = AllChem.ETKDGv3()
        fallback.randomSeed = int(seed)
        fallback.useRandomCoords = True
        conformer_id = AllChem.EmbedMolecule(molecule_with_h, fallback)
        method = "ETKDGv3_random_coords_fallback"
    if conformer_id < 0:
        raise RuntimeError("ETKDGv3 embedding failed, including random-coordinate fallback")
    conformer = molecule_with_h.GetConformer(conformer_id)
    coordinates: list[list[float]] = []
    for atom_index in range(atom_count):
        position = conformer.GetAtomPosition(atom_index)
        coordinates.append([float(position.x), float(position.y), float(position.z)])
    return coordinates, list(range(atom_count)), method


def _validate_graph_alignment(
    atom_facts: list[Mapping[str, Any]],
    bond_facts: list[Mapping[str, Any]],
    coordinates: list[list[float]],
) -> dict[str, Any]:
    atom_facts = sorted(atom_facts, key=lambda row: _index(row, ("atom_index", "index"), "atom"))
    if len(atom_facts) != len(coordinates):
        raise ValueError("source atom count and generated coordinate count differ")
    source_elements = [_element(row) for row in atom_facts]
    for index, row in enumerate(atom_facts):
        if _index(row, ("atom_index", "index"), "atom") != index:
            raise ValueError("source atom indices must be contiguous and zero-based")
    edges: set[tuple[int, int]] = set()
    kinds: dict[str, int] = {}
    for row in bond_facts:
        source, target = _bond_endpoints(row)
        if source < 0 or target < 0 or source >= len(atom_facts) or target >= len(atom_facts):
            raise ValueError("source bond endpoint is outside the atom list")
        if source == target:
            raise ValueError("source graph contains a self-loop")
        pair = tuple(sorted((source, target)))
        if pair in edges:
            raise ValueError("source graph contains duplicate undirected bonds")
        edges.add(pair)
        kind = _bond_kind(row)
        kinds[kind] = kinds.get(kind, 0) + 1
    coordinates_finite = all(
        len(row) == 3 and all(isinstance(value, float) for value in row)
        for row in coordinates
    )
    if not coordinates_finite:
        raise ValueError("generated coordinates are not finite [N,3] floats")
    return {
        "atom_count": len(source_elements),
        "bond_count": len(edges),
        "elements": source_elements,
        "bond_type_counts": kinds,
        "coordinates_finite": coordinates_finite,
    }


def _create_output_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA journal_mode=WAL;
        CREATE TABLE IF NOT EXISTS build_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS molecule_3d (
            source_part TEXT NOT NULL,
            molecule_uid TEXT NOT NULL,
            original_smiles TEXT NOT NULL,
            coordinates_json TEXT,
            source_to_rdkit_atom_map_json TEXT,
            atom_facts_json TEXT NOT NULL,
            bond_facts_json TEXT NOT NULL,
            qc_json TEXT,
            generation_method TEXT,
            status TEXT NOT NULL,
            error_code TEXT,
            error_text TEXT,
            PRIMARY KEY (source_part, molecule_uid)
        );
        CREATE TABLE IF NOT EXISTS ring_system_3d (
            source_part TEXT NOT NULL,
            ring_system_uid TEXT NOT NULL,
            molecule_uid TEXT NOT NULL,
            ring_atom_mask_json TEXT,
            noise_mask_json TEXT,
            status TEXT NOT NULL,
            error_code TEXT,
            error_text TEXT,
            PRIMARY KEY (source_part, ring_system_uid)
        );
        """
    )


def _iter_source_rows(
    connection: sqlite3.Connection,
    config: ThreeDAssetBuilderConfig,
) -> Iterable[tuple[str, str, str, str, str, str, str, str]]:
    molecule_columns = _table_columns(connection, config.molecule_table)
    ring_columns = _table_columns(connection, config.ring_table)
    required_molecule = {"source_part", "molecule_uid", "original_smiles", "atom_facts_json", "bond_facts_json"}
    required_ring = {"source_part", "ring_system_uid", "molecule_uid", "ring_atom_indices_json"}
    if not required_molecule.issubset(molecule_columns):
        raise ValueError(f"{config.molecule_table} is missing {sorted(required_molecule - molecule_columns)}")
    if not required_ring.issubset(ring_columns):
        raise ValueError(f"{config.ring_table} is missing {sorted(required_ring - ring_columns)}")
    query = f"""
        SELECT m.source_part, m.molecule_uid, m.original_smiles,
               m.atom_facts_json, m.bond_facts_json,
               r.ring_system_uid, r.ring_atom_indices_json,
               COALESCE(r.ring_bond_indices_json, '[]')
        FROM "{config.molecule_table.replace(chr(34), chr(34) * 2)}" AS m
        JOIN "{config.ring_table.replace(chr(34), chr(34) * 2)}" AS r
          ON r.source_part = m.source_part
         AND r.molecule_uid = m.molecule_uid
        ORDER BY m.source_part, m.molecule_uid, r.ring_system_uid
    """
    yield from connection.execute(query)


def build_3d_assets(config: ThreeDAssetBuilderConfig) -> dict[str, int | str]:
    """Generate one conformer per molecule and ring-system masks separately."""

    config.validate()
    config.output_db.parent.mkdir(parents=True, exist_ok=True)
    if config.output_db.exists() and not config.resume:
        raise FileExistsError(f"output database exists; pass resume=True: {config.output_db}")
    source = sqlite3.connect(_readonly_uri(config.source_db), uri=True)
    output = sqlite3.connect(str(config.output_db), timeout=300)
    molecule_count = 0
    ring_count = 0
    pass_count = 0
    fail_count = 0
    try:
        _create_output_schema(output)
        output.execute("INSERT OR REPLACE INTO build_metadata(key,value) VALUES (?,?)", ("source_db", str(config.source_db)))
        output.execute("INSERT OR REPLACE INTO build_metadata(key,value) VALUES (?,?)", ("seed", str(config.seed)))
        output.execute("INSERT OR REPLACE INTO build_metadata(key,value) VALUES (?,?)", ("rdkit_version", _rdkit_version()))
        output.commit()
        current_key: tuple[str, str] | None = None
        cached: dict[str, Any] | None = None
        for row in _iter_source_rows(source, config):
            source_part, molecule_uid, smiles, atom_json, bond_json, ring_uid, ring_atoms_json, _ring_bonds_json = row
            key = (str(source_part), str(molecule_uid))
            if current_key != key:
                if config.limit is not None and molecule_count >= config.limit:
                    break
                current_key = key
                molecule_count += 1
                existing = output.execute(
                    "SELECT status FROM molecule_3d WHERE source_part=? AND molecule_uid=?",
                    key,
                ).fetchone()
                if existing is not None and config.resume:
                    cached = {"status": existing[0], "skip": True}
                else:
                    try:
                        atom_facts = _list_rows(atom_json, "atom_facts_json")
                        bond_facts = _list_rows(bond_json, "bond_facts_json")
                        coordinates, mapping, method = _generate_coordinates(str(smiles), config.seed)
                        qc = _validate_graph_alignment(atom_facts, bond_facts, coordinates)
                        cached = {
                            "status": "PASS",
                            "coordinates": coordinates,
                            "mapping": mapping,
                            "method": method,
                            "atom_facts": atom_facts,
                            "bond_facts": bond_facts,
                            "qc": qc,
                        }
                        output.execute(
                            """INSERT OR REPLACE INTO molecule_3d
                            (source_part,molecule_uid,original_smiles,coordinates_json,
                             source_to_rdkit_atom_map_json,atom_facts_json,bond_facts_json,
                             qc_json,generation_method,status,error_code,error_text)
                            VALUES (?,?,?,?,?,?,?,?,?,?,NULL,NULL)""",
                            (
                                source_part,
                                molecule_uid,
                                str(smiles),
                                json.dumps(coordinates),
                                json.dumps(mapping),
                                json.dumps(atom_facts),
                                json.dumps(bond_facts),
                                json.dumps(qc, sort_keys=True),
                                method,
                                "PASS",
                            ),
                        )
                        pass_count += 1
                    except Exception as exc:
                        cached = {"status": "FAIL", "skip": False}
                        output.execute(
                            """INSERT OR REPLACE INTO molecule_3d
                            (source_part,molecule_uid,original_smiles,atom_facts_json,
                             bond_facts_json,status,error_code,error_text)
                            VALUES (?,?,?,?,?,?,?,?)""",
                            (
                                source_part,
                                molecule_uid,
                                str(smiles),
                                str(atom_json),
                                str(bond_json),
                                "FAIL",
                                type(exc).__name__,
                                str(exc),
                            ),
                        )
                        fail_count += 1
                    output.commit()
            if not cached or cached.get("status") != "PASS" or cached.get("skip"):
                continue
            try:
                ring_indices = _ring_indices(ring_atoms_json, cached["qc"]["atom_count"])
                ring_mask = [index in set(ring_indices) for index in range(cached["qc"]["atom_count"])]
                noise_mask = [not value for value in ring_mask]
                if not any(noise_mask):
                    raise ValueError("ring system has no ring-external atom for training noise")
                output.execute(
                    """INSERT OR REPLACE INTO ring_system_3d
                    (source_part,ring_system_uid,molecule_uid,ring_atom_mask_json,
                     noise_mask_json,status,error_code,error_text)
                    VALUES (?,?,?,?,?,?,NULL,NULL)""",
                    (
                        source_part,
                        str(ring_uid),
                        molecule_uid,
                        json.dumps(ring_mask),
                        json.dumps(noise_mask),
                        "PASS",
                    ),
                )
                ring_count += 1
            except Exception as exc:
                output.execute(
                    """INSERT OR REPLACE INTO ring_system_3d
                    (source_part,ring_system_uid,molecule_uid,status,error_code,error_text)
                    VALUES (?,?,?,?,?,?)""",
                    (source_part, str(ring_uid), molecule_uid, "FAIL", type(exc).__name__, str(exc)),
                )
            if ring_count % 100 == 0:
                output.commit()
        output.execute("INSERT OR REPLACE INTO build_metadata(key,value) VALUES (?,?)", ("completed_at_utc", _utc_now()))
        output.execute("INSERT OR REPLACE INTO build_metadata(key,value) VALUES (?,?)", ("molecule_count", str(molecule_count)))
        output.execute("INSERT OR REPLACE INTO build_metadata(key,value) VALUES (?,?)", ("molecule_pass_count", str(pass_count)))
        output.execute("INSERT OR REPLACE INTO build_metadata(key,value) VALUES (?,?)", ("molecule_fail_count", str(fail_count)))
        output.execute("INSERT OR REPLACE INTO build_metadata(key,value) VALUES (?,?)", ("ring_system_count", str(ring_count)))
        output.commit()
    finally:
        source.close()
        output.close()
    return {
        "molecule_count": molecule_count,
        "molecule_pass_count": pass_count,
        "molecule_fail_count": fail_count,
        "ring_system_count": ring_count,
        "output_db": str(config.output_db),
    }


__all__ = ["ThreeDAssetBuilderConfig", "build_3d_assets"]
