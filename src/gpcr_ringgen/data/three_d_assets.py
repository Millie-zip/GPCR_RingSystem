"""Read-only SQLite to an independent, QC-audited 3D asset database."""

from __future__ import annotations

import json
import math
import sqlite3
import struct
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from gpcr_ringgen.data.vocabulary import normalize_bond_type, normalize_element


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: object, name: str) -> Any:
    try:
        return json.loads(value) if isinstance(value, str) else value
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} is not valid JSON") from exc


def _facts(value: object, name: str) -> list[Mapping[str, Any]]:
    parsed = _json(value, name)
    if isinstance(parsed, Mapping):
        parsed = list(parsed.values())
    if not isinstance(parsed, list) or any(not isinstance(item, Mapping) for item in parsed):
        raise ValueError(f"{name} must be a JSON list of objects")
    return list(parsed)


def _index(row: Mapping[str, Any], names: tuple[str, ...], name: str) -> int:
    for key in names:
        if key in row:
            value = row[key]
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name}.{key} must be an integer")
            return int(value)
    raise ValueError(f"{name} has no index field")


def _element(row: Mapping[str, Any]) -> str:
    for key in ("symbol", "element", "element_symbol", "atom_symbol"):
        if key in row:
            return normalize_element(row[key])
    raise ValueError("atom fact has no element symbol")


def _bond_endpoints(row: Mapping[str, Any]) -> tuple[int, int]:
    return (
        _index(row, ("begin_atom_index", "begin", "start", "source", "u"), "bond"),
        _index(row, ("end_atom_index", "end", "target", "v"), "bond"),
    )


def _bond_kind(row: Mapping[str, Any]) -> str:
    for key in ("bond_type", "type", "bond_kind"):
        if key in row and row[key] is not None:
            return normalize_bond_type(row[key])
    value = row.get("bond_order", row.get("order"))
    if value is None:
        raise ValueError("bond fact has no type or order")
    return normalize_bond_type(str(value))


def _table_columns(connection: sqlite3.Connection, table: str) -> set[str]:
    identifier = table.replace('"', '""')
    rows = connection.execute(f'PRAGMA table_info("{identifier}")').fetchall()
    if not rows:
        raise ValueError(f"source table not found: {table}")
    return {str(row[1]) for row in rows}


def _table_names(connection: sqlite3.Connection) -> set[str]:
    rows = connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    return {str(row[0]) for row in rows}


def _decode_float32_coordinates(value: object, atom_count: int) -> list[list[float]]:
    """Decode the raw little-endian ``[N, 3]`` coordinates used by model DBs."""

    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise ValueError("x0 must be a raw byte buffer")
    expected_values = atom_count * 3
    raw = bytes(value)
    if len(raw) != expected_values * 4:
        raise ValueError(
            f"x0 byte length must be {expected_values * 4} for {atom_count} atoms"
        )
    values = struct.unpack(f"<{expected_values}f", raw)
    coordinates = [
        [float(values[offset]), float(values[offset + 1]), float(values[offset + 2])]
        for offset in range(0, expected_values, 3)
    ]
    if not all(math.isfinite(item) for row in coordinates for item in row):
        raise ValueError("x0 contains NaN or Inf")
    return coordinates


def _decode_uint8_mask(value: object, atom_count: int, name: str) -> list[bool]:
    """Decode a raw uint8 mask and reject malformed values early."""

    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise ValueError(f"{name} must be a raw byte buffer")
    raw = bytes(value)
    if len(raw) != atom_count:
        raise ValueError(f"{name} byte length must equal num_atoms ({atom_count})")
    if any(item not in (0, 1) for item in raw):
        raise ValueError(f"{name} must contain only uint8 values 0 or 1")
    return [bool(item) for item in raw]


def _ring_indices(value: object, atom_count: int) -> list[int]:
    parsed = _json(value, "ring_atom_indices_json")
    if not isinstance(parsed, list) or any(isinstance(item, bool) or not isinstance(item, int) for item in parsed):
        raise ValueError("ring_atom_indices_json must be an integer list")
    indices = sorted(set(parsed))
    if not indices or any(index < 0 or index >= atom_count for index in indices):
        raise ValueError("ring atom index is outside the molecule")
    return indices


def _validate_graph(atom_facts: list[Mapping[str, Any]], bond_facts: list[Mapping[str, Any]], coordinates: list[list[float]]) -> dict[str, Any]:
    ordered = sorted(atom_facts, key=lambda row: _index(row, ("atom_index", "index"), "atom"))
    if len(ordered) != len(coordinates):
        raise ValueError("source atom count and generated coordinate count differ")
    elements = []
    for expected, row in enumerate(ordered):
        if _index(row, ("atom_index", "index"), "atom") != expected:
            raise ValueError("source atom indices must be contiguous and zero-based")
        elements.append(_element(row))
    edges: set[tuple[int, int]] = set()
    bond_types: list[str] = []
    for row in bond_facts:
        source, target = _bond_endpoints(row)
        if source < 0 or target < 0 or source >= len(elements) or target >= len(elements):
            raise ValueError("bond endpoint is outside atom list")
        if source == target:
            raise ValueError("source graph contains a self-loop")
        edge = tuple(sorted((source, target)))
        if edge in edges:
            raise ValueError("source graph contains duplicate undirected bonds")
        edges.add(edge)
        bond_types.append(_bond_kind(row))
    if not all(len(row) == 3 and all(math.isfinite(float(value)) for value in row) for row in coordinates):
        raise ValueError("generated coordinates are not finite [N,3] values")
    return {"atom_count": len(elements), "bond_count": len(edges), "elements": elements, "bond_types": bond_types, "coordinates_finite": True}


def _generate_coordinates(smiles: str, seed: int) -> tuple[list[list[float]], list[int], str]:
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError as exc:
        raise RuntimeError("RDKit is required; use the GPCR environment") from exc
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("SMILES could not be parsed by RDKit")
    heavy_count = molecule.GetNumAtoms()
    with_hydrogen = Chem.AddHs(molecule)
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    conformer_id = AllChem.EmbedMolecule(with_hydrogen, params)
    method = "ETKDGv3"
    if conformer_id < 0:
        fallback = AllChem.ETKDGv3()
        fallback.randomSeed = int(seed)
        fallback.useRandomCoords = True
        conformer_id = AllChem.EmbedMolecule(with_hydrogen, fallback)
        method = "ETKDGv3_random_coords_fallback"
    if conformer_id < 0:
        raise RuntimeError("ETKDGv3 embedding failed, including fallback")
    conformer = with_hydrogen.GetConformer(conformer_id)
    coordinates = []
    for index in range(heavy_count):
        position = conformer.GetAtomPosition(index)
        coordinates.append([float(position.x), float(position.y), float(position.z)])
    return coordinates, list(range(heavy_count)), method


@dataclass(frozen=True)
class ThreeDAssetBuilderConfig:
    source_db: Path
    output_db: Path
    molecule_table: str = "selected_molecules"
    ring_table: str = "selected_ring_systems"
    seed: int = 0xF00D
    limit: int | None = None
    resume: bool = False
    progress_path: Path | None = None

    def validate(self) -> None:
        if not self.source_db.is_file():
            raise FileNotFoundError(f"source database not found: {self.source_db}")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        if self.limit is not None and (not isinstance(self.limit, int) or self.limit <= 0):
            raise ValueError("limit must be a positive integer or None")


def _create_schema(output: sqlite3.Connection) -> None:
    output.executescript(
        """
        CREATE TABLE IF NOT EXISTS build_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS molecule_3d (
            source_part TEXT NOT NULL, molecule_uid TEXT NOT NULL, original_smiles TEXT NOT NULL,
            sample_id TEXT,
            coordinates_json TEXT, source_to_rdkit_atom_map_json TEXT,
            atom_facts_json TEXT NOT NULL, bond_facts_json TEXT NOT NULL,
            qc_json TEXT, generation_method TEXT, status TEXT NOT NULL,
            error_code TEXT, error_text TEXT, PRIMARY KEY(source_part,molecule_uid)
        );
        CREATE TABLE IF NOT EXISTS ring_system_3d (
            source_part TEXT NOT NULL, ring_system_uid TEXT NOT NULL, molecule_uid TEXT NOT NULL,
            ring_atom_mask_json TEXT, noise_mask_json TEXT, ring_bond_indices_json TEXT,
            status TEXT NOT NULL, error_code TEXT, error_text TEXT,
            PRIMARY KEY(source_part,ring_system_uid)
        );
        """
    )
    if "sample_id" not in _table_columns(output, "molecule_3d"):
        output.execute("ALTER TABLE molecule_3d ADD COLUMN sample_id TEXT")


def _source_rows(source: sqlite3.Connection, config: ThreeDAssetBuilderConfig) -> Iterable[tuple[Any, ...]]:
    """Yield a normalized row for both source-library and model-input DBs.

    The model-input DB format stores one already-QC'd molecule per row in
    ``molecules`` and keeps coordinates/masks as raw blobs.  It has no separate
    ring-system table, so that schema is adapted here rather than forcing a
    lossy intermediate database rewrite.
    """

    table_names = _table_names(source)
    if config.molecule_table == "selected_molecules" and "molecules" in table_names:
        molecule_columns = _table_columns(source, "molecules")
        smiles_column = (
            "canonical_isomeric_smiles"
            if "canonical_isomeric_smiles" in molecule_columns
            else "original_smiles"
        )
        atom_count_column = (
            "num_atoms" if "num_atoms" in molecule_columns else "atom_count"
        )
        required = {
            "source_part",
            "molecule_uid",
            smiles_column,
            atom_count_column,
            "x0",
            "ring_atom_mask",
            "noise_mask",
            "atom_facts_json",
            "bond_facts_json",
        }
        if not required.issubset(molecule_columns):
            raise ValueError(
                "molecules is missing " + repr(sorted(required - molecule_columns))
            )

        # The production model-input DB stores ring systems in a normalized
        # table.  Keep one output row per (molecule, ring system), while
        # reading coordinates and chemical facts only once per molecule in
        # the builder's cache.
        if "ring_systems" in table_names:
            ring_columns = _table_columns(source, "ring_systems")
            required_ring = {
                "source_part",
                "molecule_uid",
                "ring_system_uid",
                "ring_atom_indices_json",
            }
            if not required_ring.issubset(ring_columns):
                raise ValueError(
                    "ring_systems is missing "
                    + repr(sorted(required_ring - ring_columns))
                )
            ring_bonds = (
                'r."ring_bond_indices_json"'
                if "ring_bond_indices_json" in ring_columns
                else "'[]'"
            )
            sample_order = (
                'm."sample_index"'
                if "sample_index" in molecule_columns
                else 'm.rowid'
            )
            sample_id = 'm."sample_id"' if "sample_id" in molecule_columns else "NULL"
            molecule = "molecules"
            query = f"""
                SELECT m."source_part",m."molecule_uid",m."{smiles_column}",
                       m."{atom_count_column}",m."x0",m."ring_atom_mask",m."noise_mask",
                       m."atom_facts_json",m."bond_facts_json",{sample_id},
                       r."ring_system_uid",r."ring_atom_indices_json",{ring_bonds}
                FROM "{molecule}" m
                JOIN "ring_systems" r
                  ON r."source_part"=m."source_part"
                 AND r."molecule_uid"=m."molecule_uid"
                ORDER BY {sample_order},r."system_index",r."ring_system_uid"
            """
            for row in source.execute(query):
                (
                    source_part,
                    molecule_uid,
                    smiles,
                    atom_count,
                    x0,
                    ring_mask,
                    noise_mask,
                    atom_json,
                    bond_json,
                    sample_id_value,
                    ring_uid,
                    ring_atoms_json,
                    ring_bonds_json,
                ) = row
                atom_count = int(atom_count)
                _decode_uint8_mask(ring_mask, atom_count, "ring_atom_mask")
                _decode_uint8_mask(noise_mask, atom_count, "noise_mask")
                _ring_indices(ring_atoms_json, atom_count)
                yield (
                    str(source_part),
                    str(molecule_uid),
                    str(smiles),
                    atom_json,
                    bond_json,
                    str(ring_uid),
                    None if sample_id_value is None else str(sample_id_value),
                    ring_atoms_json,
                    ring_bonds_json,
                    _decode_float32_coordinates(x0, atom_count),
                    _decode_uint8_mask(noise_mask, atom_count, "noise_mask"),
                )
            return

        # Backward-compatible adapter for the compact one-table fixture used
        # by the original Stage 3 interface tests.
        required.update({"ring_system_uid", "sample_id"})
        if not required.issubset(molecule_columns):
            raise ValueError(
                "molecules is missing " + repr(sorted(required - molecule_columns))
            )
        sample_order = (
            '"test_id"' if "test_id" in molecule_columns else '"sample_id"'
        )
        query = f"""
            SELECT "source_part","molecule_uid","{smiles_column}",
                   "{atom_count_column}","x0","ring_atom_mask","noise_mask",
                   "atom_facts_json","bond_facts_json","ring_system_uid","sample_id"
            FROM "molecules"
            ORDER BY {sample_order}
        """
        for row in source.execute(query):
            (
                source_part,
                molecule_uid,
                smiles,
                atom_count,
                x0,
                ring_mask,
                noise_mask,
                atom_json,
                bond_json,
                ring_uid,
                sample_id_value,
            ) = row
            atom_count = int(atom_count)
            indices = [
                index
                for index, is_ring in enumerate(
                    _decode_uint8_mask(ring_mask, atom_count, "ring_atom_mask")
                )
                if is_ring
            ]
            yield (
                str(source_part),
                str(molecule_uid),
                str(smiles),
                atom_json,
                bond_json,
                str(ring_uid),
                None if sample_id_value is None else str(sample_id_value),
                json.dumps(indices),
                "[]",
                _decode_float32_coordinates(x0, atom_count),
                _decode_uint8_mask(noise_mask, atom_count, "noise_mask"),
            )
        return

    molecule_columns = _table_columns(source, config.molecule_table)
    ring_columns = _table_columns(source, config.ring_table)
    required_molecule = {"source_part", "molecule_uid", "original_smiles", "atom_facts_json", "bond_facts_json"}
    required_ring = {"source_part", "ring_system_uid", "molecule_uid", "ring_atom_indices_json"}
    if not required_molecule.issubset(molecule_columns):
        raise ValueError(f"{config.molecule_table} missing {sorted(required_molecule - molecule_columns)}")
    if not required_ring.issubset(ring_columns):
        raise ValueError(f"{config.ring_table} missing {sorted(required_ring - ring_columns)}")
    ring_bonds = "r.ring_bond_indices_json" if "ring_bond_indices_json" in ring_columns else "'[]'"
    molecule = config.molecule_table.replace('"', '""')
    ring = config.ring_table.replace('"', '""')
    query = f"""
        SELECT m.source_part,m.molecule_uid,m.original_smiles,m.atom_facts_json,m.bond_facts_json,
               r.ring_system_uid,r.ring_atom_indices_json,{ring_bonds}
        FROM "{molecule}" m JOIN "{ring}" r
          ON r.source_part=m.source_part AND r.molecule_uid=m.molecule_uid
        ORDER BY m.source_part,m.molecule_uid,r.ring_system_uid
    """
    for row in source.execute(query):
        yield (row[0], row[1], row[2], row[3], row[4], row[5], None, row[6], row[7], None, None)


def build_3d_assets(config: ThreeDAssetBuilderConfig) -> dict[str, Any]:
    """Build one deterministic heavy-atom conformer per source molecule."""
    config.validate()
    config.output_db.parent.mkdir(parents=True, exist_ok=True)
    if config.output_db.exists() and not config.resume:
        raise FileExistsError(f"output database exists; use resume=True: {config.output_db}")
    source = sqlite3.connect(f"file:{config.source_db.resolve()}?mode=ro", uri=True)
    output = sqlite3.connect(str(config.output_db), timeout=300)
    progress = None if config.progress_path is None else config.progress_path.open("a", encoding="utf-8")
    molecule_count = molecule_pass = molecule_fail = ring_count = 0
    current_key: tuple[str, str] | None = None
    cached: dict[str, Any] | None = None
    try:
        _create_schema(output)
        output.execute("INSERT OR REPLACE INTO build_metadata VALUES (?,?)", ("source_db", str(config.source_db)))
        output.execute("INSERT OR REPLACE INTO build_metadata VALUES (?,?)", ("seed", str(config.seed)))
        output.execute("INSERT OR REPLACE INTO build_metadata VALUES (?,?)", ("started_at_utc", _utc_now()))
        output.commit()
        for row in _source_rows(source, config):
            (
                source_part,
                molecule_uid,
                smiles,
                atom_json,
                bond_json,
                ring_uid,
                sample_id,
                ring_atoms_json,
                ring_bonds_json,
                source_coordinates,
                _source_noise_mask,
            ) = row
            key = (str(source_part), str(molecule_uid))
            if key != current_key:
                if config.limit is not None and molecule_count >= config.limit:
                    break
                current_key = key
                molecule_count += 1
                existing = output.execute(
                    "SELECT status,coordinates_json,source_to_rdkit_atom_map_json,atom_facts_json,bond_facts_json,qc_json,generation_method FROM molecule_3d WHERE source_part=? AND molecule_uid=?", key
                ).fetchone() if config.resume else None
                if existing is not None and existing[0] == "PASS":
                    cached = {"status": "PASS", "coordinates": json.loads(existing[1]), "mapping": json.loads(existing[2]), "atom_facts": _facts(existing[3], "atom_facts_json"), "bond_facts": _facts(existing[4], "bond_facts_json"), "qc": json.loads(existing[5]), "method": existing[6]}
                else:
                    try:
                        atom_facts = _facts(atom_json, "atom_facts_json")
                        bond_facts = _facts(bond_json, "bond_facts_json")
                        if source_coordinates is None:
                            coordinates, mapping, method = _generate_coordinates(
                                str(smiles), config.seed
                            )
                        else:
                            coordinates = source_coordinates
                            mapping = list(range(len(coordinates)))
                            method = "source_x0"
                        ordered_atoms = sorted(atom_facts, key=lambda item: _index(item, ("atom_index", "index"), "atom"))
                        if len(ordered_atoms) != len(coordinates):
                            raise ValueError("source and RDKit atom counts differ")
                        if source_coordinates is None:
                            from rdkit import Chem

                            rdkit_molecule = Chem.MolFromSmiles(str(smiles))
                            if rdkit_molecule is None:
                                raise ValueError("SMILES could not be parsed by RDKit")
                            if [_element(row) for row in ordered_atoms] != [
                                normalize_element(atom.GetSymbol())
                                for atom in rdkit_molecule.GetAtoms()
                            ]:
                                raise ValueError(
                                    "source atom order cannot be mapped to RDKit order"
                                )
                        qc = _validate_graph(atom_facts, bond_facts, coordinates)
                        cached = {"status": "PASS", "coordinates": coordinates, "mapping": mapping, "atom_facts": atom_facts, "bond_facts": bond_facts, "qc": qc, "method": method}
                        output.execute(
                            """
                            INSERT OR REPLACE INTO molecule_3d
                            (source_part,molecule_uid,original_smiles,sample_id,
                             coordinates_json,source_to_rdkit_atom_map_json,
                             atom_facts_json,bond_facts_json,qc_json,generation_method,
                             status,error_code,error_text)
                            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                            """,
                            (
                                source_part,
                                molecule_uid,
                                str(smiles),
                                sample_id,
                                json.dumps(coordinates),
                                json.dumps(mapping),
                                json.dumps(atom_facts),
                                json.dumps(bond_facts),
                                json.dumps(qc, sort_keys=True),
                                method,
                                "PASS",
                                None,
                                None,
                            ),
                        )
                        molecule_pass += 1
                    except Exception as exc:
                        cached = {"status": "FAIL"}
                        output.execute("INSERT OR REPLACE INTO molecule_3d(source_part,molecule_uid,original_smiles,atom_facts_json,bond_facts_json,status,error_code,error_text) VALUES (?,?,?,?,?,?,?,?)", (source_part,molecule_uid,str(smiles),str(atom_json),str(bond_json),"FAIL",type(exc).__name__,str(exc)))
                        molecule_fail += 1
                    output.commit()
                if progress is not None:
                    progress.write(json.dumps({"event":"molecule","source_part":source_part,"molecule_uid":molecule_uid,"status":cached["status"]}, ensure_ascii=False) + "\n")
                    progress.flush()
            if not cached or cached.get("status") != "PASS":
                continue
            try:
                indices = _ring_indices(ring_atoms_json, cached["qc"]["atom_count"])
                ring_set = set(indices)
                ring_mask = [index in ring_set for index in range(cached["qc"]["atom_count"])]
                # Validate the upstream mask above, but persist the project
                # contract's ring-external audit mask.  The raw model-input
                # format uses ``noise_mask`` for an all-atom diffusion mask.
                noise_mask = [not value for value in ring_mask]
                if not any(noise_mask):
                    raise ValueError("ring system has no ring-external atom")
                output.execute("INSERT OR REPLACE INTO ring_system_3d VALUES (?,?,?,?,?,?,?,?,?)", (source_part,ring_uid,molecule_uid,json.dumps(ring_mask),json.dumps(noise_mask),str(ring_bonds_json or "[]"),"PASS",None,None))
                ring_count += 1
            except Exception as exc:
                output.execute("INSERT OR REPLACE INTO ring_system_3d VALUES (?,?,?,?,?,?,?,?,?)", (source_part,ring_uid,molecule_uid,None,None,str(ring_bonds_json or "[]"),"FAIL",type(exc).__name__,str(exc)))
            if ring_count % 100 == 0:
                output.commit()
        for key, value in (("completed_at_utc",_utc_now()),("molecule_count",str(molecule_count)),("molecule_pass_count",str(molecule_pass)),("molecule_fail_count",str(molecule_fail)),("ring_system_count",str(ring_count))):
            output.execute("INSERT OR REPLACE INTO build_metadata VALUES (?,?)", (key,value))
        output.commit()
    finally:
        if progress is not None:
            progress.close()
        source.close()
        output.close()
    return {"molecule_count":molecule_count,"molecule_pass_count":molecule_pass,"molecule_fail_count":molecule_fail,"ring_system_count":ring_count,"output_db":str(config.output_db)}


__all__ = ["ThreeDAssetBuilderConfig", "build_3d_assets"]
