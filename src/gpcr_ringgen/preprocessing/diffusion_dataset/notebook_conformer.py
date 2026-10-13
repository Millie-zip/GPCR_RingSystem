"""RDKit ETKDGv3 adapter based on the reviewed SMILES-to-2D-3D notebook.

This module deliberately contains no docking and no force-field optimization.
Its scientific core is limited to the notebook's SMILES -> AddHs -> ETKDGv3
embedding path, followed by the notebook's random-coordinate fallback.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


class ConformerGenerationError(RuntimeError):
    """A molecule could not be converted into an aligned finite conformer."""


def _require_rdkit() -> tuple[Any, Any]:
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise RuntimeError(
            "RDKit is required; install the project's preprocessing dependencies."
        ) from exc
    return Chem, AllChem


def _bond_type_name(bond: Any) -> str:
    return str(bond.GetBondType()).upper()


def _validate_mapping(
    molecule: Any,
    atoms: Sequence[Mapping[str, Any]],
    bonds: Sequence[Mapping[str, Any]],
    source_to_rdkit: Sequence[int],
) -> bool:
    if len(source_to_rdkit) != len(atoms) or len(set(source_to_rdkit)) != len(atoms):
        return False
    if min(source_to_rdkit, default=0) < 0 or max(source_to_rdkit, default=-1) >= molecule.GetNumAtoms():
        return False

    for source_index, rdkit_index in enumerate(source_to_rdkit):
        fact = atoms[source_index]
        atom = molecule.GetAtomWithIdx(rdkit_index)
        atomic_number = fact.get("atomic_number")
        if atomic_number is not None and atom.GetAtomicNum() != int(atomic_number):
            return False
        symbol = fact.get("symbol")
        if symbol is not None and atom.GetSymbol() != symbol:
            return False
        formal_charge = fact.get("formal_charge")
        if formal_charge is not None and atom.GetFormalCharge() != int(formal_charge):
            return False
        aromatic = fact.get("is_aromatic")
        if aromatic is not None and atom.GetIsAromatic() != bool(aromatic):
            return False

    if molecule.GetNumBonds() != len(bonds):
        return False
    for expected_index, fact in enumerate(bonds):
        if fact.get("bond_index") != expected_index:
            return False
        begin = fact.get("begin_atom_index")
        end = fact.get("end_atom_index")
        if not isinstance(begin, int) or not isinstance(end, int):
            return False
        if not (0 <= begin < len(atoms) and 0 <= end < len(atoms)):
            return False
        bond = molecule.GetBondBetweenAtoms(
            source_to_rdkit[begin], source_to_rdkit[end]
        )
        if bond is None or _bond_type_name(bond) != str(fact.get("bond_type")).upper():
            return False
    return True


def _source_query_molecule(
    atoms: Sequence[Mapping[str, Any]],
    bonds: Sequence[Mapping[str, Any]],
) -> Any:
    Chem, _ = _require_rdkit()
    editable = Chem.RWMol()
    for expected_index, fact in enumerate(atoms):
        if fact.get("atom_index") != expected_index:
            raise ConformerGenerationError(
                "atom_facts_json is not in contiguous atom-index order"
            )
        atomic_number = fact.get("atomic_number")
        if atomic_number is None:
            symbol = fact.get("symbol")
            if not isinstance(symbol, str):
                raise ConformerGenerationError("atom fact lacks atomic identity")
            atomic_number = Chem.GetPeriodicTable().GetAtomicNumber(symbol)
        atom = Chem.Atom(int(atomic_number))
        atom.SetFormalCharge(int(fact.get("formal_charge", 0)))
        atom.SetIsAromatic(bool(fact.get("is_aromatic", False)))
        atom.SetIsotope(int(fact.get("isotope", 0)))
        editable.AddAtom(atom)

    bond_types = {
        "SINGLE": Chem.BondType.SINGLE,
        "DOUBLE": Chem.BondType.DOUBLE,
        "TRIPLE": Chem.BondType.TRIPLE,
        "AROMATIC": Chem.BondType.AROMATIC,
    }
    for expected_index, fact in enumerate(bonds):
        if fact.get("bond_index") != expected_index:
            raise ConformerGenerationError(
                "bond_facts_json is not in contiguous bond-index order"
            )
        name = str(fact.get("bond_type")).upper()
        if name not in bond_types:
            raise ConformerGenerationError(f"unsupported source bond type: {name!r}")
        begin = int(fact["begin_atom_index"])
        end = int(fact["end_atom_index"])
        editable.AddBond(begin, end, bond_types[name])
        if name == "AROMATIC":
            editable.GetBondBetweenAtoms(begin, end).SetIsAromatic(True)

    source = editable.GetMol()
    try:
        Chem.SanitizeMol(source)
    except Exception as exc:
        raise ConformerGenerationError(
            f"source atom/bond facts do not form a sanitizable molecule: {exc}"
        ) from exc
    return source


def _resolve_atom_mapping(
    molecule: Any,
    atoms: Sequence[Mapping[str, Any]],
    bonds: Sequence[Mapping[str, Any]],
) -> tuple[list[int], int, str]:
    if molecule.GetNumAtoms() != len(atoms):
        raise ConformerGenerationError(
            f"atom count mismatch: SMILES={molecule.GetNumAtoms()}, source={len(atoms)}"
        )

    identity = list(range(len(atoms)))
    if _validate_mapping(molecule, atoms, bonds, identity):
        return identity, 1, "identity"

    source = _source_query_molecule(atoms, bonds)
    matches = molecule.GetSubstructMatches(
        source,
        useChirality=False,
        uniquify=False,
        maxMatches=1000,
    )
    valid = sorted(
        tuple(int(value) for value in match)
        for match in matches
        if _validate_mapping(molecule, atoms, bonds, match)
    )
    if not valid:
        raise ConformerGenerationError(
            "no topology-preserving source-to-RDKit atom mapping was found"
        )
    return list(valid[0]), len(valid), "graph_isomorphism"


def _geometry_metrics(
    molecule: Any,
    coordinates: Sequence[Sequence[float]],
    source_to_rdkit: Sequence[int],
) -> dict[str, float | None]:
    distances: list[float] = []
    bonded_pairs: set[tuple[int, int]] = set()
    inverse = {rdkit: source for source, rdkit in enumerate(source_to_rdkit)}
    for bond in molecule.GetBonds():
        left = inverse[bond.GetBeginAtomIdx()]
        right = inverse[bond.GetEndAtomIdx()]
        bonded_pairs.add(tuple(sorted((left, right))))
        distances.append(math.dist(coordinates[left], coordinates[right]))

    nonbonded: list[float] = []
    for left in range(len(coordinates)):
        for right in range(left + 1, len(coordinates)):
            if (left, right) not in bonded_pairs:
                nonbonded.append(math.dist(coordinates[left], coordinates[right]))
    return {
        "min_bond_length_A": min(distances) if distances else None,
        "max_bond_length_A": max(distances) if distances else None,
        "min_nonbonded_distance_A": min(nonbonded) if nonbonded else None,
    }


def generate_notebook_conformer(
    smiles: str,
    atoms: Sequence[Mapping[str, Any]],
    bonds: Sequence[Mapping[str, Any]],
    *,
    random_seed: int = 0xF00D,
) -> dict[str, Any]:
    """Generate one aligned heavy-atom conformer using the reviewed notebook path."""

    Chem, AllChem = _require_rdkit()
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ConformerGenerationError("RDKit could not parse the selected SMILES")

    source_to_rdkit, mapping_count, mapping_method = _resolve_atom_mapping(
        molecule, atoms, bonds
    )
    with_hydrogens = Chem.AddHs(molecule)
    if with_hydrogens.GetNumAtoms() < molecule.GetNumAtoms():
        raise ConformerGenerationError("Chem.AddHs unexpectedly removed atoms")

    params = AllChem.ETKDGv3()
    params.randomSeed = int(random_seed)
    params.trackFailures = True
    return_code = int(AllChem.EmbedMolecule(with_hydrogens, params))
    failure_counts = (
        list(params.GetFailureCounts())
        if hasattr(params, "GetFailureCounts")
        else []
    )
    method = "ETKDGv3"
    embedding_attempts = [
        {
            "method": method,
            "return_code": return_code,
            "failure_counts": failure_counts,
        }
    ]

    if return_code != 0:
        with_hydrogens.RemoveAllConformers()
        return_code = int(
            AllChem.EmbedMolecule(
                with_hydrogens,
                maxAttempts=5000,
                randomSeed=int(random_seed),
            )
        )
        method = "distance_geometry_max_attempts_5000"
        embedding_attempts.append(
            {"method": method, "return_code": return_code}
        )
    if return_code != 0:
        with_hydrogens.RemoveAllConformers()
        return_code = int(
            AllChem.EmbedMolecule(
                with_hydrogens,
                randomSeed=int(random_seed),
                useRandomCoords=True,
            )
        )
        method = "random_coordinates_fallback"
        embedding_attempts.append(
            {"method": method, "return_code": return_code}
        )
    if return_code != 0 or with_hydrogens.GetNumConformers() != 1:
        raise ConformerGenerationError(
            f"RDKit embedding attempts all failed: {embedding_attempts}"
        )

    conformer = with_hydrogens.GetConformer()
    coordinates = [
        [
            float(conformer.GetAtomPosition(rdkit_index).x),
            float(conformer.GetAtomPosition(rdkit_index).y),
            float(conformer.GetAtomPosition(rdkit_index).z),
        ]
        for rdkit_index in source_to_rdkit
    ]
    if any(not math.isfinite(value) for row in coordinates for value in row):
        raise ConformerGenerationError("generated coordinates contain NaN or Inf")

    rdkit_to_source = [-1] * molecule.GetNumAtoms()
    for source_index, rdkit_index in enumerate(source_to_rdkit):
        rdkit_to_source[rdkit_index] = source_index
    if any(value < 0 for value in rdkit_to_source):
        raise ConformerGenerationError("atom mapping is not bijective")

    return {
        "coordinates": coordinates,
        "source_to_rdkit": source_to_rdkit,
        "rdkit_to_source": rdkit_to_source,
        "mapping_method": mapping_method,
        "mapping_match_count": mapping_count,
        "embedding_method": method,
        "embedding_return_code": return_code,
        "embedding_attempts": embedding_attempts,
        "etkdg_failure_counts": failure_counts,
        "geometry": _geometry_metrics(molecule, coordinates, source_to_rdkit),
    }


__all__ = ["ConformerGenerationError", "generate_notebook_conformer"]
