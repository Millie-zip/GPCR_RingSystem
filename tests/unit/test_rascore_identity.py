"""Identity evidence, label consistency and non-destructive CSV publication."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from gpcr_ringgen.preprocessing import rascore_identity
from gpcr_ringgen.preprocessing.rascore_identity import audit_pilot_training_csv, classify_identity


KEY = "AAAAAAAAAAAAAA-BBBBBBBBBB-C"
OTHER_STEREO_KEY = "AAAAAAAAAAAAAA-DDDDDDDDDD-C"


def _row(**updates: str) -> dict[str, str]:
    return {
        "molecule_uid": "test_molecule",
        "canonical_isomeric_smiles": "C/C=C/C",
        "canonical_nonisomeric_smiles": "CC=CC",
        "teacher_target": "C/C=C/C",
        "input_full_inchikey": KEY,
        "teacher_result_full_inchikey": KEY,
        "teacher_stereo_identity_match": "True",
        "solved": "False",
        "teacher_is_solved": "False",
        **updates,
    }


@pytest.mark.parametrize("updates,status,evidence", [
    ({}, "match", "full_inchikey_match"),
    ({"teacher_target": "different canonical representation"}, "match", "full_inchikey_match"),
    ({"teacher_result_full_inchikey": OTHER_STEREO_KEY}, "mismatch", "conflicting_identity_evidence"),
    ({"teacher_stereo_identity_match": "False"}, "mismatch", "conflicting_identity_evidence"),
    ({"input_full_inchikey": "", "teacher_result_full_inchikey": "", "teacher_stereo_identity_match": ""}, "match", "exact_isomeric_smiles_match"),
    ({"teacher_result_full_inchikey": "", "teacher_target": "C/C=C\\C"}, "pending", "insufficient_identity_evidence"),
    ({"input_full_inchikey": "", "teacher_result_full_inchikey": "", "teacher_target": "C/C=C\\C"}, "pending", "insufficient_identity_evidence"),
    ({"input_full_inchikey": "invalid key"}, "pending", "invalid_full_inchikey"),
    ({"teacher_stereo_identity_match": "unknown"}, "pending", "invalid_identity_flag"),
    ({"teacher_target": ""}, "pending", "missing_identity_fields"),
    ({"molecule_uid": ""}, "pending", "missing_identity_fields"),
])
def test_identity_requires_positive_evidence_and_preserves_stereo_conflicts(updates, status, evidence) -> None:
    original = _row(**updates)
    before = original.copy()
    result = classify_identity(original)
    assert result.status == status
    assert result.evidence == evidence
    assert original == before


def _write(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(_row()))
        writer.writeheader()
        writer.writerows(rows)


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_audit_separates_identity_and_label_failures_without_modifying_source(tmp_path: Path) -> None:
    source = tmp_path / "teacher.csv"
    rows = [
        _row(molecule_uid="keys"),
        _row(molecule_uid="strings", input_full_inchikey="", teacher_result_full_inchikey="", teacher_stereo_identity_match=""),
        _row(molecule_uid="mismatch", teacher_result_full_inchikey=OTHER_STEREO_KEY),
        _row(molecule_uid="pending", input_full_inchikey="", teacher_result_full_inchikey="", teacher_target="C/C=C\\C"),
        _row(molecule_uid="label_conflict", solved="True"),
        _row(molecule_uid="invalid_label", solved="not a boolean"),
    ]
    _write(source, rows)
    original = source.read_bytes()
    historical = tmp_path / "historical_model.joblib"
    historical.write_bytes(b"test historical artifact; never unpickle")
    output = tmp_path / "release"

    report_path = audit_pilot_training_csv(source, output, historical_artifacts=[historical])

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert source.read_bytes() == original
    assert historical.read_bytes() == b"test historical artifact; never unpickle"
    assert report["counts"] == {"source_rows": 6, "eligible_rows": 2, "quarantined_rows": 4,
                                "identity_mismatch_rows": 1, "identity_pending_rows": 1, "invalid_label_rows": 2}
    assert report["n10_status"] == "data_isolated_retraining_pending"
    assert report["models_retrained"] is False
    assert report["cv_recomputed"] is False
    assert report["historical_artifacts"][0]["status"] == "historical_unretrained"
    accepted = _read(output / "training_eligible.csv")
    isolated = _read(output / "quarantined.csv")
    assert [row["molecule_uid"] for row in accepted] == ["keys", "strings"]
    for before, after in zip(rows[:2], accepted):
        assert {key: after[key] for key in before} == before
    reasons = {row["molecule_uid"]: row["audit_exclusion_reasons"] for row in isolated}
    assert reasons["label_conflict"] == "conflicting_solved_labels"
    assert reasons["invalid_label"] == "invalid_solved_label"
    assert accepted[0]["audit_source_row"] == "2"
    published_bytes = {p.name: p.read_bytes() for p in output.iterdir()}
    with pytest.raises(FileExistsError, match="choose a new release path"):
        audit_pilot_training_csv(source, output)
    assert {p.name: p.read_bytes() for p in output.iterdir()} == published_bytes


@pytest.mark.parametrize("content", [
    "molecule_uid,solved\nM,False\n",
    ",".join(_row()) + "\n",
    ",".join(_row()) + "\n" + ",".join(_row().values()) + ",extra_field\n",
])
def test_invalid_source_is_not_published(tmp_path: Path, content: str) -> None:
    source = tmp_path / "invalid.csv"
    source.write_text(content, encoding="utf-8")
    before = source.read_bytes()
    destination = tmp_path / "release"
    with pytest.raises(ValueError):
        audit_pilot_training_csv(source, destination)
    assert not destination.exists()
    assert source.read_bytes() == before
    assert not list(tmp_path.glob(".rascore_identity_*"))


def test_source_version_change_prevents_publication(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "teacher.csv"
    _write(source, [_row()])
    original_digest = rascore_identity._digest
    calls = 0

    def changed_source_digest(path: Path) -> str:
        nonlocal calls
        if path == source:
            calls += 1
            if calls == 2:
                return "0" * 64
        return original_digest(path)

    monkeypatch.setattr(rascore_identity, "_digest", changed_source_digest)
    with pytest.raises(ValueError, match="source changed"):
        audit_pilot_training_csv(source, tmp_path / "release")
    assert not (tmp_path / "release").exists()


def test_current_pilot_has_one_mismatch_and_400_exact_string_matches(tmp_path: Path) -> None:
    root = Path(__file__).parents[2]
    source = root / "rascore_pilot_v1/rascore_teacher_combined_1400.csv"
    if not source.is_file():
        pytest.skip("historical pilot CSV not checked out")
    before = source.read_bytes()
    output = tmp_path / "current_pilot_audit"
    report_path = audit_pilot_training_csv(source, output)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["counts"] == {"source_rows": 1400, "eligible_rows": 1399, "quarantined_rows": 1,
                                "identity_mismatch_rows": 1, "identity_pending_rows": 0, "invalid_label_rows": 0}
    assert report["accepted_identity_evidence"] == {"full_inchikey_match": 999, "exact_isomeric_smiles_match": 400}
    isolated = _read(output / "quarantined.csv")
    assert len(isolated) == 1
    assert isolated[0]["molecule_uid"] == "MOL_00000249675062"
    assert isolated[0]["teacher_stereo_identity_match"] == "False"
    assert isolated[0]["audit_exclusion_reasons"] == "explicit_identity_mismatch;full_inchikey_mismatch"
    assert source.read_bytes() == before
