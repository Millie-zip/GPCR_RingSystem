"""Read-only identity checks and a separate training-input release for pilot CSVs."""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile
from typing import Mapping, Sequence


POLICY_VERSION = "rascore_stereo_identity_v1"
_INCHIKEY = re.compile(r"[A-Z]{14}-[A-Z]{10}-[A-Z]")
_AUDIT_COLUMNS = (
    "audit_source_row", "audit_source_sha256", "audit_status",
    "audit_identity_evidence", "audit_exclusion_reasons",
)
_REQUIRED_COLUMNS = {"molecule_uid", "canonical_isomeric_smiles", "teacher_target", "solved"}


@dataclass(frozen=True)
class IdentityDecision:
    status: str
    evidence: str
    reasons: tuple[str, ...] = ()


def _boolean(value: str) -> bool | None:
    text = value.strip().casefold()
    if text in {"true", "1"}:
        return True
    if text in {"false", "0"}:
        return False
    return None


def classify_identity(row: Mapping[str, str]) -> IdentityDecision:
    """Use supplied full keys or exact isomeric strings; never infer equivalence."""
    flag_text = row.get("teacher_stereo_identity_match", "").strip()
    flag = _boolean(flag_text)
    first = row.get("input_full_inchikey", "").strip()
    second = row.get("teacher_result_full_inchikey", "").strip()
    reasons = []
    if flag is False:
        reasons.append("explicit_identity_mismatch")
    keys_valid = bool(_INCHIKEY.fullmatch(first) and _INCHIKEY.fullmatch(second))
    if keys_valid and first != second:
        reasons.append("full_inchikey_mismatch")
    # Explicit contradictory evidence cannot be overridden by equal strings.
    if reasons:
        return IdentityDecision("mismatch", "conflicting_identity_evidence", tuple(reasons))
    if flag_text and flag is None:
        return IdentityDecision("pending", "invalid_identity_flag", ("invalid_identity_flag",))
    if any(key and not _INCHIKEY.fullmatch(key) for key in (first, second)):
        return IdentityDecision("pending", "invalid_full_inchikey", ("invalid_full_inchikey",))
    input_smiles = row.get("canonical_isomeric_smiles", "").strip()
    target_smiles = row.get("teacher_target", "").strip()
    if not row.get("molecule_uid", "").strip() or not input_smiles or not target_smiles:
        return IdentityDecision("pending", "missing_identity_fields", ("missing_identity_fields",))
    if keys_valid:
        return IdentityDecision("match", "full_inchikey_match")
    if input_smiles == target_smiles:
        return IdentityDecision("match", "exact_isomeric_smiles_match")
    return IdentityDecision("pending", "insufficient_identity_evidence", ("insufficient_identity_evidence",))


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def _staging_directory(parent: Path):
    """Stage beside the private temporary directory to inherit output-parent ACLs."""
    with tempfile.TemporaryDirectory(prefix=".rascore_identity_", dir=parent) as temporary:
        claim = Path(temporary).resolve()
        if claim.parent != parent:
            raise ValueError("temporary directory escaped the intended output parent")
        # A child of TemporaryDirectory inherits its private Windows ACL. A
        # sibling created normally instead inherits the intended parent ACL.
        staging = claim.with_name(claim.name + "_release")
        staging.mkdir()
        try:
            yield staging
        finally:
            if staging.exists():
                if staging.resolve().parent != parent:
                    raise ValueError("staging directory escaped the intended output parent")
                shutil.rmtree(staging)


def _markdown(report: dict) -> str:
    counts = report["counts"]
    evidence = report["accepted_identity_evidence"]
    return (
        "# RAscore 训练输入身份校验\n\n"
        f"规则版本：`{POLICY_VERSION}`。生成时间（UTC）：{report['created_utc']}。\n\n"
        f"原表 {counts['source_rows']} 条；可用 {counts['eligible_rows']} 条；"
        f"隔离 {counts['quarantined_rows']} 条，其中明确身份不匹配 {counts['identity_mismatch_rows']} 条、"
        f"身份待核对 {counts['identity_pending_rows']} 条、标签不合格 {counts['invalid_label_rows']} 条"
        "（各原因计数可能重叠）。\n\n"
        f"可用记录的证据：Full InChIKey 一致 {evidence.get('full_inchikey_match', 0)} 条；"
        f"isomeric SMILES 与 teacher target 完全一致 {evidence.get('exact_isomeric_smiles_match', 0)} 条。\n\n"
        "- `training_eligible.csv`：通过本规则的训练输入，原始列保持不变，追加校验字段。\n"
        "- `quarantined.csv`：明确冲突、证据不足或标签不合格的记录及隔离原因。\n"
        "- `audit.json`：原始文件、输出文件的 SHA-256，计数、逐条隔离摘要和历史产物状态。\n\n"
        "校验只使用已保存的身份字段，不调用 RDKit、InChI 或 teacher。Full InChIKey 未重新计算；"
        "字符串不同时不会推断两者等价，True 标记或相同的非立体 SMILES 也不能单独证明匹配。"
        "显式 False 或不同的完整 InChIKey 优先隔离。\n\n"
        "原始 teacher 表、模型、CV 和导出包均保留。这里没有重新训练模型或重算 CV；"
        "旧预测不能通过删去一行变成新 CV 结果。历史产物列表见 `audit.json`，其 status 均为"
        "`historical_unretrained`，不能作为清洗后模型的结果使用。原训练/CV 流程补齐并重新生成产物后，"
        "N10 才能完全关闭。\n"
    )


def audit_pilot_training_csv(
    source: str | Path,
    output_dir: str | Path,
    *,
    historical_artifacts: Sequence[str | Path] = (),
) -> Path:
    """Publish a new directory; existing input and output artifacts are never overwritten."""
    source_path = Path(source).resolve()
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError(f"output directory exists; choose a new release path: {destination}")
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if source_path.is_relative_to(destination):
        raise ValueError("output directory must not contain the source file")
    source_hash = _digest(source_path)
    historical = []
    for value in historical_artifacts:
        artifact = Path(value).resolve()
        if not artifact.is_file():
            raise FileNotFoundError(artifact)
        historical.append({"path": str(artifact), "sha256": _digest(artifact), "status": "historical_unretrained"})
    counts = Counter()
    accepted_evidence = Counter()
    excluded_reasons = Counter()
    excluded = []
    destination.parent.mkdir(parents=True, exist_ok=True)
    with _staging_directory(destination.parent) as staging:
        with source_path.open(encoding="utf-8-sig", newline="") as input_handle, \
                (staging / "training_eligible.csv").open("w", encoding="utf-8", newline="") as eligible_handle, \
                (staging / "quarantined.csv").open("w", encoding="utf-8", newline="") as quarantine_handle:
            reader = csv.DictReader(input_handle)
            fields = reader.fieldnames
            if not fields or len(set(fields)) != len(fields) or not _REQUIRED_COLUMNS.issubset(fields):
                raise ValueError("CSV needs unique headers including molecule_uid, canonical_isomeric_smiles, teacher_target and solved")
            if set(fields).intersection(_AUDIT_COLUMNS):
                raise ValueError("source already has audit columns; use the original teacher table")
            writers = [csv.DictWriter(handle, fieldnames=[*fields, *_AUDIT_COLUMNS], lineterminator="\n")
                       for handle in (eligible_handle, quarantine_handle)]
            for writer in writers:
                writer.writeheader()
            for source_row, row in enumerate(reader, start=2):
                if None in row or any(value is None for value in row.values()):
                    raise ValueError(f"malformed CSV record at source row {source_row}")
                decision = classify_identity(row)
                reasons = list(decision.reasons)
                solved = _boolean(row["solved"])
                teacher_text = row.get("teacher_is_solved", "").strip()
                teacher_solved = _boolean(teacher_text)
                if solved is None:
                    reasons.append("invalid_solved_label")
                if teacher_text and teacher_solved is None:
                    reasons.append("invalid_teacher_label")
                elif teacher_solved is not None and solved is not None and teacher_solved != solved:
                    reasons.append("conflicting_solved_labels")
                invalid_label = len(reasons) > len(decision.reasons)
                eligible = decision.status == "match" and not invalid_label
                counts["source_rows"] += 1
                counts["eligible_rows" if eligible else "quarantined_rows"] += 1
                counts["identity_mismatch_rows"] += decision.status == "mismatch"
                counts["identity_pending_rows"] += decision.status == "pending"
                counts["invalid_label_rows"] += invalid_label
                audit = {
                    "audit_source_row": str(source_row), "audit_source_sha256": source_hash,
                    "audit_status": "eligible" if eligible else "quarantined",
                    "audit_identity_evidence": decision.evidence,
                    "audit_exclusion_reasons": ";".join(reasons),
                }
                writers[0 if eligible else 1].writerow({**row, **audit})
                if eligible:
                    accepted_evidence[decision.evidence] += 1
                else:
                    excluded_reasons.update(reasons)
                    excluded.append({"source_row": source_row, "molecule_uid": row["molecule_uid"],
                                     "identity_status": decision.status, "evidence": decision.evidence, "reasons": reasons})
        if counts["source_rows"] == 0:
            raise ValueError("source teacher table has no records")
        if _digest(source_path) != source_hash:
            raise ValueError("source changed during identity audit; release not published")
        report = {
            "policy_version": POLICY_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source": {"path": str(source_path), "sha256": source_hash},
            "counts": {key: counts[key] for key in ("source_rows", "eligible_rows", "quarantined_rows",
                                                    "identity_mismatch_rows", "identity_pending_rows", "invalid_label_rows")},
            "accepted_identity_evidence": dict(accepted_evidence),
            "quarantine_reasons": dict(excluded_reasons),
            "quarantined_records": excluded,
            "historical_artifacts": historical,
            "models_retrained": False,
            "cv_recomputed": False,
            "n10_status": "data_isolated_retraining_pending",
            "outputs": {name: {"sha256": _digest(staging / name)} for name in ("training_eligible.csv", "quarantined.csv")},
        }
        (staging / "audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        (staging / "README.md").write_text(_markdown(report), encoding="utf-8")
        # rename refuses an already-existing destination on Windows; the
        # explicit second check also prevents routine accidental replacement.
        if destination.exists():
            raise FileExistsError(f"output directory appeared during audit: {destination}")
        staging.rename(destination)
    return destination / "audit.json"
