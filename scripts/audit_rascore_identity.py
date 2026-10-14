"""Validate pilot identity fields and write separate eligible/quarantined CSVs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gpcr_ringgen.preprocessing.rascore_identity import audit_pilot_training_csv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path, help="New directory; existing releases are not overwritten")
    parser.add_argument("--historical-artifact", action="append", type=Path, default=[], help="Record an unretrained model, CV table or export in the audit")
    args = parser.parse_args()
    try:
        path = audit_pilot_training_csv(args.source, args.output_dir, historical_artifacts=args.historical_artifact)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Identity audit failed: {error}\n")
    report = json.loads(path.read_text(encoding="utf-8"))
    print(json.dumps({"audit": str(path), "counts": report["counts"], "n10_status": report["n10_status"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
