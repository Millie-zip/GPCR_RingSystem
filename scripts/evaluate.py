"""Run structural/test-time validation for a Stage 3 output JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gpcr_ringgen.evaluation import validate_generated_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Serialized Stage 3 output JSON.")
    parser.add_argument("--output", help="Optional validation report JSON path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    report = validate_generated_output(payload)
    serialized = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(serialized + "\n", encoding="utf-8")
    else:
        print(serialized)
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
