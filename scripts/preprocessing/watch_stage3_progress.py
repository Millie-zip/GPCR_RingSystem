"""Watch the live JSON heartbeat emitted by build_stage3_assets.py."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


def _summary(state: dict[str, Any]) -> str:
    processed = state.get("processed_molecules", state.get("processed", 0))
    limit = state.get("requested_limit", state.get("limit")) or "?"
    current = state.get("current_index") or "-"
    molecule = state.get("current_molecule_uid") or "-"
    elapsed = state.get("phase_elapsed_seconds", state.get("elapsed_seconds", 0))
    passed = state.get("passed_molecules", state.get("passed", 0))
    failed = state.get("failed_molecules", state.get("failed", 0))
    updated = state.get("updated_at_utc", state.get("updated_at"))
    eta = state.get("estimated_remaining_seconds")
    return (
        f"status={state.get('status')} | completed={processed}/{limit} | "
        f"current={current}/{limit} | phase={state.get('phase')} | "
        f"elapsed_seconds={elapsed} | eta_seconds={eta} | molecule={molecule} | "
        f"PASS={passed} FAIL={failed} | updated={updated}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch a Stage3 build heartbeat.")
    parser.add_argument("progress_file", type=Path)
    parser.add_argument("--interval", type=float, default=15.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval must be positive")

    while True:
        try:
            state = json.loads(args.progress_file.read_text(encoding="utf-8"))
        except FileNotFoundError:
            print(f"waiting for {args.progress_file}", flush=True)
        except json.JSONDecodeError as exc:
            print(f"waiting for an atomic progress update: {exc}", flush=True)
        else:
            print(_summary(state), flush=True)
            if args.once or state.get("status") in {
                "complete",
                "failed",
                "interrupted",
            }:
                error = state.get("last_error", state.get("message"))
                if error:
                    print(f"last_error={error}", flush=True)
                return
        if args.once:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
