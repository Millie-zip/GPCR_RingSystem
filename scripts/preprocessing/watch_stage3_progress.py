"""Watch the live JSON heartbeat emitted by build_stage3_assets.py."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


def _summary(state: dict[str, Any]) -> str:
    processed = state.get("processed", 0)
    limit = state.get("limit") or "?"
    current = state.get("current_index") or "-"
    molecule = state.get("current_molecule_uid") or "-"
    elapsed = state.get("phase_elapsed_seconds", 0)
    return (
        f"status={state.get('status')} | completed={processed}/{limit} | "
        f"current={current}/{limit} | phase={state.get('phase')} | "
        f"phase_seconds={elapsed} | molecule={molecule} | "
        f"PASS={state.get('passed', 0)} FAIL={state.get('failed', 0)} | "
        f"updated={state.get('updated_at')}"
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
                if state.get("last_error"):
                    print(f"last_error={state['last_error']}", flush=True)
                return
        if args.once:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
