#!/usr/bin/env python3
"""Execute a real offline dense full-update control into a fresh directory."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nemochat.training_perf import run_benchmark
from nemochat.experiments import MAX_CONFIG_BYTES, load_strict_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="complete resolved schema-1 JSON")
    parser.add_argument("--output", required=True, help="fresh measurement directory")
    args = parser.parse_args()
    try:
        config = load_strict_json(args.config, max_bytes=MAX_CONFIG_BYTES)
        result = run_benchmark(config, args.output)
    except (ValueError, OSError, TypeError) as exc:
        print(f"benchmark rejected: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"status": result["status"], "evidence_scope": result["evidence_scope"],
                      "result": str(Path(args.output) / "result.json"),
                      "failure_kind": result.get("failure_kind")}, allow_nan=False))
    if result.get("failure_kind") == "time-budget-exceeded":
        return 4
    if result.get("failure_kind") == "out-of-memory":
        return 3
    return 0 if result["status"] == "success" else 3 if result["status"] == "blocked" else 1


if __name__ == "__main__":
    raise SystemExit(main())
