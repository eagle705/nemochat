#!/usr/bin/env python3
"""Run a local explicit JSON configuration; publish a fresh report, never overwrite."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nemochat.planning.budget import plan_training


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        destination = Path(args.output)
        if destination.exists():
            raise ValueError("output already exists")
        with open(args.config, encoding="utf-8") as handle:
            config = json.load(handle)
        report = plan_training(config)
        payload = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x", encoding="utf-8") as handle:
            handle.write(payload)
    except Exception as exc:
        print(f"plan_training failed: {exc}", file=sys.stderr)
        return 1
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
