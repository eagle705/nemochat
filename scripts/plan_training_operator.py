#!/usr/bin/env python3
"""Estimate dense linear forward/backward GEMMs; roofline needs supplied DRAM evidence."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nemochat.planning.training_operator import report_from_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-synthetic", action="store_true")
    args = parser.parse_args()
    try:
        with args.config.open("rb") as stream:
            raw = stream.read(65537)
        report = report_from_json(raw, allow_synthetic=args.allow_synthetic)
        encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
    except (ValueError, OSError, OverflowError, RecursionError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(f"Wrote {args.output}: analytical estimates, not GPU measurements")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
