#!/usr/bin/env python3
"""Write a model-relative single-operator roofline report (stdlib only)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nemochat.planning.inference_roofline import report_from_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Strict JSON operator inputs with provenance")
    parser.add_argument("--output", type=Path, required=True, help="New JSON report path; never overwritten")
    parser.add_argument("--allow-synthetic", action="store_true", help="Explicitly accept illustrative synthetic inputs")
    args = parser.parse_args()
    try:
        report = report_from_json(args.config.read_bytes(), allow_synthetic=args.allow_synthetic)
        encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
        # Exclusive creation also rejects existing symlinks and input/output aliasing.
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
    except (OSError, ValueError, OverflowError, RecursionError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(f"Wrote {args.output}: {report['status']} / {report['classification_model_estimate']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
