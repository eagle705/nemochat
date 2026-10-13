#!/usr/bin/env python3
"""Compare compatible full-update artifacts; never adopt solely on speed."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nemochat.training_perf import compare_results
from nemochat.experiments import load_strict_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True, help="fresh JSON file")
    args = parser.parse_args()
    try:
        destination = Path(args.output)
        if destination.exists():
            raise ValueError("output already exists")
        baseline = load_strict_json(args.baseline, max_bytes=8 * 1024 * 1024)
        candidate = load_strict_json(args.candidate, max_bytes=8 * 1024 * 1024)
        result = compare_results(baseline, candidate)
        result["inputs"] = {"baseline": str(Path(args.baseline)), "candidate": str(Path(args.candidate))}
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, allow_nan=False)
            handle.write("\n")
    except (ValueError, OSError, TypeError, KeyError, OverflowError, RecursionError) as exc:
        print(f"comparison rejected: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"status": result["status"], "decision": result["decision"],
                      "candidate_to_baseline_median_ratio": result["candidate_to_baseline_median_ratio"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
