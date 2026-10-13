"""Analyze strict client request JSONL on CPU. No model, GPU, or network calls."""

import argparse
import json
from pathlib import Path
import sys

# Keep the script directly runnable without an editable installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemochat.inference_perf import analysis


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, epilog=analysis.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--requests", type=Path, required=True, help="Strict schema-v1 JSONL")
    parser.add_argument("--config", type=Path, required=True, help="Explicit clock window and SLO JSON")
    parser.add_argument("--output", type=Path, required=True, help="New report JSON path; never overwritten")
    parser.add_argument("--allow-synthetic", action="store_true", help="Accept explicitly labeled synthetic fixtures, not measured inference")
    args = parser.parse_args(argv)
    try:
        if args.output.exists() or args.output.is_symlink():
            raise ValueError("output already exists; choose a new path")
        config = analysis.load_config(args.config)
        records = analysis.load_records(args.requests, allow_synthetic=args.allow_synthetic)
        report = analysis.analyze(records, config, allow_synthetic=args.allow_synthetic)
        report["sources"] = {"requests": str(args.requests.resolve()), "config": str(args.config.resolve())}
        serialized = json.dumps(report, indent=2, allow_nan=False) + "\n"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation also rejects a race or a dangling symlink.
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(serialized)
    except (ValueError, TypeError, OSError, OverflowError) as exc:
        parser.error(str(exc))
    print(json.dumps({"output": str(args.output), "evidence": report["evidence"],
                      "counts": report["counts"],
                      "slo_goodput_requests_per_s": report["slo_goodput_requests_per_s"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
