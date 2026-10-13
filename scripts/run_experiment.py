#!/usr/bin/env python3
"""Validate, preview or execute one strictly bounded offline dense control."""
import argparse
import json
from pathlib import Path
import sys

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from nemochat.experiments import (ContractError, EVIDENCE, benchmark_argv, load_config, run_experiment)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate", "dry-run", "run"))
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.command == "run" and args.output is None:
        parser.error("run requires --output (a fresh directory)")
    try:
        resolved = load_config(args.config)
        if args.command != "run":
            result = {"status": "valid" if args.command == "validate" else "dry_run", "source_sha256": resolved.source_sha256, "config": resolved.to_dict(), "evidence_label": EVIDENCE, "hardware_checked": False}
            if args.command == "dry-run":
                # A placeholder is displayed, never created or interpreted as a command.
                result["argv"] = benchmark_argv(args.output or Path("OUTPUT_REQUIRED_FOR_RUN"), REPOSITORY)
            print(json.dumps(result, indent=2))
            return 0
        result = run_experiment(resolved, args.output, REPOSITORY)
        print(json.dumps(result, indent=2))
        return result["exit_code"]
    except (ContractError, OSError, ImportError) as exc:
        print(f"experiment error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
