#!/usr/bin/env python3
"""Prepare one fresh bounded flat-token pack from a JSON config."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nemochat.tracking import add_tracking_args, from_args
from nemochat.data import prepare_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True, help="fresh output directory; never overwritten")
    add_tracking_args(parser)
    args = parser.parse_args()
    with from_args(args, "data", run_type="stage", config=vars(args)) as tracking:
        try:
            with open(args.config, encoding="utf-8") as handle:
                config = json.load(handle)
            tracking.update_config(config)
            manifest = prepare_data(config, args.output)
            tracking.log({"data/" + k: v for k, v in manifest["counters"].items()})
            tracking.update_config({"quota_reached": manifest["quota_reached"], "output": args.output})
            tracking.artifact_metadata("dataset-" + tracking.id, kind="dataset-metadata",
                metadata={"path": str(Path(args.output).resolve()), "counters": manifest["counters"],
                          "tokens_sha256": manifest["tokens_sha256"], "data_uploaded": False})
        except Exception as exc:
            print(f"data preparation failed: {exc}", file=sys.stderr)
            raise
        print(json.dumps({"output": str(Path(args.output).absolute()),
                          "quota_reached": manifest["quota_reached"],
                          "counters": manifest["counters"]}, indent=2))
        return 0

if __name__ == "__main__":
    raise SystemExit(main())
