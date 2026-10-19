"""Compare saved official NeMo Gym metrics; never calls a model or judge API."""

import argparse
import json
from pathlib import Path

from nemochat.tracking import add_tracking_args, from_args
from nemochat.evaluation.gym_benchmarks import compare_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gym-root", type=Path)
    parser.add_argument("--allow-fixtures", action="store_true", help="Explicit synthetic test artifacts only")
    parser.add_argument("--output", type=Path, required=True)
    add_tracking_args(parser)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a new artifact path")
    with from_args(args, "pretrain", run_type="evaluation", config=vars(args)) as tracking:
        try:
            result = compare_manifest(args.manifest, gym_root=args.gym_root,
                                      allow_fixtures=args.allow_fixtures)
        except (ValueError, KeyError, TypeError, IndexError, OSError) as exc:
            parser.error(str(exc))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
        print(json.dumps({"output": str(args.output), "fixture": result["fixture"],
                          "paired_task_results": len(result["pairs"])}))
        tracking.update_config({"evidence": "imported synthetic fixture" if result["fixture"] else "user-attested imported official results; not independently reproduced inference", "fixture": result["fixture"]})
        tracking.log({"evaluation": result})
        for pair in result["pairs"]:
            tracking.log({f"gym/{pair['task']}": {
                "checkpoint_step": pair["checkpoint_step"],
                "raw_score": pair["raw"]["score"],
                "merged_score": pair["merged"]["score"],
                "merged_minus_raw": pair["merged_minus_raw"]}})

if __name__ == "__main__":
    main()
