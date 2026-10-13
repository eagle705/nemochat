#!/usr/bin/env python3
"""Emit a resumable long-context schedule, without training any model."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nemochat.tracking import add_tracking_args, from_args
from nemochat.long_context.schedule import make_plan, load_cursor, advance_cursor, save_cursor
from nemochat.long_context.io import write_jsonl


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--long-only-steps", type=int, default=2)
    parser.add_argument("--alternating-steps", type=int, default=4)
    parser.add_argument("--effective-tokens", type=int, default=131072)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--cursor", required=True, help="Single-writer JSON cursor, created if missing")
    parser.add_argument("--output", required=True, help="JSONL batch of emitted steps (overwritten atomically)")
    parser.add_argument("--count", type=int, default=1)
    add_tracking_args(parser)
    args = parser.parse_args(argv)
    if Path(args.cursor).resolve() == Path(args.output).resolve():
        parser.error("cursor and output paths must differ")
    with from_args(args, "longcontext", run_type="schedule-plan", config=vars(args)) as tracking:
        try:
            plan = make_plan(args.long_only_steps, args.alternating_steps, args.effective_tokens,
                             args.micro_batch_size, args.world_size)
            cursor = load_cursor(args.cursor, plan)
            records, cursor = advance_cursor(cursor, args.count)
            write_jsonl(args.output, records)
            save_cursor(args.cursor, cursor)
        except (ValueError, OSError) as exc:
            parser.error(str(exc))
        print(f"Emitted {len(records)} steps; next_step={cursor['next_step']}; scheduler only, no training")
        tracking.log({"planned_steps_emitted": len(records), "next_planned_step": cursor["next_step"]})
        tracking.update_config({"evidence": "schedule planning only; no training executed"})

if __name__ == "__main__":
    main()
