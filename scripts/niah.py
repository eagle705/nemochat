#!/usr/bin/env python3
"""Generate synthetic fixtures, run real completions, score or plot JSONL."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nemochat.long_context.io import FreshOutputs, read_jsonl
from nemochat.long_context.niah import generate_fixtures, run_predictions, score_predictions
from nemochat.long_context.plot import plot_predictions


def load_tokenizer(args):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(args.tokenizer, revision=args.revision,
                                         trust_remote_code=False, local_files_only=args.local_files_only)


def tokenizer_args(parser):
    parser.add_argument("--tokenizer", required=True, help="HF tokenizer ID or local directory")
    parser.add_argument("--revision", default="main", help="Pin a commit for reproducibility")
    parser.add_argument("--local-files-only", action="store_true")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate", help="Synthetic inputs only; no model performance")
    tokenizer_args(generate)
    generate.add_argument("--config", help="JSON fixture parameters; config values override explicit CLI flags")
    generate.add_argument("--context-lengths", type=int, nargs="+", default=[4096, 65536])
    generate.add_argument("--depths", type=float, nargs="+", default=[0., .5, 1.])
    generate.add_argument("--random-depths", type=int, default=0, help="Add frozen seeded random depths")
    generate.add_argument("--samples", type=int, default=3)
    generate.add_argument("--seed", type=int, default=1729)
    generate.add_argument("--generation-tokens", type=int, default=32)
    generate.add_argument("--haystack", help="Optional UTF-8 corpus file")
    generate.add_argument("--output", required=True)
    run = commands.add_parser("run", help="Call an actual plain-text completion endpoint")
    tokenizer_args(run)
    run.add_argument("--input", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--endpoint", required=True, help="Full URL, e.g. http://localhost:8000/v1/completions")
    run.add_argument("--model", required=True)
    run.add_argument("--timeout", type=float, default=60)
    run.add_argument("--api-key-env", default="OPENAI_API_KEY")
    score = commands.add_parser("score")
    score.add_argument("--input", required=True)
    score.add_argument("--output", required=True, help="Cell summary JSONL")
    score.add_argument("--csv", help="Optional CSV summary")
    plot = commands.add_parser("plot")
    plot.add_argument("--input", required=True, help="Fixture or actual prediction JSONL")
    plot.add_argument("--output-prefix", required=True, help="Writes .png, .svg, .metadata.json")
    args = parser.parse_args(argv)
    try:
        inputs = [getattr(args, name, None) for name in ("input", "config", "haystack")]
        if args.command in ("generate", "run"):
            inputs.append(args.tokenizer)
        inputs = [p for p in inputs if p]
        destinations = ([str(args.output_prefix) + suffix for suffix in (".png", ".svg", ".metadata.json")]
                        if args.command == "plot" else [args.output])
        if args.command == "score" and args.csv:
            destinations.append(args.csv)
        with FreshOutputs(destinations, inputs) as outputs:
            message = execute(args, outputs)
        print(message)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.error(str(exc))


def execute(args, outputs):
    if args.command == "generate":
        options = {"context_lengths": args.context_lengths, "depths": args.depths, "samples": args.samples,
                   "seed": args.seed, "generation_tokens": args.generation_tokens,
                   "random_depths": args.random_depths}
        if args.config:
            with open(args.config, encoding="utf-8") as stream:
                configured = json.load(stream)
            if not isinstance(configured, dict) or set(configured) - set(options):
                raise ValueError("unknown fixture config fields")
            options.update(configured)
        if args.haystack:
            options["haystack"] = Path(args.haystack).read_text(encoding="utf-8")
        rows = generate_fixtures(load_tokenizer(args), **options)
        for row in rows:
            row["tokenizer_revision"] = args.revision
        outputs.write_jsonl(args.output, rows)
        return f"Wrote {len(rows)} synthetic fixtures; no model predictions"
    elif args.command == "run":
        rows = run_predictions(read_jsonl(args.input), load_tokenizer(args), args.endpoint,
                               args.model, args.timeout, args.api_key_env)
        outputs.write_jsonl(args.output, rows)
        return f"Recorded {len(rows)} requests; {sum(r['status'] == 'ok' for r in rows)} validated responses"
    elif args.command == "score":
        cells = score_predictions(read_jsonl(args.input))
        outputs.write_jsonl(args.output, cells)
        if args.csv:
            csv_rows = [{**cell, "statuses": json.dumps(cell["statuses"], sort_keys=True)} for cell in cells]
            outputs.write_csv(args.csv, csv_rows, list(csv_rows[0]))
        return f"Wrote {len(cells)} cells (excluded requests are not failures)"
    else:
        paths = plot_predictions(read_jsonl(args.input), args.output_prefix, outputs=outputs)
        return "Wrote " + ", ".join(paths)


if __name__ == "__main__":
    main()
