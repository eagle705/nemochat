"""Fresh-process, rotated-order Lens overhead measurement (CPU control only)."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

from train import MODES, clean_environment, loopback_endpoint
from verify_jaeger import fetch_trace, wait_ready


def equivalent(rows):
    reference = rows[0]
    for row in rows[1:]:
        if row["parameter_sha256"] != reference["parameter_sha256"] or row["losses"] != reference["losses"]:
            raise ValueError("Training correctness differs across modes")


def verify_file(path, result):
    spans = [json.loads(line) for line in path.read_text().splitlines()]
    expected = Counter({"train.run": 1, "train.step": result["steps"], "train.forward": result["steps"],
                        "train.backward": result["steps"], "train.optimizer": result["steps"]})
    if len(spans) != result["expected_spans"] or Counter(span["name"] for span in spans) != expected:
        raise ValueError("file exporter did not write complete spans")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--warmup", type=int, default=40)
    parser.add_argument("--repetitions", type=int, default=8)
    parser.add_argument("--skip-otlp", action="store_true")
    parser.add_argument("--api", default="http://127.0.0.1:16687")
    parser.add_argument("--endpoint", type=loopback_endpoint, default="http://127.0.0.1:14318/v1/traces")
    args = parser.parse_args()
    if args.steps < 1 or args.warmup < 0 or args.repetitions < 2:
        parser.error("positive steps, nonnegative warmup and at least two repetitions required")
    clean_environment()
    modes = list(MODES[:-1] if args.skip_otlp else MODES)
    if not args.skip_otlp:
        wait_ready(args.api)
    args.output.mkdir(parents=True, exist_ok=False)
    rows, order = [], []
    # Rotate first position; default eight repetitions balances all four positions twice.
    for repetition in range(args.repetitions):
        sequence = modes[repetition % len(modes):] + modes[:repetition % len(modes)]
        for mode in sequence:
            child = args.output / f"r{repetition:02d}-{mode}"
            command = [sys.executable, str(Path(__file__).with_name("train.py")), "--mode", mode,
                       "--steps", str(args.steps), "--warmup", str(args.warmup),
                       "--endpoint", args.endpoint, "--output", str(child)]
            start = time.perf_counter_ns()
            completed = subprocess.run(command, capture_output=True, text=True, timeout=120, check=True)
            process_ns = time.perf_counter_ns() - start
            row = json.loads((child / "result.json").read_text())
            row.update(repetition=repetition, process_ns=process_ns)
            (child / "stdout.log").write_text(completed.stdout)
            (child / "stderr.log").write_text(completed.stderr)
            if mode == "file":
                verify_file(child / "spans.jsonl", row)
            if mode == "otlp":
                payload, receipt = fetch_trace(args.api, row)
                (child / "jaeger_trace.json").write_text(json.dumps(payload, indent=2))
                (child / "jaeger_receipt.json").write_text(json.dumps(receipt, indent=2))
            rows.append(row)
            order.append(child.name)
            print(child.name, row["training_ns"], flush=True)
    equivalent(rows)
    medians = {mode: {field: statistics.median(row[field] for row in rows if row["mode"] == mode)
                      for field in ("training_ns", "setup_ns", "shutdown_ns", "process_ns")} for mode in modes}
    baseline = medians["baseline"]["training_ns"]
    for mode in modes:
        medians[mode]["training_overhead_percent"] = (medians[mode]["training_ns"] / baseline - 1) * 100
    summary = dict(scope="offline generated-data CPU MLP; NOT NemoChat GPU training", order=order,
                   steps=args.steps, warmup=args.warmup, repetitions=args.repetitions,
                   correctness="exact loss sequence and parameter SHA256 equal", raw=rows,
                   medians=medians, otlp="skipped explicitly" if args.skip_otlp else "all traces API-verified")
    (args.output / "benchmark.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(medians, indent=2))


if __name__ == "__main__":
    main()
