# Offline inference timing analyzer fixtures

**These files are handwritten synthetic mechanics fixtures. No inference ran and
none of their numbers measures a model, GPU, serving engine, or quality.**
`synthetic_requests.jsonl` labels every record `synthetic: true`; accepting it
requires `--allow-synthetic`. Do not relabel these records as measured data.

From the repository root, choose a new output path:

```bash
python3 scripts/analyze_inference_requests.py --help
python3 scripts/analyze_inference_requests.py \
  --requests configs/inference_perf/synthetic_requests.jsonl \
  --config configs/inference_perf/synthetic_config.json \
  --allow-synthetic --output /tmp/nemochat-synthetic-inference-report.json
.venv/bin/python -m pytest -q tests/test_inference_perf.py
```

The script and analyzer use only the Python standard library. The exact schema
and metric formulas are in `src/nemochat/inference_perf/analysis.py` and CLI help.
No output overwrites are permitted, including symlinks. Malformed JSON, blank
lines, unknown/missing fields, repeated JSON keys/request IDs, nonfinite or
nonmonotonic timestamps, invalid counts, token-count/timing disagreements, mixed
synthetic/real evidence, and out-of-window requests cause errors, not exclusions.

The fixture covers four successes (one SLO pass, one violation, two unavailable),
one error, one timeout, and one cancellation. Zero output and absent per-token
timing yield unavailable TTFT/gap samples where applicable, not fabricated zeros.
Counts of actual tokens include unsuccessful requests separately from success.

## Measurement contract

Use one monotonic client clock domain with explicit seconds units. Record all
admitted attempts, including those ending in failure/cancellation. Config defines
`[measurement_start_s, measurement_end_s)` arrivals; completions must fit at or
before the end. Capture/drain a complete cohort and choose the full explicit
window including drain time. The tool rejects censored requests rather than
silently dropping slow attempts. Throughput/goodput denominators include the
entire configured interval, including idle time. This is not a steady-state
serving/load generator or a scheduler simulator.

TTFT/E2E are send-relative, while arrival-to-completion and arrival-to-send are
separate metrics. Latencies/gaps summarize successes only. Token gaps are pooled
adjacent individual output token observations; streaming transport chunks are NOT
tokens and must use `token_timing: "unavailable"`, `token_arrivals_s: null` unless
there is independent individual-token timing. Repeating a chunk timestamp to
match its token count violates the producer contract. The analyzer can reject a
`chunks` timing label but cannot discover dishonestly relabeled observations.

A maximum-token-gap SLO cannot be established with missing token timing or fewer
than two output tokens. Such successes are SLO-unavailable unless a known
threshold violation already proves failure. They never contribute to goodput.
Disable an irrelevant SLO explicitly with null; at least one threshold is needed.
Nearest-rank sample counts and low-count p99 warnings are reported per metric;
100 samples is a minimum granularity check, NOT a statistical confidence gate.
Pooled token gaps are correlated within requests and token-weighted.

Workload/tokenizer/decoding IDs attest frozen measurement protocols, not hashes
verified by this analyzer. Imported records cannot establish model identity,
quality parity, missed attempts, true GPU utilization, or hardware headroom.
Cross-report quality/performance comparison is intentionally deferred: a credible
paired comparison also needs verified engine/model/workload identities and a
compatible independently scored quality artifact. Do not compare arbitrary
reports or call this a measured optimization/speedup.
