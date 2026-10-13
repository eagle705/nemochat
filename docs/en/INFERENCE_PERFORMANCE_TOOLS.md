# Inference performance tools: local contracts

Track: [design and GPU gates](INFERENCE_PERFORMANCE_TRACK.md). These stdlib tools analyze saved producer-attested inputs on CPU. They do not start an engine, send traffic, measure a GPU or evaluate model quality.

## Saved-request analysis

```bash
python3 scripts/analyze_inference_requests.py --help
mkdir -p outputs/inference_perf/local-smoke
python3 scripts/analyze_inference_requests.py \
  --requests configs/inference_perf/synthetic_requests.jsonl \
  --config configs/inference_perf/synthetic_config.json \
  --output outputs/inference_perf/local-smoke/requests.json \
  --allow-synthetic
```

The directory can exist, but the output file must be new. Example inputs are explicitly synthetic, not measured service performance. Without `--allow-synthetic` the fixture is rejected. Invalid records and existing outputs fail without producing a successful report.

### Request JSONL schema v1

Every line is one JSON object; no blank lines, duplicate keys/IDs or unknown fields. All records share one client monotonic clock domain in seconds.

- `schema_version`: integer `1`.
- `request_id`: unique nonempty string.
- `synthetic`: boolean; true requires opt-in.
- `clock`: `client_monotonic_seconds`.
- `status`: `success`, `error`, `timeout`, `cancelled`.
- `arrival_s`, `send_s`, `completion_s`: finite nonnegative timestamps, arrival ≤ send ≤ completion. Arrival is the producer's admitted/scheduled attempt timestamp, not independently observed by the importer.
- `first_token_s`: timestamp or null; positive output requires a value between send and completion; zero output requires null.
- `input_tokens`, `output_tokens`: actual nonnegative integer counts, including partial failed output; not configured budgets or character counts.
- `token_timing`: `individual_tokens` or `unavailable`.
- `token_arrivals_s`: one ordered timestamp per output token, with first equal to `first_token_s`, or null when unavailable. Zero output with individual timing uses `[]`.

Equal per-token times are allowed for individually identified tokens, but repeating a chunk timestamp to fabricate token observations violates the producer contract. The importer cannot authenticate timing instrumentation. For ordinary chunk streaming use unavailable token timing, and do not call chunk gaps token ITL. Counts for failed attempts must still be supplied by the producer; the importer cannot recover missing usage.

### Config schema v1

Exact fields: `schema_version`, nonempty `workload_id`, `tokenizer_id`, `decoding_id`, `measurement_start_s`, `measurement_end_s`, and `slo`.

`measurement_end_s > measurement_start_s`. Every arrival is in `[start,end)` and every terminal completion is ≤ end. This intentionally requires a fully drained attempt cohort: it does not silently filter warmup, drop in-flight attempts or infer duration. Collect warmup separately and set the explicit drain-inclusive denominator. A future sustained-window adapter needs a different documented censoring policy, not silent reuse of this one.

`slo` has exactly `ttft_s`, `e2e_s`, `max_token_gap_s`, each a nonnegative threshold or null; at least one must be set. All configured thresholds must pass inclusively. Token-gap SLO uses the per-request maximum observed adjacent gap; missing timing or fewer than two output tokens is unknown and cannot count as goodput. This conservative convention is explicit, not a universal serving definition.

### Metric meanings

- Client TTFT: first token minus send.
- Client E2E: completion minus send.
- Arrival-to-send and arrival-to-completion: separate producer-clock delay metrics; they are not measured server queue time.
- Percentiles: nearest-rank successful-request samples. Token gaps pool adjacent individually observed successful output tokens, not server decode durations. Report sample counts/coverage and small-sample P99 warnings.
- SLO goodput: success requests satisfying all configured SLOs divided by explicit window duration. Error/timeout/cancelled remain in attempt counts. Missing SLO samples do not pass.

Saved-record analysis is not a complete benchmark run record. Keep engine/model/source/hardware/topology/config identities and quality results alongside the report. Workload ID strings and attested timestamps do not prove execution conditions or completeness.

## Single-operator roofline

```bash
python3 scripts/plan_inference_roofline.py --help
python3 scripts/plan_inference_roofline.py \
  --config configs/planning/inference_roofline_synthetic.json \
  --output outputs/inference_perf/local-smoke/roofline.json \
  --allow-synthetic
```

Parent directory must exist and output must be new. This illustrative fixture contains no actual B300 specifications or measurements.

Input fields:
- `scope`: `single_operator`.
- `operator`: nonempty label.
- `flops`, `dram_bytes`: finite positive work/traffic estimates.
- `provenance`: exact `kind` (`synthetic`, `source`, `measurement`) and nonempty `description` identifying conditions/source. This attests all inputs; it does not authenticate them.
- Optional/null `effective_compute_flops_per_second`, `effective_bandwidth_bytes_per_second`, `observed_seconds`: finite positive values when supplied.

Output includes intensity, individual time bounds, their max, a model-based compute/bandwidth/balanced classification, warnings/limitations and exact input-byte SHA256. Both ceilings are required for a combined bound; otherwise status is `measurement_required`. Balanced means bounds within 5% of the larger bound.

Observation/headroom values are model-relative, not hardware counter utilization or guaranteed speedup. Observed time below any available bound flags inconsistent assumptions and suppresses headroom inference. Logical tensor bytes are not measured DRAM traffic; account for cache reuse. Compute/memory ideal overlap, contention, shape/precision and excluded launch/sync/communication costs limit this model.

## Verification

```bash
.venv/bin/python -m pytest -q tests/test_inference_perf.py tests/test_inference_roofline.py
```

These tests and synthetic CLI runs verify local analysis contracts only. Real engine/load instrumentation, service latency, GPU memory, task quality, Nsight evidence, regression comparison and upstream optimization remain separate pending work.
