"""Strict offline inference timing analysis (schema_version=1).

Each JSONL line (blank lines are invalid) must have EXACTLY these fields:
  schema_version: integer 1
  request_id: unique nonempty string
  synthetic: boolean (true requires explicit allow_synthetic)
  clock: "client_monotonic_seconds"
  status: "success", "error", "timeout", or "cancelled"
  arrival_s, send_s, completion_s: finite nonnegative seconds in the same
    client monotonic clock; arrival <= send <= completion
  first_token_s: finite seconds or null; send <= first_token <= completion
  input_tokens, output_tokens: actual nonnegative integer token counts, not
    requested budgets, character counts, or estimates (including failed requests)
  token_timing: "individual_tokens" or "unavailable"
  token_arrivals_s: list of one timestamp per ACTUAL output token, in observed
    nondecreasing order, or null when unavailable. With individual_tokens the
    list length must equal output_tokens and its first entry equals first_token_s.
    Equal times are permitted only for individually identified tokens. Streaming
    chunk timestamps are NOT token timings: do not repeat chunk times to fill a
    token list. This is a producer-attested contract, not independently verified.
    Zero output requires first_token_s=null; individual token timing then uses [].
    Positive output requires first_token_s even if full token timing is unavailable.

Config is an exact object: schema_version=1, workload_id/tokenizer_id/decoding_id
(nonempty strings identifying frozen protocols), measurement_start_s and
measurement_end_s (explicit finite seconds, end > start), slo (exact fields
 ttft_s, e2e_s, max_token_gap_s: nonnegative seconds or null; at least one set).
All request arrivals must be in [start,end); all completions must be <= end.
No filtering, censoring, warmup subtraction, or inferred measurement duration.
Use one client clock domain and capture every admitted attempt; missing in-flight
attempts cannot be detected from an imported file. Report validity relies on this.

Client TTFT = first_token - send; E2E = completion - send. Arrival-to-completion
and arrival-to-send are separate queue-inclusive and dispatch-delay metrics.
Latency percentiles pool SUCCESSFUL requests only. Adjacent observed individual
output-token gaps pool success tokens, NOT streaming chunks or server decode time.
A missing sample is unavailable (null), never zero. SLO uses success, inclusive
thresholds, and per-request maximum gap; gap SLO is unknown with <2 output tokens
or missing token timing. Unknown success requests never count as goodput.
Goodput = successful requests satisfying ALL configured SLOs / window seconds.
Percentiles use nearest rank ceil(p*n); small samples do not establish tails.
No model quality, GPU throughput improvement, or hardware headroom is inferred.
"""

from collections import Counter
import json
import math
from pathlib import Path


RECORD_FIELDS = {
    "schema_version", "request_id", "synthetic", "clock", "status", "arrival_s",
    "send_s", "first_token_s", "completion_s", "input_tokens", "output_tokens",
    "token_timing", "token_arrivals_s",
}
CONFIG_FIELDS = {
    "schema_version", "workload_id", "tokenizer_id", "decoding_id",
    "measurement_start_s", "measurement_end_s", "slo",
}
SLO_FIELDS = {"ttft_s", "e2e_s", "max_token_gap_s"}
STATUSES = ("success", "error", "timeout", "cancelled")


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _constant(value):
    raise ValueError(f"nonfinite JSON constant: {value}")


def strict_json(text):
    """Reject duplicate keys and NaN/Infinity (including nested objects)."""
    return json.loads(text, object_pairs_hook=_object, parse_constant=_constant)


def _fields(value, expected, name):
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{name} must have exactly fields {sorted(expected)}")


def _seconds(value, name):
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be finite nonnegative seconds (not a boolean)")
    try:
        valid = math.isfinite(value) and value >= 0
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f"{name} must be finite nonnegative seconds")
    return value


def _version(value):
    if type(value) is not int or value != 1:
        raise ValueError("schema_version must be integer 1")


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def validate_record(record, *, allow_synthetic=False):
    _fields(record, RECORD_FIELDS, "request record")
    _version(record["schema_version"])
    _text(record["request_id"], "request_id")
    if type(record["synthetic"]) is not bool:
        raise ValueError("synthetic must be a boolean")
    if record["synthetic"] and not allow_synthetic:
        raise ValueError("synthetic records require --allow-synthetic")
    if record["clock"] != "client_monotonic_seconds":
        raise ValueError("clock must be client_monotonic_seconds")
    if record["status"] not in STATUSES:
        raise ValueError(f"status must be one of {STATUSES}")
    for name in ("input_tokens", "output_tokens"):
        if type(record[name]) is not int or record[name] < 0:
            raise ValueError(f"{name} must be an actual nonnegative integer count")
    arrival, send, completion = (
        _seconds(record[name], name) for name in ("arrival_s", "send_s", "completion_s")
    )
    if not arrival <= send <= completion:
        raise ValueError("require arrival_s <= send_s <= completion_s")
    first = record["first_token_s"]
    if record["output_tokens"] == 0:
        if first is not None:
            raise ValueError("zero output requires first_token_s=null")
    else:
        _seconds(first, "first_token_s")
        if not send <= first <= completion:
            raise ValueError("require send_s <= first_token_s <= completion_s")
    times = record["token_arrivals_s"]
    if record["token_timing"] == "unavailable":
        if times is not None:
            raise ValueError("unavailable token timing requires token_arrivals_s=null")
    elif record["token_timing"] == "individual_tokens":
        if not isinstance(times, list) or len(times) != record["output_tokens"]:
            raise ValueError("token_arrivals_s length must equal actual output_tokens")
        previous = send
        for timestamp in times:
            _seconds(timestamp, "token_arrivals_s entry")
            if not previous <= timestamp <= completion:
                raise ValueError("token timestamps must be nondecreasing within send/completion")
            previous = timestamp
        if times and times[0] != first:
            raise ValueError("first token timestamp must equal first_token_s")
    else:
        raise ValueError("token_timing must be individual_tokens or unavailable; chunks are not tokens")
    return record


def load_records(path, *, allow_synthetic=False):
    """Read all records strictly; reject blank lines, duplicate IDs, and empty files."""
    records, seen = [], set()
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            try:
                record = validate_record(strict_json(line), allow_synthetic=allow_synthetic)
                if record["request_id"] in seen:
                    raise ValueError(f"duplicate request_id: {record['request_id']}")
                seen.add(record["request_id"])
                records.append(record)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{path}: line {number}: {exc}") from exc
    if not records:
        raise ValueError("request file is empty")
    return records


def validate_config(config):
    _fields(config, CONFIG_FIELDS, "config")
    _version(config["schema_version"])
    for name in ("workload_id", "tokenizer_id", "decoding_id"):
        _text(config[name], name)
    start = _seconds(config["measurement_start_s"], "measurement_start_s")
    end = _seconds(config["measurement_end_s"], "measurement_end_s")
    if end <= start:
        raise ValueError("measurement_end_s must exceed measurement_start_s")
    _fields(config["slo"], SLO_FIELDS, "slo")
    for name, threshold in config["slo"].items():
        if threshold is not None:
            _seconds(threshold, name)
    if all(value is None for value in config["slo"].values()):
        raise ValueError("at least one SLO threshold is required")
    return config


def load_config(path):
    return validate_config(strict_json(Path(path).read_text(encoding="utf-8")))


def summarize(samples):
    """Nearest-rank seconds distribution; no samples means null statistics."""
    values = sorted(_seconds(value, "sample") for value in samples)
    count = len(values)
    result = {"sample_count": count, "unit": "seconds", "method": "nearest_rank"}
    for name, quantile in (("p50", 0.50), ("p95", 0.95), ("p99", 0.99)):
        result[name] = values[math.ceil(quantile * count) - 1] if count else None
    result["warnings"] = []
    if count == 0:
        result["warnings"].append("unavailable: no observed samples")
    if count < 100:
        result["warnings"].append("insufficient p99 evidence: fewer than 100 samples; no tail-confidence claim")
    return result


def observed_gaps(record):
    times = record["token_arrivals_s"]
    if times is None or len(times) < 2:
        return None
    return [later - earlier for earlier, later in zip(times, times[1:])]


def _request_metrics(record):
    first = record["first_token_s"]
    gaps = observed_gaps(record)
    return {
        "ttft_s": None if first is None else first - record["send_s"],
        "e2e_s": record["completion_s"] - record["send_s"],
        "max_token_gap_s": None if gaps is None else max(gaps),
    }


def analyze(records, config, *, allow_synthetic=False):
    """Validate even direct API inputs; report successful-client latency and goodput."""
    validate_config(config)
    if not records:
        raise ValueError("at least one request is required")
    seen = set()
    start, end = config["measurement_start_s"], config["measurement_end_s"]
    for record in records:
        validate_record(record, allow_synthetic=allow_synthetic)
        if record["request_id"] in seen:
            raise ValueError(f"duplicate request_id: {record['request_id']}")
        seen.add(record["request_id"])
        if not start <= record["arrival_s"] < end or record["completion_s"] > end:
            raise ValueError(f"request {record['request_id']} is outside explicit complete measurement window")
    labels = {record["synthetic"] for record in records}
    if len(labels) != 1:
        raise ValueError("cannot mix synthetic and nonsynthetic records")
    success = [record for record in records if record["status"] == "success"]
    samples = {name: [] for name in (
        "client_ttft_s", "client_e2e_s", "arrival_to_completion_s",
        "dispatch_delay_s", "observed_token_gap_s",
    )}
    met, violated, unknown = [], [], []
    missing_token_timing = no_gap = 0
    for record in success:
        metrics = _request_metrics(record)
        if metrics["ttft_s"] is not None:
            samples["client_ttft_s"].append(metrics["ttft_s"])
        samples["client_e2e_s"].append(metrics["e2e_s"])
        samples["arrival_to_completion_s"].append(record["completion_s"] - record["arrival_s"])
        samples["dispatch_delay_s"].append(record["send_s"] - record["arrival_s"])
        gaps = observed_gaps(record)
        missing_token_timing += record["token_arrivals_s"] is None
        no_gap += gaps is None
        if gaps is not None:
            samples["observed_token_gap_s"].extend(gaps)
        checks = [(metrics[name], threshold) for name, threshold in config["slo"].items()
                  if threshold is not None]
        if any(value is not None and value > threshold for value, threshold in checks):
            violated.append(record["request_id"])
        elif any(value is None for value, _ in checks):
            unknown.append(record["request_id"])
        else:
            met.append(record["request_id"])
    counts = Counter(record["status"] for record in records)
    duration = end - start
    report = {
        "schema_version": 1,
        "evidence": "synthetic-fixture-tested" if True in labels else "imported-client-records-not-independently-verified",
        "synthetic": True in labels,
        "config": config,
        "measurement_window_s": duration,
        "counts": {"total": len(records), **{status: counts[status] for status in STATUSES}},
        "actual_tokens_all_attempts": {name: sum(record[name] for record in records)
                                       for name in ("input_tokens", "output_tokens")},
        "actual_tokens_success": {name: sum(record[name] for record in success)
                                  for name in ("input_tokens", "output_tokens")},
        "latency_population": "successful requests only; observed_token_gap_s pools adjacent individual tokens",
        "latencies": {name: summarize(values) for name, values in samples.items()},
        "success_missing_token_timing": missing_token_timing,
        "success_without_observable_gap": no_gap,
        "slo": {"met": len(met), "violated": len(violated), "unavailable": len(unknown),
                "met_request_ids": met, "violated_request_ids": violated,
                "unavailable_request_ids": unknown},
        "slo_goodput_requests_per_s": len(met) / duration,
        "offered_requests_per_s": len(records) / duration,
        "warnings": ["Imported producer-attested data; no quality or GPU speedup evidence.",
                     "Pooled token gaps are token-weighted and correlated within requests; not independent request samples.",
                     "Nearest-rank percentiles have no confidence interval; >=100 samples alone does not establish p99 reliability.",
                     "All admitted attempts must be recorded to avoid survivor bias; analyzer cannot detect omitted attempts."],
    }
    # Catch finite input combinations that nevertheless overflow rates.
    try:
        json.dumps(report, allow_nan=False)
    except (ValueError, OverflowError) as exc:
        raise ValueError("derived report values exceed finite numeric range") from exc
    return report
