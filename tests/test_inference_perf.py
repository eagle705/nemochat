"""Handwritten synthetic mechanics tests: not real inference benchmark evidence."""

import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from nemochat.inference_perf.analysis import (
    analyze, load_config, load_records, summarize, validate_config, validate_record,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "configs" / "inference_perf"


@pytest.fixture
def config():
    return load_config(FIXTURES / "synthetic_config.json")


@pytest.fixture
def records():
    return load_records(FIXTURES / "synthetic_requests.jsonl", allow_synthetic=True)


def test_report_counts_metrics_and_slo(records, config):
    report = analyze(records, config, allow_synthetic=True)
    assert report["counts"] == dict(total=7, success=4, error=1, timeout=1, cancelled=1)
    assert report["synthetic"] is True
    assert report["slo"]["met"] == 1
    assert report["slo"]["violated"] == 1
    assert report["slo"]["unavailable"] == 2
    assert report["slo_goodput_requests_per_s"] == pytest.approx(0.1)
    assert report["offered_requests_per_s"] == pytest.approx(0.7)
    assert report["actual_tokens_all_attempts"] == dict(input_tokens=56, output_tokens=8)
    assert report["actual_tokens_success"]["output_tokens"] == 7
    assert report["latencies"]["client_ttft_s"]["sample_count"] == 3
    assert report["latencies"]["client_ttft_s"]["p99"] == 1.5
    assert report["latencies"]["client_e2e_s"]["sample_count"] == 4
    assert report["latencies"]["observed_token_gap_s"]["sample_count"] == 3
    assert report["latencies"]["observed_token_gap_s"]["p99"] == 1
    assert report["success_missing_token_timing"] == 1
    assert report["success_without_observable_gap"] == 2


def test_nearest_rank_and_empty():
    assert summarize([3, 1, 2, 4])["p50"] == 2
    assert summarize([3, 1, 2, 4])["p99"] == 4
    assert summarize(range(1, 101))["p99"] == 99
    assert summarize(range(100))["warnings"] == []
    empty = summarize([])
    assert empty["sample_count"] == 0
    assert empty["p50"] is None and empty["p95"] is None and empty["p99"] is None
    assert any("insufficient p99" in warning for warning in empty["warnings"])


@pytest.mark.parametrize("change", [
    {"schema_version": True}, {"schema_version": 2}, {"synthetic": "yes"},
    {"clock": "wall_seconds"}, {"status": "pending"},
    {"input_tokens": -1}, {"output_tokens": True}, {"input_tokens": 1.0},
    {"arrival_s": float("nan")}, {"send_s": float("inf")},
    {"completion_s": -1}, {"send_s": True}, {"send_s": 1},
    {"arrival_s": 0.2}, {"first_token_s": None}, {"first_token_s": 0.05},
    {"first_token_s": 0.9}, {"token_timing": "chunks"},
    {"token_arrivals_s": [0.3, 0.5]},
    {"token_arrivals_s": [0.3, 0.2, 0.7]},
    {"token_arrivals_s": [0.4, 0.5, 0.7]},
    {"token_arrivals_s": [0.3, 0.5, 0.9]},
    {"token_arrivals_s": [0.3, True, 0.7]},
    {"token_arrivals_s": [0.3, float("nan"), 0.7]},
    {"token_timing": "unavailable"}, {"token_arrivals_s": None},
    {"output_tokens": 0}, {"request_id": " "}, {"unexpected": 1},
    {"arrival_s": 10**400},
])
def test_invalid_records(records, change):
    record = {**records[0], **change}
    with pytest.raises(ValueError):
        validate_record(record, allow_synthetic=True)


def test_missing_field_and_synthetic_guard(records):
    record = copy.deepcopy(records[0])
    del record["first_token_s"]
    with pytest.raises(ValueError):
        validate_record(record, allow_synthetic=True)
    with pytest.raises(ValueError, match="allow-synthetic"):
        validate_record(records[0])


@pytest.mark.parametrize("text", ["", "\n", "{bad}\n", "[]\n", "null\n", '{"a":1,"a":2}\n', '{"x":NaN}\n', '{"x":Infinity}\n'])
def test_invalid_jsonl(tmp_path, text):
    path = tmp_path / "requests.jsonl"
    path.write_text(text)
    with pytest.raises(ValueError):
        load_records(path, allow_synthetic=True)


def test_duplicate_ids_and_json_keys(records, config, tmp_path):
    with pytest.raises(ValueError, match="duplicate request_id"):
        analyze([records[0], records[0]], config, allow_synthetic=True)
    path = tmp_path / "requests.jsonl"
    path.write_text(json.dumps(records[0]) + "\n" + json.dumps(records[0]) + "\n")
    with pytest.raises(ValueError, match="line 2.*duplicate request_id"):
        load_records(path, allow_synthetic=True)
    text = json.dumps(records[0]).replace('"send_s": 0.1', '"send_s": 0.1, "send_s": 0.1')
    path.write_text(text)
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_records(path, allow_synthetic=True)


@pytest.mark.parametrize("change", [
    {"measurement_end_s": 0}, {"measurement_start_s": -1},
    {"measurement_end_s": float("inf")}, {"schema_version": True},
    {"workload_id": ""}, {"unexpected": 1},
    {"slo": {"ttft_s": None, "e2e_s": None, "max_token_gap_s": None}},
    {"slo": {"ttft_s": -1, "e2e_s": None, "max_token_gap_s": None}},
    {"slo": {"ttft_s": True, "e2e_s": None, "max_token_gap_s": None}},
    {"slo": {"ttft_ms": 1}},
])
def test_invalid_config(config, change):
    with pytest.raises(ValueError):
        validate_config({**config, **change})


def test_window_and_mixed_evidence(records, config):
    with pytest.raises(ValueError, match="outside"):
        analyze(records, {**config, "measurement_end_s": 8}, allow_synthetic=True)
    with pytest.raises(ValueError, match="outside"):
        analyze(records, {**config, "measurement_start_s": 1}, allow_synthetic=True)
    mixed = copy.deepcopy(records)
    mixed[0]["synthetic"] = False
    with pytest.raises(ValueError, match="cannot mix"):
        analyze(mixed, config, allow_synthetic=True)
    with pytest.raises(ValueError):
        analyze([], config)


def test_zero_output_single_token_and_missing_timing(records, config):
    for record in (records[2], records[3]):
        result = analyze([record], config, allow_synthetic=True)
        assert result["latencies"]["observed_token_gap_s"]["p99"] is None
        assert result["slo"]["unavailable"] == 1
    result = analyze([records[3]], config, allow_synthetic=True)
    assert result["latencies"]["client_ttft_s"]["p99"] is None
    one = copy.deepcopy(records[0])
    one.update(output_tokens=1, token_arrivals_s=[0.3])
    assert analyze([one], config, allow_synthetic=True)["slo"]["unavailable"] == 1
    only_e2e = copy.deepcopy(config)
    only_e2e["slo"] = dict(ttft_s=None, e2e_s=0.5, max_token_gap_s=None)
    # Inclusive threshold, no implicit requirement for output or gap samples.
    assert analyze([records[3]], only_e2e, allow_synthetic=True)["slo"]["met"] == 1


def test_failed_requests_excluded_from_success_latencies(records, config):
    result = analyze(records[4:], config, allow_synthetic=True)
    assert result["slo_goodput_requests_per_s"] == 0
    assert all(value["sample_count"] == 0 for value in result["latencies"].values())


def test_individually_identified_simultaneous_tokens(records, config):
    record = copy.deepcopy(records[0])
    record["token_arrivals_s"] = [0.3, 0.3, 0.3]
    result = analyze([record], config, allow_synthetic=True)
    assert result["latencies"]["observed_token_gap_s"]["p99"] == 0


def test_finite_numeric_overflow(records, config):
    record = copy.deepcopy(records[3])
    record.update(arrival_s=0, send_s=0, completion_s=0)
    config = {**config, "measurement_end_s": 5e-324}
    config["slo"] = dict(ttft_s=None, e2e_s=1, max_token_gap_s=None)
    with pytest.raises(ValueError, match="finite numeric range"):
        analyze([record], config, allow_synthetic=True)


def run_cli(output, *extra, requests=None):
    return subprocess.run([
        sys.executable, str(ROOT / "scripts/analyze_inference_requests.py"),
        "--config", str(FIXTURES / "synthetic_config.json"),
        "--requests", str(requests or FIXTURES / "synthetic_requests.jsonl"),
        "--output", str(output), *extra,
    ], capture_output=True, text=True, cwd=ROOT)


def test_cli_success_guard_overwrite_and_malformed(tmp_path):
    output = tmp_path / "report.json"
    guard = run_cli(output)
    assert guard.returncode == 2 and "allow-synthetic" in guard.stderr
    assert not output.exists()
    success = run_cli(output, "--allow-synthetic")
    assert success.returncode == 0, success.stderr
    original = output.read_bytes()
    report = json.loads(original)
    assert report["synthetic"] is True and report["slo"]["met"] == 1
    assert "synthetic-fixture-tested" in success.stdout
    overwrite = run_cli(output, "--allow-synthetic")
    assert overwrite.returncode == 2 and "already exists" in overwrite.stderr
    assert output.read_bytes() == original
    bad = tmp_path / "bad.jsonl"
    bad.write_text("{bad}\n")
    failure = run_cli(tmp_path / "new.json", "--allow-synthetic", requests=bad)
    assert failure.returncode == 2 and "line 1" in failure.stderr
    assert not (tmp_path / "new.json").exists()


def test_cli_dangling_symlink_not_overwritten(tmp_path):
    output = tmp_path / "report.json"
    target = tmp_path / "absent.json"
    output.symlink_to(target)
    result = run_cli(output, "--allow-synthetic")
    assert result.returncode == 2
    assert output.is_symlink() and not target.exists()


def test_cli_schema_help():
    result = subprocess.run([sys.executable, str(ROOT / "scripts/analyze_inference_requests.py"), "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "Streaming" in result.stdout and "nearest rank" in result.stdout
    assert "client_monotonic_seconds" in result.stdout
