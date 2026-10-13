"""Tutorial-only tests; all four-arm training mechanics run without Docker."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from benchmark import equivalent, verify_file
from train import clean_environment, loopback_endpoint
from verify_jaeger import get_json, validate_trace


@pytest.mark.parametrize("endpoint", ["https://127.0.0.1:14318/v1/traces", "http://example.com/v1/traces",
    "http://user:pass@127.0.0.1:14318/v1/traces", "http://127.0.0.1:14318/v1/traces?key=secret"])
def test_reject_nonlocal_export(endpoint):
    with pytest.raises(ValueError):
        loopback_endpoint(endpoint)


def test_environment_scrub(monkeypatch):
    for key in ("OTEL_EXPORTER_OTLP_HEADERS", "WANDB_API_KEY", "NEMO_LENS_ENABLED", "HTTP_PROXY"):
        monkeypatch.setenv(key, "dummy")
    clean_environment()
    assert all(key not in os.environ for key in ("OTEL_EXPORTER_OTLP_HEADERS", "WANDB_API_KEY", "NEMO_LENS_ENABLED", "HTTP_PROXY"))


def test_api_reject_remote():
    with pytest.raises(ValueError):
        get_json("http://example.com", "/api/services")


def test_actual_cpu_modes_equivalent(tmp_path):
    rows = []
    for mode in ("baseline", "disabled", "file"):
        output = tmp_path / mode
        subprocess.run([sys.executable, str(Path(__file__).with_name("train.py")), "--mode", mode,
                        "--steps", "4", "--warmup", "2", "--output", str(output)], check=True, timeout=60)
        row = json.loads((output / "result.json").read_text())
        rows.append(row)
        if mode == "file":
            verify_file(output / "spans.jsonl", row)
        else:
            assert not (output / "spans.jsonl").exists()
    equivalent(rows)
    assert rows[0]["losses"][0] > rows[0]["losses"][-1]
    rows[-1]["parameter_sha256"] = "incorrect"
    with pytest.raises(ValueError):
        equivalent(rows)


def test_refuse_output_overwrite(tmp_path):
    result = subprocess.run([sys.executable, str(Path(__file__).with_name("train.py")), "--mode", "baseline",
                             "--output", str(tmp_path)], capture_output=True, timeout=30)
    assert result.returncode != 0


def test_reject_incomplete_trace():
    with pytest.raises(ValueError):
        validate_trace({"spans": []}, {"expected_spans": 5})
