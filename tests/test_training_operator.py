import json
import math
import pytest
from nemochat.planning.training_operator import estimate_linear, report_from_json


def config():
    return {"schema_version": 1, "batch_tokens": 2, "in_features": 3,
            "out_features": 4, "dtype": "float32",
            "provenance": {"kind": "source", "description": "declared fixture dimensions"}}


def test_forward_backward_counts_not_measured_roofline():
    result = estimate_linear(config())
    assert result["total_gemm_flops"] == 144
    assert [o["gemm_mnk"] for o in result["operators"]] == [[2, 4, 3], [2, 3, 4], [3, 4, 2]]
    assert all(o["logical_operand_bytes"] == 104 for o in result["operators"])
    assert all(o["roofline"] is None for o in result["operators"])


@pytest.mark.parametrize("key,value", [("batch_tokens", True), ("schema_version", True),
    ("in_features", 0), ("out_features", 1000001), ("dtype", "fp8"),
    ("effective_compute_flops_per_second", math.inf), ("measurements", {"bad": {}})])
def test_invalid(key, value):
    c = config(); c[key] = value
    with pytest.raises(ValueError):
        estimate_linear(c)


def test_actual_traffic_required_even_with_ceilings():
    c = config(); c.update(effective_compute_flops_per_second=1000, effective_bandwidth_bytes_per_second=1000)
    assert estimate_linear(c)["operators"][0]["roofline"] is None
    c["measurements"] = {"forward": {"dram_bytes": 200, "observed_seconds": 0.3,
                                    "description": "handwritten numeric fixture; not GPU evidence"}}
    c["provenance"]["kind"] = "synthetic"
    op = report_from_json(json.dumps(c).encode(), allow_synthetic=True)["operators"][0]
    assert op["roofline"]["inputs"]["dram_bytes"] == 200
    assert op["roofline"]["classification_model_estimate"] == "bandwidth"
    c["measurements"]["forward"]["observed_seconds"] = 0.01
    assert estimate_linear(c)["operators"][0]["status"] == "assumptions_inconsistent"


def test_missing_ceiling_and_synthetic_opt_in():
    c = config(); c["provenance"]["kind"] = "synthetic"
    c["measurements"] = {"forward": {"dram_bytes": 200, "observed_seconds": 0.3, "description": "fixture"}}
    with pytest.raises(ValueError, match="allow-synthetic"):
        report_from_json(json.dumps(c).encode())
    r = report_from_json(json.dumps(c).encode(), allow_synthetic=True)
    assert r["operators"][0]["status"] == "measurement_required"
    assert len(r["input_sha256"]) == 64


@pytest.mark.parametrize("raw", [b'{"schema_version":1,"schema_version":1}', b'{"a":NaN}', b'{}' * 40000])
def test_strict_json(raw):
    with pytest.raises(ValueError):
        report_from_json(raw)


def test_unknown_and_incomplete_measurements():
    c = config(); c["unexpected"] = 1
    with pytest.raises(ValueError): estimate_linear(c)
    del c["unexpected"]
    c["measurements"] = {"forward": {"dram_bytes": 1}}
    with pytest.raises(ValueError): estimate_linear(c)
