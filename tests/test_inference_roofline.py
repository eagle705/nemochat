"""CPU arithmetic and CLI fixtures, not hardware performance measurements."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from nemochat.planning.inference_roofline import calculate_roofline, report_from_json

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/plan_inference_roofline.py"
FIXTURE = ROOT / "configs/planning/inference_roofline_synthetic.json"


@pytest.fixture
def inputs():
    return {
        "scope": "single_operator", "operator": "test_operator",
        "flops": 1000, "dram_bytes": 100,
        "effective_compute_flops_per_second": 500,
        "effective_bandwidth_bytes_per_second": 100,
        "observed_seconds": 4,
        "provenance": {"kind": "synthetic", "description": "Handwritten arithmetic fixture, not measured."},
    }


def test_arithmetic(inputs):
    report = calculate_roofline(inputs)
    assert report["arithmetic_intensity_flops_per_byte"] == 10
    assert report["compute_lower_bound_seconds"] == 2
    assert report["memory_lower_bound_seconds"] == 1
    assert report["modeled_lower_bound_seconds"] == 2
    assert report["classification_model_estimate"] == "compute"
    assert report["status"] == "model_estimate"
    assert report["observation"] == {
        "observed_seconds": 4, "compute_utilization_fraction": 0.5,
        "bandwidth_utilization_fraction": 0.25, "bound_utilization_fraction": 0.5,
        "modeled_headroom_fraction": 0.5, "observed_to_bound_ratio": 2,
        "status": "model_relative_estimate",
    }
    assert any("Synthetic" in warning for warning in report["warnings"])
    assert all(any(word in caveat for caveat in report["limitations"])
               for word in ("Concurrency", "overlap", "Cache", "speedup", "Single operator"))
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("bandwidth,classification", [(25, "bandwidth"), (50, "balanced"), (51, "balanced"), (55, "compute")])
def test_classification(inputs, bandwidth, classification):
    inputs["effective_bandwidth_bytes_per_second"] = bandwidth
    assert calculate_roofline(inputs)["classification_model_estimate"] == classification


def test_no_observation_and_provenance_copy(inputs):
    inputs.pop("observed_seconds")
    inputs["provenance"] = {"kind": "source", "description": "Caller supplied source reference with precision/shape conditions."}
    report = calculate_roofline(inputs)
    assert report["observation"] is None
    assert report["provenance"] == inputs["provenance"]
    report["provenance"]["kind"] = "measurement"
    assert inputs["provenance"]["kind"] == "source"


@pytest.mark.parametrize("observed", [0.5, 1.999999999])
def test_inconsistent_observation(inputs, observed):
    inputs["observed_seconds"] = observed
    report = calculate_roofline(inputs)
    assert report["observation"]["status"] == "assumptions_inconsistent"
    assert report["observation"]["modeled_headroom_fraction"] is None
    assert report["observation"]["observed_to_bound_ratio"] is None
    assert report["observation"]["compute_utilization_fraction"] > 1
    assert any("ceiling/traffic assumptions are inconsistent" in w for w in report["warnings"])


def test_at_bound(inputs):
    inputs["observed_seconds"] = 2
    report = calculate_roofline(inputs)
    assert report["observation"]["modeled_headroom_fraction"] == 0
    assert report["observation"]["observed_to_bound_ratio"] == 1


@pytest.mark.parametrize("missing", [("effective_compute_flops_per_second",), ("effective_bandwidth_bytes_per_second",), ("effective_compute_flops_per_second", "effective_bandwidth_bytes_per_second")])
@pytest.mark.parametrize("omit", [True, False])
def test_measurement_required(inputs, missing, omit):
    for key in missing:
        if omit:
            inputs.pop(key)
        else:
            inputs[key] = None
    report = calculate_roofline(inputs)
    assert report["status"] == report["classification_model_estimate"] == "measurement_required"
    assert report["modeled_lower_bound_seconds"] is None
    assert report["observation"]["status"] == "measurement_required"
    assert report["observation"]["modeled_headroom_fraction"] is None
    assert report["observation"]["bound_utilization_fraction"] is None


def test_partial_bound_inconsistent(inputs):
    inputs["effective_bandwidth_bytes_per_second"] = None
    inputs["observed_seconds"] = 1
    report = calculate_roofline(inputs)
    assert report["status"] == "measurement_required"
    assert report["observation"]["status"] == "assumptions_inconsistent"


@pytest.mark.parametrize("key", ["flops", "dram_bytes", "effective_compute_flops_per_second", "effective_bandwidth_bytes_per_second", "observed_seconds"])
@pytest.mark.parametrize("value", [True, False, 0, -1, float("nan"), float("inf"), -float("inf"), "100", [], {}])
def test_bad_numbers(inputs, key, value):
    inputs[key] = value
    with pytest.raises(ValueError, match="finite positive"):
        calculate_roofline(inputs)


@pytest.mark.parametrize("mutate", [
    lambda x: x.update(scope="whole_request"),
    lambda x: x.update(operator=" "),
    lambda x: x.update(operator=True),
    lambda x: x.update(unexpected=1),
    lambda x: x.pop("flops"),
    lambda x: x.update(flops=None),
    lambda x: x.update(provenance={}),
    lambda x: x.update(provenance={"kind": "measured", "description": "fixture"}),
    lambda x: x.update(provenance={"kind": "source", "description": " "}),
    lambda x: x.update(provenance={"kind": "source", "description": True}),
    lambda x: x.update(provenance={"kind": "source", "description": "fixture", "extra": True}),
])
def test_bad_schema(inputs, mutate):
    mutate(inputs)
    with pytest.raises(ValueError):
        calculate_roofline(inputs)


@pytest.mark.parametrize("value", [None, [], True, "input"])
def test_nonobject(value):
    with pytest.raises(ValueError, match="JSON object"):
        calculate_roofline(value)


@pytest.mark.parametrize("flops,dram", [(1e308, 1e-308), (1e-308, 1e308), (10**1000, 1)])
def test_unrepresentable_arithmetic(inputs, flops, dram):
    inputs.update(flops=flops, dram_bytes=dram)
    with pytest.raises(ValueError, match="finite positive"):
        calculate_roofline(inputs)


@pytest.mark.parametrize("raw", [b'{"x":NaN}', b'{"x":Infinity}', b'{"x":-Infinity}', b'{"x":1,"x":2}', b'{"x":{"a":1,"a":2}}', b'{} trailing', b'\xff', b'[]'])
def test_strict_json(raw):
    with pytest.raises(ValueError):
        report_from_json(raw, allow_synthetic=True)


def test_exact_hash_and_optin(inputs):
    raw = (json.dumps(inputs, indent=2) + "\n").encode()
    with pytest.raises(ValueError, match="--allow-synthetic"):
        report_from_json(raw)
    assert report_from_json(raw, allow_synthetic=True)["input_sha256"] == hashlib.sha256(raw).hexdigest()
    assert report_from_json(raw + b" ", allow_synthetic=True)["input_sha256"] != hashlib.sha256(raw).hexdigest()


def cli(config, output, *extra):
    return subprocess.run([sys.executable, str(SCRIPT), "--config", str(config), "--output", str(output), *extra], capture_output=True, text=True, cwd=ROOT)


def test_cli_success_and_no_overwrite(tmp_path):
    output = tmp_path / "report.json"
    run = cli(FIXTURE, output, "--allow-synthetic")
    assert run.returncode == 0, run.stderr
    raw_report = output.read_bytes()
    report = json.loads(raw_report)
    assert report["input_sha256"] == hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
    assert report["classification_model_estimate"] == "bandwidth"
    assert report["modeled_lower_bound_seconds"] == pytest.approx(0.003)
    assert report["observation"]["modeled_headroom_fraction"] == pytest.approx(0.5)
    assert cli(FIXTURE, output, "--allow-synthetic").returncode == 2
    assert output.read_bytes() == raw_report


def test_cli_rejects_synthetic_without_output(tmp_path):
    output = tmp_path / "report.json"
    run = cli(FIXTURE, output)
    assert run.returncode == 2
    assert "--allow-synthetic" in run.stderr
    assert not output.exists()


@pytest.mark.parametrize("raw", ['{"x": NaN}', '{"scope":"single_operator","scope":"single_operator"}', '{', '[]'])
def test_cli_malformed(tmp_path, raw):
    config = tmp_path / "input.json"
    config.write_text(raw)
    output = tmp_path / "report.json"
    run = cli(config, output, "--allow-synthetic")
    assert run.returncode == 2 and "error:" in run.stderr
    assert not output.exists()


def test_cli_missing_input(tmp_path):
    output = tmp_path / "report.json"
    assert cli(tmp_path / "missing.json", output).returncode == 2
    assert not output.exists()


def test_cli_measured_unknown_and_input_alias(tmp_path, inputs):
    inputs["provenance"] = {"kind": "measurement", "description": "Caller-provided measurement reference, not independently verified by this test."}
    inputs["effective_compute_flops_per_second"] = None
    config = tmp_path / "input.json"
    config.write_text(json.dumps(inputs))
    original = config.read_bytes()
    output = tmp_path / "report.json"
    assert cli(config, output).returncode == 0
    assert json.loads(output.read_text())["status"] == "measurement_required"
    assert cli(config, config).returncode == 2
    assert config.read_bytes() == original


def test_cli_existing_symlink(tmp_path):
    target = tmp_path / "target.json"
    target.write_text("preserve")
    output = tmp_path / "report.json"
    output.symlink_to(target)
    assert cli(FIXTURE, output, "--allow-synthetic").returncode == 2
    assert target.read_text() == "preserve"
