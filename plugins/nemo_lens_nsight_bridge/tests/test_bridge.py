import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
import yaml

from lens_nsight_bridge import Bridge
from lens_nsight_bridge.control import activation, run
from lens_nsight_bridge.source import extract_verified
from lens_nsight_bridge import cli

ROOT = Path(__file__).resolve().parents[1]


def clean_environment():
    return {k: v for k, v in os.environ.items() if not k.startswith(("NEMO_LENS", "OTEL_"))}


def test_cpu_control_no_nvtx(tmp_path):
    out = tmp_path / "cpu"
    with Bridge(out, ROOT) as bridge:
        result = run(bridge)
    assert result["forward_reference"] == "passed"
    assert result["gradient_reference"] == "passed"
    records = [json.loads(line) for line in (out / "operations.jsonl").read_text().splitlines()]
    assert len(records) == 21
    assert len({r["bridge_id"] for r in records}) == 21
    assert {r["role"] for r in records} == {"routing", "topk", "permutation", "dispatch",
        "expert_up", "activation", "expert_down", "combine", "backward"}
    for record in records:
        assert not record["nvtx_emitted"]
        assert record["status"] == "ok"
        assert record["bridge_id"] in record["nvtx_label"]
        assert extract_verified(out, record["source"], ROOT).startswith("def ")


def test_import_is_side_effect_free_without_optional_backends():
    result = subprocess.run([sys.executable, "-c",
        "import sys; import lens_nsight_bridge; "
        "assert 'torch' not in sys.modules; assert 'nemo.lens' not in sys.modules"],
        cwd=ROOT, env=clean_environment(), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_configuration_flags_are_explicit_and_fail_without_nvtx(tmp_path):
    if torch.cuda.is_available():
        pytest.skip("CPU unavailable-backend regression only")
    env = clean_environment()
    env.update(NEMO_LENS_NSIGHT_BRIDGE="1", NEMO_LENS_NSIGHT_NVTX="1")
    result = subprocess.run([sys.executable, "-m", "lens_nsight_bridge", "run",
        "--output", str(tmp_path / "requested")], cwd=ROOT, env=env,
        capture_output=True, text=True)
    assert result.returncode == 2
    assert "NVTX requested" in result.stderr
    assert not (tmp_path / "requested").exists()


def test_output_reuse_refused(tmp_path):
    with pytest.raises(FileExistsError):
        Bridge(tmp_path, ROOT)


def test_nvtx_request_fails_without_cuda(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="NVTX requested"):
        Bridge(tmp_path / "fail", ROOT, nvtx=True)
    assert not (tmp_path / "fail").exists()


def test_exception_status_and_pop_mock_only(monkeypatch, tmp_path):
    # Mock validates control-flow cleanup, NOT real NVTX capture.
    calls = []
    class MockNvtx:
        def range_push(self, label):
            calls.append(label)
        def range_pop(self):
            calls.append("pop")
    monkeypatch.setattr("lens_nsight_bridge.bridge.requested_nvtx", lambda: MockNvtx())
    with Bridge(tmp_path / "error", ROOT, nvtx=True) as bridge:
        with pytest.raises(AttributeError):
            bridge.call("activation", activation, None)
    assert calls[-1] == "pop"
    record = json.loads((tmp_path / "error/operations.jsonl").read_text())
    assert record["status"] == "error"


def test_lens_real_local_spans_and_cli_snippet(tmp_path):
    pytest.importorskip("nemo.lens")
    out = tmp_path / "lens"
    result = subprocess.run([sys.executable, "-m", "lens_nsight_bridge", "run", "--lens",
                             "--output", str(out)], cwd=ROOT, env=clean_environment(),
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    operations = [json.loads(line) for line in (out / "operations.jsonl").read_text().splitlines()]
    spans = [json.loads(line) for line in (out / "lens-spans.jsonl").read_text().splitlines()]
    assert len(spans) == len(operations) == 21
    by_id = {span["attributes"]["bridge.id"]: span for span in spans}
    for record in operations:
        span = by_id[record["bridge_id"]]
        assert span["name"] == record["nvtx_label"]
        assert span["attributes"]["source.file_sha256"] == record["source"]["file_sha256"]
        assert span["attributes"]["source.id"] == record["source"]["source_id"]
        assert not record["nvtx_emitted"]
    selection = next(r for r in operations if r["role"] == "activation")
    snippet = subprocess.run([sys.executable, "-m", "lens_nsight_bridge", "snippet",
        "--output", str(out), "--bridge-id", selection["bridge_id"], "--current-root", str(ROOT)],
        cwd=ROOT, env=clean_environment(), capture_output=True, text=True)
    assert snippet.returncode == 0, snippet.stderr
    assert "F.silu(gate) * value" in snippet.stdout


def test_requested_lens_sampler_drop_fails(tmp_path):
    pytest.importorskip("nemo.lens")
    env = clean_environment()
    env["OTEL_TRACES_SAMPLER"] = "always_off"
    result = subprocess.run([sys.executable, "-m", "lens_nsight_bridge", "run", "--lens",
        "--output", str(tmp_path / "drop")], cwd=ROOT, env=env, capture_output=True, text=True)
    assert result.returncode == 2
    assert "not recording" in result.stderr
    assert not (tmp_path / "drop/result.json").exists()


def test_manifest_is_official_configuration_shape():
    manifest = yaml.safe_load((ROOT / "nsight/nemo_lens_bridge.yaml").read_text())
    assert set(manifest) == {"PluginName", "Description", "ExtendedDescription", "TargetEnvironment"}
    assert manifest["TargetEnvironment"]["NEMO_LENS_NSIGHT_BRIDGE"] == "1"
    assert "LibraryPath" not in manifest
    assert "ExecutablePath" not in manifest


def test_doctor_installed_probes_mock_only(monkeypatch):
    calls = []
    monkeypatch.setattr(cli.shutil, "which", lambda name: "/mock/nsys")
    def fake(command, env=None):
        calls.append(command)
        assert str(ROOT / "nsight") in env["NSYS_PLUGIN_SEARCH_DIRS"]
        return {"command": command, "returncode": 0, "stdout": "--enable --backtrace sampling", "stderr": ""}
    monkeypatch.setattr(cli, "probe", fake)
    result = cli.doctor()
    assert len(calls) == 4
    assert ["/mock/nsys", "plugins", "list"] in calls
    assert result["backtrace_help_mentions"]


def test_launch_blocked_without_nsys(monkeypatch, tmp_path):
    monkeypatch.setattr(cli.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="nsys absent"):
        cli.launch(type("Args", (), {"output": str(tmp_path / "out")})())
