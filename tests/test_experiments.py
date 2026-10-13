"""Contract and runner tests; no native hybrid/GPU evidence is claimed."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from nemochat import experiments as exp

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "configs/experiments/dense_cpu.yaml"


@pytest.fixture
def config():
    return exp.load_config(TEMPLATE).to_dict()


def cli(*args):
    return subprocess.run([sys.executable, str(ROOT / "scripts/run_experiment.py"), *map(str, args)], capture_output=True, text=True, timeout=15)


def test_templates_and_immutability():
    resolved = exp.load_config(TEMPLATE)
    assert len(resolved.source_sha256) == 64
    with pytest.raises(TypeError):
        resolved.config["workload"]["batch_size"] = 9
    detached = resolved.to_dict()
    detached["workload"]["batch_size"] = 9
    assert resolved.config["workload"]["batch_size"] == 2
    assert exp.load_config(ROOT / "configs/experiments/dense_cuda.yaml").config["resources"]["max_gpu_count"] == 1


@pytest.mark.parametrize("section,key,value", [
    ("root", "unknown", 1), ("workload", "command", "curl invalid"),
    ("root", "schema_version", True), ("experiment", "seed", -1),
    ("experiment", "name", "../bad"), ("workload", "batch_size", True),
    ("workload", "sequence_length", 0), ("workload", "hidden_size", 33),
    ("workload", "learning_rate", float("nan")), ("workload", "learning_rate", 0), ("workload", "learning_rate", 10**1000),
    ("workload", "activation_checkpointing", 1), ("measurement", "profile", "false"),
    ("measurement", "warmup_steps", -1), ("measurement", "measured_steps", 1001),
    ("measurement", "repeats", 21), ("resources", "max_seconds", 3601),
    ("resources", "max_gpu_count", 2), ("workload", "device", "cuda"),
    ("workload", "dtype", "bfloat16"), ("workload", "vocab_size", 65537),
])
def test_bad_fields(config, section, key, value):
    target = config if section == "root" else config[section]
    target[key] = value
    with pytest.raises(exp.ContractError):
        exp.validate_config(config)


@pytest.mark.parametrize("mutation", [
    {"measurement": {"measured_steps": 1000, "repeats": 20}},
    {"workload": {"hidden_size": 4096, "ffn_size": 16384}},
    {"workload": {"batch_size": 64, "sequence_length": 4096}},
])
def test_combined_budgets(config, mutation):
    for section, updates in mutation.items():
        config[section].update(updates)
    with pytest.raises(exp.ContractError):
        exp.validate_config(config)


@pytest.mark.parametrize("text", [
    "schema_version: 1\nschema_version: 1\n", "x: &anchor 1\ny: *anchor\n",
    "x: !!python/object/apply:os.system ['false']", "x: !!str hello",
    "x: [", "[]", "x: {y: 1, y: 2}", "? [a, b]\n: 1", "x: " + "9" * 5000, "x: " + "[" * 12 + "0" + "]" * 12,
    "x" * (exp.MAX_CONFIG_BYTES + 1), "true: 1", "---\nx: 1\n---\nx: 2",
])
def test_unsafe_or_malformed_yaml(tmp_path, text):
    path = tmp_path / "config.yaml"
    path.write_text(text)
    with pytest.raises(exp.ContractError):
        exp.load_config(path)


def test_missing_fields(config):
    del config["measurement"]["profile"]
    with pytest.raises(exp.ContractError):
        exp.validate_config(config)


def test_cli_preview_no_writes(tmp_path):
    output = tmp_path / "no-write"
    for command in ("validate", "dry-run"):
        result = cli(command, "--config", TEMPLATE, "--output", output)
        assert result.returncode == 0, result.stderr
        data = json.loads(result.stdout)
        assert data["hardware_checked"] is False
        assert "native hybrid NOT validated" in data["evidence_label"]
        assert not output.exists()
    assert cli("run", "--config", TEMPLATE).returncode == 2
    assert cli("validate", "--config", tmp_path / "missing").returncode == 2


@pytest.mark.parametrize("returncode,timed_out,stderr,status", [
    (0, False, "", "success"), (1, False, "bug", "failed"),
    (3, False, "unavailable", "blocked"), (-9, True, "", "timeout"),
    (1, False, "torch.OutOfMemoryError: CUDA out of memory", "blocked"),
])
def test_run_status_artifacts(tmp_path, monkeypatch, returncode, timed_out, stderr, status):
    monkeypatch.setattr(exp, "_preflight", lambda config: None)
    def execute(argv, **kwargs):
        assert argv[1] == str(ROOT / "scripts/benchmark_training.py")
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
        assert 0 < kwargs["timeout"] <= 60
        kwargs["stderr"].write(stderr)
        return returncode, timed_out
    monkeypatch.setattr(exp, "execute_process", execute)
    output = tmp_path / "run"
    result = exp.run_experiment(exp.load_config(TEMPLATE), output, ROOT)
    assert result["status"] == status
    assert result["exit_code"] == exp.EXIT_CODES[status]
    assert json.loads((output / "result.json").read_text())["status"] == status
    assert "native hybrid NOT validated" in (output / "report.md").read_text()
    assert (output / "resolved_config.json").exists()
    with pytest.raises(FileExistsError):
        exp.run_experiment(exp.load_config(TEMPLATE), output, ROOT)


@pytest.mark.parametrize('returncode,kind,status,reason', [
    (3, 'out-of-memory', 'blocked', 'benchmark out-of-memory'),
    (4, 'time-budget-exceeded', 'timeout', 'benchmark time-budget-exceeded (cooperative deadline)'),
])
def test_typed_benchmark_failure_propagation(tmp_path, monkeypatch, returncode, kind, status, reason):
    monkeypatch.setattr(exp, '_preflight', lambda config: None)
    def execute(argv, **kwargs):
        kwargs['stdout'].write(json.dumps({'status': 'failed', 'failure_kind': kind}))
        # Cooperative timeout wins over an incidental OOM warning.
        if returncode == 4:
            kwargs['stderr'].write('out of memory warning')
        return returncode, False
    monkeypatch.setattr(exp, 'execute_process', execute)
    result = exp.run_experiment(exp.load_config(TEMPLATE), tmp_path / 'run', ROOT)
    assert result['status'] == status
    assert result['exit_code'] == returncode
    assert result['benchmark_returncode'] == returncode
    assert result['reason'] == reason


def test_blocked_preflight(tmp_path, monkeypatch):
    monkeypatch.setattr(exp, "_preflight", lambda config: "CUDA unavailable")
    monkeypatch.setattr(exp, "execute_process", lambda *a, **k: pytest.fail("must not spawn"))
    result = exp.run_experiment(exp.load_config(TEMPLATE), tmp_path / "blocked", ROOT)
    assert result["status"] == "blocked"
    assert result["reason"] == "CUDA unavailable"


def test_timeout_kills_descendant_group(tmp_path):
    # Exercise the helper with a real local child, not the experiment target.
    pid_path = tmp_path / "child.pid"
    code = "import subprocess,time,pathlib; p=subprocess.Popen(['" + sys.executable + "','-c','import time; time.sleep(30)']); pathlib.Path(" + repr(str(pid_path)) + ").write_text(str(p.pid)); time.sleep(30)"
    with (tmp_path / "stdout").open("w") as stdout, (tmp_path / "stderr").open("w") as stderr:
        returncode, timed_out = exp.execute_process([sys.executable, "-c", code], cwd=ROOT, env=os.environ.copy(), timeout=0.5, stdout=stdout, stderr=stderr)
    assert timed_out and returncode != 0
    child_pid = int(pid_path.read_text())
    # A zombie is terminated and cannot execute; Linux may await init reaping.
    stat = Path(f"/proc/{child_pid}/stat")
    assert not stat.exists() or stat.read_text().split()[2] == "Z"
