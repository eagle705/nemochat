"""Runtime directory defaults and separate benchmark log regression tests."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from nemochat.tracking import StageRun

ROOT = Path(__file__).resolve().parents[1]


def load_benchmark(monkeypatch):
    # Benchmark imports the isolated Lens training runtime; parser/path tests mock it.
    monkeypatch.setitem(sys.modules, "train", SimpleNamespace(
        MODES=("baseline", "disabled", "file", "otlp"),
        clean_environment=lambda: None, loopback_endpoint=lambda value: value))
    monkeypatch.setitem(sys.modules, "verify_jaeger", SimpleNamespace(
        fetch_trace=lambda *args: None, wait_ready=lambda *args: None))
    spec = importlib.util.spec_from_file_location("layout_benchmark", ROOT / "tutorials/nemo_lens/benchmark.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tracking_default_is_logs():
    run = StageRun("data")
    assert run.directory == Path("logs/wandb")
    assert not run.active


@pytest.mark.parametrize("relation", ["same", "nested", "existing"])
def test_benchmark_rejects_overlapping_or_existing_logs(tmp_path, monkeypatch, relation):
    benchmark = load_benchmark(monkeypatch)
    result = tmp_path / "results"
    logs = result if relation == "same" else result / "logs" if relation == "nested" else tmp_path / "logs"
    if relation == "existing":
        logs.mkdir()
        (logs / "keep").write_text("existing")
    monkeypatch.setattr(sys, "argv", ["benchmark", "--output", str(result), "--log-output", str(logs),
                                    "--skip-otlp", "--repetitions", "2"])
    with pytest.raises(SystemExit) as error:
        benchmark.main()
    assert error.value.code == 2
    assert not result.exists()
    if relation == "existing":
        assert (logs / "keep").read_text() == "existing"


def test_benchmark_writes_logs_separately(tmp_path, monkeypatch):
    benchmark = load_benchmark(monkeypatch)
    result, logs = tmp_path / "results", tmp_path / "logs"
    monkeypatch.setattr(sys, "argv", ["benchmark", "--output", str(result), "--log-output", str(logs),
                                    "--skip-otlp", "--repetitions", "2", "--steps", "1"])

    def fake_run(command, **kwargs):
        # Synthetic child-result fixture, not a performance measurement.
        child = Path(command[command.index("--output") + 1])
        child.mkdir()
        (child / "result.json").write_text(json.dumps(dict(
            parameter_sha256="fixture", losses=[1.0], mode=command[command.index("--mode") + 1],
            training_ns=100, setup_ns=10, shutdown_ns=10)))
        return SimpleNamespace(stdout="synthetic stdout", stderr="synthetic stderr")

    monkeypatch.setattr(benchmark.subprocess, "run", fake_run)
    monkeypatch.setattr(benchmark, "verify_file", lambda *args: None)
    benchmark.main()
    assert len(list(logs.rglob("*.log"))) == 12
    assert not list(result.rglob("*.log"))
    summary = json.loads((result / "benchmark.json").read_text())
    assert all(Path(row["log_directory"]).is_relative_to(logs) for row in summary["raw"])


def test_experiment_separates_logs_and_preserves_worker_failure(tmp_path, monkeypatch):
    from nemochat import experiments as exp

    repository = tmp_path / "repo"
    repository.mkdir()
    output = repository / "results" / "test"
    monkeypatch.setattr(exp, "_preflight", lambda config: None)

    def fake_execute(argv, **kwargs):
        kwargs["stdout"].write(json.dumps({"failure_kind": "out-of-memory"}))
        kwargs["stderr"].write("synthetic worker failure")
        return 3, False

    monkeypatch.setattr(exp, "execute_process", fake_execute)
    result = exp.run_experiment(exp.load_config(ROOT / "configs/experiments/dense_cpu.yaml"),
                                output, repository)
    logs = Path(result["log_directory"])
    assert logs.is_relative_to(repository / "logs" / "experiments")
    assert (logs / "stdout.log").exists() and (logs / "stderr.log").exists()
    assert not list(output.glob("*.log"))
    assert result["status"] == "blocked" and result["reason"] == "benchmark out-of-memory"
    assert json.loads((output / "result.json").read_text())["log_directory"] == str(logs)
    with pytest.raises(FileExistsError):
        exp.run_experiment(exp.load_config(ROOT / "configs/experiments/dense_cpu.yaml"),
                           output, repository)
