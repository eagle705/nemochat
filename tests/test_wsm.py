"""Launcher guards, plus opt-in unmodified official CPU integration."""
import argparse
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import Mock

import pytest
from nemochat.checkpoints.wsm import REVISION, parser, run, validate_upstream


@pytest.mark.parametrize("basename", ["iter_0", "iter_30", "iter_0000030"])
@pytest.mark.parametrize("dry_run", [False, True])
def test_iteration_output_rejected_before_invocation(tmp_path, monkeypatch, basename, dry_run):
    parent = tmp_path / "outputs"
    parent.mkdir()
    tracker = parent / "latest_checkpointed_iteration.txt"
    sentinel = b"existing training tracker: 98765\n"
    tracker.write_bytes(sentinel)
    (tmp_path / "inputs").mkdir()
    invocation = Mock(side_effect=AssertionError("No subprocess may run"))
    monkeypatch.setattr(subprocess, "run", invocation)
    monkeypatch.setattr(subprocess, "check_output", invocation)
    with pytest.warns(UserWarning), pytest.raises(ValueError, match="basename"):
        run(args(tmp_path, output=parent / basename, dry_run=dry_run))
    invocation.assert_not_called()
    assert tracker.read_bytes() == sentinel
    assert not (parent / basename).exists()


@pytest.mark.parametrize("failure", ["tensor", "bytes", "optimizer", "common", "style", "opt_in", "revision", "hash", "guard"])
def test_verification_guards_survive_optimized_python(failure):
    script = Path(__file__).resolve().parents[1] / "scripts" / "verify_wsm_merge.py"
    code = r'''
import ast
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace
module = runpy.run_path(sys.argv[1])
if sys.flags.optimize != 1:
    raise RuntimeError("Regression must run under -O")
if any(isinstance(node, ast.Assert) for node in ast.walk(ast.parse(Path(sys.argv[1]).read_text()))):
    raise RuntimeError("Verifier must not contain optimization-removable assertions")
loaded = {"model.tensor._extra_state": SimpleNamespace(tolist=lambda: [30]),
          "model.bytes._extra_state/shard_0_1": [{"marker": 30}]}
common = {"iteration": 30, "weighted_merge_provenance": {
    "merge_style": "minus-sqrt", "ignore_non_model_state": True}}
wrapper = {"revision": module["REVISION"], "tool_sha256": "a" * 64}
if module["validate_observed_state"](loaded, common, wrapper) != 30:
    raise RuntimeError("Observed iteration not returned")
failure = sys.argv[2]
if failure == "tensor":
    loaded["model.tensor._extra_state"] = SimpleNamespace(tolist=lambda: [10])
elif failure == "bytes":
    loaded["model.bytes._extra_state/shard_0_1"] = [{"marker": 10}]
elif failure == "optimizer":
    loaded["optimizer.moment"] = None
elif failure == "common":
    common["iteration"] = 10
elif failure == "style":
    common["weighted_merge_provenance"]["merge_style"] = "linear"
elif failure == "opt_in":
    common["weighted_merge_provenance"]["ignore_non_model_state"] = False
elif failure == "revision":
    wrapper["revision"] = "wrong"
elif failure == "hash":
    wrapper["tool_sha256"] = "short"
try:
    if failure == "guard":
        module["require"](False, "deliberate failure")
    else:
        module["validate_observed_state"](loaded, common, wrapper)
except AssertionError:
    print("optimized failure guard active:", failure)
else:
    raise RuntimeError("Invalid fixture accepted under -O: " + failure)
'''
    result = subprocess.run([sys.executable, "-O", "-c", code, str(script), failure],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"optimized failure guard active: {failure}" in result.stdout


def args(tmp_path, **overrides):
    values = dict(upstream=tmp_path / "upstream", python="python", checkpoint_root=tmp_path / "inputs",
                  output=tmp_path / "output", target_iteration=30, window_iterations=20,
                  min_iteration_interval=1, min_checkpoints=2, trusted_inputs=True,
                  ignore_non_model_state=False, dry_run=False)
    values.update(overrides)
    return argparse.Namespace(**values)


def test_requires_explicit_trust(tmp_path):
    with pytest.raises(ValueError, match="trusted-inputs"):
        run(args(tmp_path, trusted_inputs=False))


@pytest.mark.parametrize("override", [{"target_iteration": -1}, {"window_iterations": 0},
                                      {"min_iteration_interval": 0}, {"min_checkpoints": 0}])
def test_invalid_arguments(tmp_path, override):
    with pytest.warns(UserWarning), pytest.raises(ValueError):
        run(args(tmp_path, **override))


def test_no_overwrite(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    with pytest.warns(UserWarning), pytest.raises(ValueError, match="overwrite"):
        run(args(tmp_path))


def test_reject_overlap(tmp_path):
    with pytest.warns(UserWarning), pytest.raises(ValueError, match="overlap"):
        run(args(tmp_path, output=tmp_path / "inputs" / "merged"))


def test_wrong_revision(tmp_path):
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(tmp_path), "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                    "commit", "--allow-empty", "-m", "test fixture"], check=True, capture_output=True)
    with pytest.raises(ValueError, match=REVISION):
        validate_upstream(tmp_path)


def test_policy_has_no_schedule_branches():
    with pytest.raises(SystemExit):
        parser().parse_args(["--merge-style", "linear"])


def test_official_cpu_fixture(tmp_path):
    upstream = os.environ.get("NEMOCHAT_WSM_UPSTREAM")
    if not upstream:
        pytest.skip("Set NEMOCHAT_WSM_UPSTREAM and run with the isolated upstream interpreter")
    script = Path(__file__).resolve().parents[1] / "scripts" / "verify_wsm_merge.py"
    spec = importlib.util.spec_from_file_location("verify_wsm_merge", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = module.verify(Path(upstream), tmp_path / "fixture")
    assert report["revision"] == REVISION
    assert len(report["checks"]) == 8
