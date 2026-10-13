"""Mock-only backend failure/launch regressions; no GPU capture evidence."""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from lens_nsight_bridge import Bridge, cli
from lens_nsight_bridge.control import activation

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cleanup_bridge(tmp_path, monkeypatch):
    bridge = Bridge(tmp_path / "cleanup", ROOT)
    bridge.records.close()  # Replace only local streams/exporter with cleanup mocks.
    attempts = []
    errors = {}

    def cleanup(stage):
        attempts.append(stage)
        if stage in errors:
            raise errors[stage]

    monkeypatch.setattr(bridge, "handle", SimpleNamespace(shutdown=lambda: cleanup("handle.shutdown")))
    monkeypatch.setattr(bridge, "lens_file", SimpleNamespace(close=lambda: cleanup("lens_file.close")))
    monkeypatch.setattr(bridge, "records", SimpleNamespace(close=lambda: cleanup("records.close")))
    return bridge, attempts, errors


@pytest.mark.parametrize("stage", ["handle.shutdown", "lens_file.close", "records.close"])
def test_body_exception_preserved_when_cleanup_fails(cleanup_bridge, stage):
    bridge, attempts, errors = cleanup_bridge
    errors[stage] = OSError("private cleanup detail")
    original = ValueError("body failure")
    with pytest.raises(ValueError) as caught:
        with bridge:
            raise original
    assert caught.value is original
    assert attempts == ["handle.shutdown", "lens_file.close", "records.close"]
    assert any(stage in note and "OSError" in note for note in original.__notes__)
    assert all("private cleanup detail" not in note for note in original.__notes__)
    bridge.close()
    assert len(attempts) == 3  # Cleanup attempts are not retried, including failures.


@pytest.mark.parametrize("stage", ["handle.shutdown", "lens_file.close", "records.close"])
def test_preannotated_cleanup_notes_not_transferred_to_body(cleanup_bridge, stage):
    bridge, attempts, errors = cleanup_bridge
    cleanup_error = OSError("SYNTHETIC_PRIVATE_VALUE_IN_MESSAGE")
    cleanup_error.add_note("SYNTHETIC_PRIVATE_VALUE_IN_EXISTING_NOTE")
    errors[stage] = cleanup_error
    original = ValueError("body failure")
    original.add_note("existing body context")
    with pytest.raises(ValueError) as caught, bridge:
        raise original
    assert caught.value is original
    assert attempts == ["handle.shutdown", "lens_file.close", "records.close"]
    assert original.__notes__ == [
        "existing body context",
        f"Bridge cleanup failed during {stage}: OSError (message omitted)",
    ]
    assert "SYNTHETIC_PRIVATE_VALUE_IN_EXISTING_NOTE" in cleanup_error.__notes__
    assert all("SYNTHETIC_PRIVATE_VALUE" not in note for note in original.__notes__)
    bridge.close()
    assert len(attempts) == 3


@pytest.mark.parametrize("stage", ["handle.shutdown", "lens_file.close", "records.close"])
def test_successful_body_propagates_cleanup_failure(cleanup_bridge, stage):
    bridge, attempts, errors = cleanup_bridge
    errors[stage] = OSError("cleanup failure")
    with pytest.raises(OSError) as caught:
        with bridge:
            pass
    assert caught.value is errors[stage]
    assert attempts == ["handle.shutdown", "lens_file.close", "records.close"]
    bridge.close()
    assert len(attempts) == 3


@pytest.mark.parametrize("body_fails", [False, True])
def test_all_cleanup_attempts_and_failures_retained(cleanup_bridge, body_fails):
    bridge, attempts, errors = cleanup_bridge
    errors.update({"handle.shutdown": OSError("shutdown detail"),
                   "lens_file.close": RuntimeError("stream detail"),
                   "records.close": KeyboardInterrupt("interrupt detail")})
    original = ValueError("body failure") if body_fails else errors["handle.shutdown"]
    with pytest.raises(type(original)) as caught:
        with bridge:
            if body_fails:
                raise original
    assert caught.value is original
    assert attempts == ["handle.shutdown", "lens_file.close", "records.close"]
    for stage, failure in errors.items():
        assert any(stage in note and type(failure).__name__ in note
                   for note in original.__notes__)
    bridge.close()
    assert len(attempts) == 3


def test_successful_cleanup_is_idempotent(cleanup_bridge):
    bridge, attempts, _ = cleanup_bridge
    with bridge:
        pass
    bridge.close()
    assert attempts == ["handle.shutdown", "lens_file.close", "records.close"]


@pytest.mark.parametrize("stage,function_fails", [("push", False), ("pop", False), ("pop", True)])
def test_nvtx_backend_failure_records_attempt_and_preserves_primary(monkeypatch, tmp_path, stage, function_fails):
    calls = []
    backend_error = RuntimeError("fake NVTX failure")
    class FakeNvtx:
        def range_push(self, label):
            calls.append("push")
            if stage == "push":
                raise backend_error
        def range_pop(self):
            calls.append("pop")
            raise backend_error
    monkeypatch.setattr("lens_nsight_bridge.bridge.requested_nvtx", lambda: FakeNvtx())
    out = tmp_path / "attempt"
    with Bridge(out, ROOT, nvtx=True) as bridge:
        with pytest.raises(AttributeError if function_fails else RuntimeError) as caught:
            bridge.call("activation", activation, None if function_fails else torch.ones(2))
    if function_fails:
        assert "chunk" in str(caught.value)
    else:
        assert caught.value is backend_error
    assert calls == (["push"] if stage == "push" else ["push", "pop"])
    record = json.loads((out / "operations.jsonl").read_text())
    assert record["status"] == "error"
    assert record["nvtx_emitted"] == (stage == "pop")
    assert record["nvtx_error"] == {"stage": stage, "type": "RuntimeError"}
    assert record["source"]["path"] == "lens_nsight_bridge/control.py"
    assert record["bridge_id"] in record["nvtx_label"]
    assert record["host_end_monotonic_ns"] >= record["host_start_monotonic_ns"]


@pytest.fixture
def mock_launch_backend(monkeypatch):
    monkeypatch.setattr(cli, "doctor", lambda: {
        "nsys_path": "/mock/nsys", "cuda_available": True,
        "profile_help": {"returncode": 0, "stdout": "--enable"},
        "plugins_list": {"returncode": 0, "stdout": "nemo_lens_nsight_bridge"},
    })
    calls = []
    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=17)
    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    return calls


def test_launch_argv_cwd_environment_and_exit_status(monkeypatch, tmp_path, mock_launch_backend):
    monkeypatch.setenv("NSYS_PLUGIN_SEARCH_DIRS", "/trusted/first" + os.pathsep + "/trusted/second")
    monkeypatch.setenv("BRIDGE_TEST_INHERITED", "preserved")
    before = os.environ.copy()
    out = tmp_path / "fresh"
    assert cli.launch(SimpleNamespace(output=str(out))) == 17
    assert len(mock_launch_backend) == 1
    command, kwargs = mock_launch_backend[0]
    assert command == ["/mock/nsys", "profile", "--enable=nemo_lens_nsight_bridge",
        "--trace=cuda,nvtx", "--output=" + str(tmp_path / "fresh-nsys"), cli.sys.executable,
        "-m", "lens_nsight_bridge", "run", "--device=cuda", "--lens", "--output=" + str(out)]
    assert kwargs["cwd"] == ROOT
    expected = before.copy()
    expected["NSYS_PLUGIN_SEARCH_DIRS"] = str(ROOT / "nsight") + os.pathsep + before["NSYS_PLUGIN_SEARCH_DIRS"]
    assert kwargs["env"] == expected
    assert os.environ == before
    assert not out.exists()  # Mock did not launch/capture or create result artifacts.


@pytest.mark.parametrize("existing", ["fresh", "fresh-nsys", "fresh-nsys.nsys-rep", "fresh-nsys.sqlite"])
def test_launch_output_and_sidecar_prefix_refused(tmp_path, mock_launch_backend, existing):
    occupied = tmp_path / existing
    occupied.write_text("preserve me")
    with pytest.raises(FileExistsError):
        cli.launch(SimpleNamespace(output=str(tmp_path / "fresh")))
    assert not mock_launch_backend
    assert occupied.read_text() == "preserve me"
