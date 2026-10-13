import copy
import importlib.util
import inspect
from pathlib import Path

import pytest

from lens_nsight_bridge.control import activation
from lens_nsight_bridge.source import extract_verified, identity, safe_path, snapshot_function

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def captured(tmp_path):
    ref = snapshot_function(activation, ROOT, tmp_path)
    return tmp_path, ref


def test_actual_python_snippet(captured):
    out, ref = captured
    snippet = extract_verified(out, ref, ROOT)
    assert snippet == inspect.getsource(activation)
    assert "F.silu(gate) * value" in snippet
    assert len(ref["file_sha256"]) == 64
    assert ref["source_id"] == identity(ref)


@pytest.mark.parametrize("name", ["../control.py", "/etc/passwd", "sources/../../a", "a\\b", "./a", "a/../b"])
def test_traversal_rejected(tmp_path, name):
    with pytest.raises(ValueError):
        safe_path(tmp_path, name)


def test_symlink_escape(tmp_path):
    (tmp_path / "outside").symlink_to("/etc", target_is_directory=True)
    with pytest.raises(ValueError):
        safe_path(tmp_path, "outside/passwd")


def test_snapshot_tamper(captured):
    out, ref = captured
    (out / ref["snapshot"]).write_text("# changed snapshot\n")
    with pytest.raises(ValueError, match="hash"):
        extract_verified(out, ref)


@pytest.mark.parametrize("key,value", [("path", "../evil.py"), ("snapshot", "../evil.py"),
    ("start_line", 0), ("end_line", 1000000), ("snippet_sha256", "0"*64),
    ("source_id", "0"*64)])
def test_reference_tamper(captured, key, value):
    out, ref = captured
    ref = copy.deepcopy(ref)
    ref[key] = value
    with pytest.raises(ValueError):
        extract_verified(out, ref)


def test_invalid_range_even_with_rehashed_identity(captured):
    out, ref = captured
    ref["end_line"] = 99999
    ref["source_id"] = identity(ref)
    with pytest.raises(ValueError, match="line range"):
        extract_verified(out, ref)


def test_wrong_snippet_hash_even_with_rehashed_identity(captured):
    out, ref = captured
    ref["snippet_sha256"] = "f"*64
    ref["source_id"] = identity(ref)
    with pytest.raises(ValueError, match="Snippet hash"):
        extract_verified(out, ref)


def test_checkout_drift(captured, tmp_path):
    out, ref = captured
    current = tmp_path / "current"
    target = current / ref["path"]
    target.parent.mkdir(parents=True)
    target.write_text("# changed checkout\n")
    assert "def activation" in extract_verified(out, ref)
    with pytest.raises(ValueError, match="checkout differs"):
        extract_verified(out, ref, current)


def test_verifier_rejects_tamper_under_python_optimization(captured):
    import json
    import subprocess
    import sys
    out, ref = captured
    reference = out / "reference.json"
    reference.write_text(json.dumps(ref))
    (out / ref["snapshot"]).write_text("# tampered\n")
    script = ("import json,sys; from pathlib import Path; "
              "from lens_nsight_bridge.source import extract_verified; "
              "extract_verified(sys.argv[1], json.loads(Path(sys.argv[2]).read_text()))")
    result = subprocess.run([sys.executable, "-O", "-c", script, str(out), str(reference)],
                            cwd=ROOT, capture_output=True, text=True)
    assert result.returncode != 0
    assert "Source snapshot hash/identity mismatch" in result.stderr


def test_snapshot_outside_declared_root(tmp_path):
    with pytest.raises(ValueError, match="outside"):
        snapshot_function(activation, tmp_path, tmp_path)


def test_stale_loaded_code_refused(tmp_path):
    path = tmp_path / "loaded.py"
    path.write_text("def f():\n    return 1\n")
    spec = importlib.util.spec_from_file_location("loaded", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = tmp_path / "out"
    out.mkdir()
    snapshot_function(module.f, tmp_path, out)
    path.write_text("def f():\n    return 2\n")
    with pytest.raises(ValueError, match="Loaded function differs"):
        snapshot_function(module.f, tmp_path, out)
