"""CPU-only downloader contracts; no remote downloads or student reads."""
import copy
import hashlib
import importlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def restore_downloader_environment():
    """Cache isolation tests must not change subsequent test-process behavior."""
    import os
    before = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(before)


def module():
    return importlib.import_module("midtraining.data.download_teacher")


def metadata():
    """Handwritten inventory shape, NOT a retrieved Hub response or real digests."""
    downloader = module()
    shard_bytes, remainder = divmod(downloader.WEIGHT_BYTES, downloader.WEIGHT_COUNT)
    siblings = []
    for number in range(1, downloader.WEIGHT_COUNT + 1):
        size = shard_bytes + (number <= remainder)
        siblings.append({
            'rfilename': f'model-{number:05d}-of-00063.safetensors', 'size': size,
            'lfs': {'size': size, 'sha256': hashlib.sha256(f'fixture-{number}'.encode()).hexdigest()}})
    for name in ['config.json', 'model.safetensors.index.json', 'tokenizer.json',
                 'tokenizer_config.json', 'README.md']:
        siblings.append({'rfilename': name, 'size': 2,
                         'blobId': hashlib.sha1(b'blob 2\0{}').hexdigest()})
    return {'id': downloader.REPO_ID, 'modelId': downloader.REPO_ID,
            'sha': downloader.REVISION, 'siblings': siblings}


@pytest.mark.parametrize("mutation", ["repo", "revision", "traversal", "absolute", "backslash", "duplicate", "digest", "lfs_size", "missing_weight", "size", "alias"])
def test_manifest_rejects_untrusted_or_incomplete_metadata(mutation):
    m = metadata()
    weight = next(s for s in m["siblings"] if s["rfilename"].endswith(".safetensors"))
    if mutation == "repo":
        m["id"] = "wrong/model"
    elif mutation == "alias":
        m["modelId"] = "wrong/model"
    elif mutation == "revision":
        m["sha"] = "main"
    elif mutation in {"traversal", "absolute", "backslash"}:
        m["siblings"].append({"rfilename": {"traversal": "../x.py", "absolute": "/x", "backslash": "x\\\\y"}[mutation], "size": 0})
    elif mutation == "duplicate":
        m["siblings"].append(copy.deepcopy(weight))
    elif mutation == "digest":
        del weight["lfs"]["sha256"]
    elif mutation == "lfs_size":
        weight["lfs"]["size"] -= 1
    elif mutation == "missing_weight":
        m["siblings"].remove(weight)
    elif mutation == "size":
        weight["size"] = True
    with pytest.raises(ValueError):
        module().build_manifest(m)


def test_verify_local_file_checks_length_and_sha256(tmp_path):
    path = tmp_path / "small.safetensors"
    path.write_bytes(b"abc")
    entry = {"path": path.name, "size": 3, "sha256": hashlib.sha256(b"abc").hexdigest(), "git_blob": None}
    assert module().verify_file(path, entry)["sha256"] == entry["sha256"]
    path.write_bytes(b"abd")
    with pytest.raises(ValueError, match="SHA256"):
        module().verify_file(path, entry)
    path.write_bytes(b"a")
    with pytest.raises(ValueError, match="length"):
        module().verify_file(path, entry)


def test_verify_git_blob_and_reject_symlink(tmp_path):
    path = tmp_path / "config.json"
    path.write_bytes(b"{}")
    entry = {"path": path.name, "size": 2, "sha256": None, "git_blob": hashlib.sha1(b"blob 2\0{}").hexdigest()}
    assert module().verify_file(path, entry)["git_blob"] == entry["git_blob"]
    path.write_bytes(b"[]")
    with pytest.raises(ValueError, match="Git blob"):
        module().verify_file(path, entry)
    link = tmp_path / "link"
    link.symlink_to(path)
    with pytest.raises(ValueError, match="regular"):
        module().verify_file(link, entry)


def test_download_retries_resumes_verifies_and_is_idempotent(tmp_path, monkeypatch):
    d = module()
    monkeypatch.setattr(d, "RAID_ROOT", tmp_path)
    body = b"teacher bytes"
    entry = {"path": "model-00001-of-00063.safetensors", "size": len(body), "sha256": hashlib.sha256(body).hexdigest(), "git_blob": None}
    monkeypatch.setattr(d, "build_manifest", lambda _: [entry])
    output, receipts = tmp_path / "teacher", tmp_path / "receipts"
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        partial = output / ".cache/huggingface/download/partial.incomplete"
        partial.parent.mkdir(parents=True, exist_ok=True)
        if len(calls) == 1:
            partial.write_bytes(body[:3])
            raise OSError("transient transport failure")
        assert partial.read_bytes() == body[:3]
        (output / entry["path"]).write_bytes(body)
        return str(output / entry["path"])

    result = d.download_teacher({}, output, receipts, download_fn=download, sleep=lambda _: None)
    assert result["status"] == "complete"
    assert result["complete_files"] == result["expected_files"] == 1
    assert len(calls) == 2
    for kwargs in calls:
        assert kwargs["repo_id"] == d.REPO_ID
        assert kwargs["revision"] == d.REVISION
        assert kwargs["token"] is False
        assert Path(kwargs["cache_dir"]).is_relative_to(tmp_path)
        assert not kwargs.get("force_download", False)
    assert json.loads((receipts / "receipt.json").read_text()) == result
    snapshots = [json.loads(p.read_text()) for p in sorted(receipts.glob("progress-*.json"))]
    assert snapshots[0]["complete_files"] == 0
    assert snapshots[-1]["complete_files"] == 1
    assert any(s.get("event") == "retry" for s in snapshots)
    assert all("transient transport failure" not in json.dumps(s) for s in snapshots)
    d.download_teacher({}, output, receipts, download_fn=download, sleep=lambda _: None)
    assert len(calls) == 2
    assert Path(__import__("os").environ["HF_HOME"]).is_relative_to(tmp_path)
    assert Path(__import__("os").environ["HF_XET_CACHE"]).is_relative_to(tmp_path)


def test_failed_download_has_no_final_receipt_and_bounded_attempts(tmp_path, monkeypatch):
    d = module()
    monkeypatch.setattr(d, "RAID_ROOT", tmp_path)
    entry = {"path": "config.json", "size": 2, "sha256": hashlib.sha256(b"{}").hexdigest(), "git_blob": None}
    monkeypatch.setattr(d, "build_manifest", lambda _: [entry])
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        raise RuntimeError("secret-looking transport detail must not be logged")

    receipts = tmp_path / "receipts"
    with pytest.raises(RuntimeError, match="acquisition failed"):
        d.download_teacher({}, tmp_path / "output", receipts, download_fn=download, attempts=2, sleep=lambda _: None)
    assert len(calls) == 2
    assert not (receipts / "receipt.json").exists()
    progress = json.loads((receipts / "progress.json").read_text())
    assert progress["status"] == "failed" and progress["complete_files"] == 0
    assert "secret-looking" not in json.dumps(progress)


@pytest.mark.parametrize("unsafe", ["home", "hub_local", "existing_mismatch", "outside", "workers", "attempts"])
def test_downloader_fails_closed_without_network(tmp_path, monkeypatch, unsafe):
    d = module()
    monkeypatch.setattr(d, "RAID_ROOT", tmp_path)
    entry = {"path": "config.json", "size": 2, "sha256": hashlib.sha256(b"{}").hexdigest(), "git_blob": None}
    monkeypatch.setattr(d, "build_manifest", lambda _: [entry])
    output = tmp_path / "output"
    output.mkdir()
    other = tmp_path / "other"
    other.mkdir()
    if unsafe in {"home", "hub_local"}:
        cache = output / (".hf-home" if unsafe == "home" else ".cache")
        cache.symlink_to(other, target_is_directory=True)
    if unsafe == "existing_mismatch":
        (output / "config.json").write_bytes(b"[]")
    if unsafe == "outside":
        monkeypatch.setattr(d, "RAID_ROOT", tmp_path / "approved-only")
    calls = []
    with pytest.raises((ValueError, RuntimeError)):
        d.download_teacher({}, output, tmp_path / "receipts", download_fn=lambda **k: calls.append(k),
                           workers=16 if unsafe == "workers" else 4,
                           attempts=0 if unsafe == "attempts" else 2, sleep=lambda _: None)
    assert not calls
    assert not (tmp_path / "receipts/receipt.json").exists()


def test_cli_help_and_reject_bad_metadata_without_network(tmp_path):
    import subprocess
    import sys
    script = ROOT / "midtraining/data/download_teacher.py"
    help_result = subprocess.run([sys.executable, str(script), "--help"], text=True, capture_output=True)
    assert help_result.returncode == 0
    assert all(flag in help_result.stdout for flag in ("--metadata", "--output", "--receipt-dir"))
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"id": "wrong", "sha": "main"}))
    result = subprocess.run([sys.executable, str(script), "--metadata", str(bad), "--output", str(tmp_path / "output"), "--receipt-dir", str(tmp_path / "receipts")], text=True, capture_output=True)
    assert result.returncode != 0
    assert not (tmp_path / "output").exists()


def test_preimported_hub_cache_mismatch_fails_before_network(tmp_path, monkeypatch):
    d = module()
    import huggingface_hub
    from huggingface_hub import constants
    monkeypatch.setattr(d, "RAID_ROOT", tmp_path)
    monkeypatch.setattr(constants, "HF_XET_CACHE", "/home/unsafe-cache")
    monkeypatch.setattr(d, "build_manifest", lambda _: [])
    calls = []
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda **k: calls.append(k))
    with pytest.raises(RuntimeError, match="fresh process"):
        d.download_teacher({}, tmp_path / "output", tmp_path / "receipts")
    assert not calls


def test_retry_does_not_enable_hf_transfer(tmp_path, monkeypatch):
    d = module()
    monkeypatch.setattr(d, "RAID_ROOT", tmp_path)
    monkeypatch.setattr(d, "build_manifest", lambda _: [])
    monkeypatch.setenv("HF_HUB_ENABLE_HF_TRANSFER", "1")
    d.download_teacher({}, tmp_path / "output", tmp_path / "receipts", download_fn=lambda **_: None)
    assert __import__("os").environ["HF_HUB_ENABLE_HF_TRANSFER"] == "0"


def test_reverification_failure_retires_stale_success_receipt(tmp_path, monkeypatch):
    d = module()
    monkeypatch.setattr(d, "RAID_ROOT", tmp_path)
    entry = {"path": "config.json", "size": 2, "sha256": hashlib.sha256(b"{}").hexdigest(), "git_blob": None}
    monkeypatch.setattr(d, "build_manifest", lambda _: [entry])
    output, receipts = tmp_path / "output", tmp_path / "receipts"
    output.mkdir()
    (output / "config.json").write_bytes(b"{}")
    d.download_teacher({}, output, receipts, download_fn=lambda **_: pytest.fail("must not download existing files"))
    (output / "config.json").write_bytes(b"[]")
    with pytest.raises(RuntimeError, match="acquisition failed"):
        d.download_teacher({}, output, receipts, download_fn=lambda **_: pytest.fail("must not overwrite mismatching files"))
    assert not (receipts / "receipt.json").exists()
    assert len(list(receipts.glob("receipt-previous-*.json"))) == 1
    assert json.loads((receipts / "progress.json").read_text())["status"] == "failed"
    assert (output / "config.json").read_bytes() == b"[]"


@pytest.mark.parametrize("workers", [4, 8])
def test_actual_worker_pool_is_bounded(tmp_path, monkeypatch, workers):
    import threading
    import time
    d = module()
    monkeypatch.setattr(d, "RAID_ROOT", tmp_path)
    entries = [{"path": f"file-{i}.json", "size": 2, "sha256": hashlib.sha256(b"{}").hexdigest(), "git_blob": None} for i in range(12)]
    monkeypatch.setattr(d, "build_manifest", lambda _: entries)
    lock = threading.Lock()
    active, peak = 0, 0

    def download(**kwargs):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.02)
        path = Path(kwargs["local_dir"]) / kwargs["filename"]
        path.write_bytes(b"{}")
        with lock:
            active -= 1
        return str(path)

    result = d.download_teacher({}, tmp_path / "output", tmp_path / "receipts", workers=workers, download_fn=download)
    assert 1 < peak <= workers
    assert result["complete_files"] == 12
    assert len(result["files"]) == 12


def test_installed_hub_fresh_import_uses_raid_caches_without_transfer(tmp_path):
    import subprocess
    import sys
    code = '''
import hashlib, sys
from pathlib import Path
from midtraining.data import download_teacher as d
root = Path(sys.argv[1])
d.RAID_ROOT = root
output = root / "output"
output.mkdir()
(output / "config.json").write_bytes(b"{}")
entry = {"path": "config.json", "size": 2, "sha256": hashlib.sha256(b"{}").hexdigest(), "git_blob": None}
d.build_manifest = lambda _: [entry]
r = d.download_teacher({}, output, root / "receipts")
from huggingface_hub import constants
assert Path(constants.HF_XET_CACHE).is_relative_to(root)
assert Path(constants.HF_HOME).is_relative_to(root)
assert r["complete_files"] == 1
print("installed_hub_cache_and_local_verification_passed")
'''
    result = subprocess.run([sys.executable, "-c", code, str(tmp_path)], cwd=ROOT, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert "installed_hub_cache_and_local_verification_passed" in result.stdout


def test_pinned_manifest_exact_weights_and_no_code():
    manifest = module().build_manifest(metadata())
    weights = [f for f in manifest if f["path"].endswith(".safetensors")]
    assert len(weights) == 63
    assert sum(f["size"] for f in weights) == 248913061312
    assert all(len(f["sha256"]) == 64 for f in weights)
    names = {f["path"] for f in manifest}
    assert {"config.json", "tokenizer.json", "tokenizer_config.json", "README.md", "model.safetensors.index.json"} <= names
    assert not any(n.endswith((".py", ".png", ".pdf")) for n in names)
