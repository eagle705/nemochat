"""Offline contracts and mocked Hub transports; not remote/B300 validation."""
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from nemochat.artifacts import hf_checkpoint as h


@pytest.fixture
def source(tmp_path):
    p = tmp_path / "source"
    p.mkdir()
    (p / "config.json").write_text('{"model_type":"fixture"}')
    (p / "model-00001-of-00002.safetensors").write_bytes(b"fixture shard one")
    (p / "model-00002-of-00002.safetensors").write_bytes(b"fixture shard two")
    (p / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
        "one": "model-00001-of-00002.safetensors", "two": "model-00002-of-00002.safetensors"}}))
    return p


@pytest.fixture
def packed(source, tmp_path):
    out = tmp_path / "packed"
    h.pack(source, out, kind="hf_model", stage="pretrain", run="fixture", checkpoint="step_10")
    return out


class Hub:
    def __init__(self, tmp_path, private=True):
        self.remote = tmp_path / "remote"
        self.remote.mkdir()
        self.private = private
        self.sha = "a" * 40
        self.calls = []
        self.corrupt = False
        self.fail = False

    def repo_info(self, **kw):
        self.calls.append(("info", kw))
        return SimpleNamespace(private=self.private, sha="b" * 40)

    def list_repo_files(self, **kw):
        self.calls.append(("list", kw))
        return [p.relative_to(self.remote).as_posix() for p in self.remote.rglob("*") if p.is_file()]

    def upload_folder(self, **kw):
        self.calls.append(("upload", kw))
        if self.fail:
            raise RuntimeError("mock network failure")
        shutil.copytree(kw["folder_path"], self.remote / kw["path_in_repo"])
        if self.corrupt:
            (self.remote / kw["path_in_repo"] / "model-00002-of-00002.safetensors").write_bytes(b"corrupt")
        return SimpleNamespace(oid=self.sha)

    def snapshot(self, **kw):
        self.calls.append(("snapshot", kw))
        for name in kw["allow_patterns"]:
            assert "*" not in name
            p = Path(kw["local_dir"]) / name
            p.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self.remote / name, p)
        return kw["local_dir"]


@pytest.fixture
def mocked(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "secret-fixture-never-in-receipt")
    return Hub(tmp_path)


def transfer(packed, mocked, **kw):
    return h.upload(packed, "owner/private", api=mocked, snapshot=mocked.snapshot, **kw)


def test_local_pack_restore(packed, tmp_path):
    out = tmp_path / "restored"
    h.restore_local(packed, out, expected_kind="hf_model")
    assert h.files(out) == h.files(packed)
    with pytest.raises(ValueError):
        h.restore_local(packed, out, expected_kind="hf_model")


@pytest.mark.parametrize("change", ["tamper", "extra", "missing", "symlink"])
def test_inventory_failures(packed, change):
    p = packed / "model-00002-of-00002.safetensors"
    if change == "tamper":
        p.write_bytes(b"tamper")
    elif change == "extra":
        (packed / "untracked").write_bytes(b"extra")
    elif change == "missing":
        p.unlink()
    else:
        p.unlink()
        p.symlink_to(packed / "config.json")
    with pytest.raises(ValueError):
        h.verify(packed)


@pytest.mark.parametrize("name", ["../escape", "/absolute", "a/../b", "a\\b", "a//b", "./a", "*.pt", "x?", "x[0]"])
def test_unsafe_names(name):
    with pytest.raises(ValueError):
        h.safe_name(name)


def test_index_shards(source, tmp_path):
    (source / "model.safetensors.index.json").write_text('{"weight_map":{"x":"../outside"}}')
    with pytest.raises(ValueError):
        h.pack(source, tmp_path / "bad", kind="hf_model", stage="s", run="r", checkpoint="c")


def test_training_contract_and_trust(tmp_path):
    src = tmp_path / "train"
    src.mkdir()
    names = {"model": ["torch_dist/.metadata", "torch_dist/shard_0.distcp", "torch_dist/shard_1.distcp"],
             "optimizer": ["optimizer.pt"], "rng": ["rng.pt"], "tokenizer": ["tokenizer.json"],
             "scheduler": ["scheduler.json"], "cursor": ["cursor.json"],
             "environment": ["environment.json"], "training_manifest": ["training_manifest.json"]}
    for paths in names.values():
        for name in paths:
            p = src / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b'{"fixture":true}' if name.endswith(".json") else b"opaque fixture, not real torch state")
    packed = tmp_path / "training-bundle"
    with pytest.raises(ValueError):
        h.pack(src, packed, kind="torch_dist_training", stage="s", run="r", checkpoint="c", components={})
    h.pack(src, packed, kind="torch_dist_training", stage="s", run="r", checkpoint="c", components=names)
    with pytest.raises(ValueError):
        h.restore_local(packed, tmp_path / "restore", expected_kind="torch_dist_training")
    h.restore_local(packed, tmp_path / "restore", expected_kind="torch_dist_training", trusted_inputs=True)
    with pytest.raises(ValueError):
        h.verify(packed, "hf_model")
    with pytest.raises(ValueError):
        h.pack(src, tmp_path / "fake-hf", kind="hf_model", stage="s", run="r", checkpoint="c")


def test_credentials_missing(packed, mocked, monkeypatch):
    monkeypatch.delenv("HF_TOKEN")
    with pytest.raises(ValueError, match="HF_TOKEN"):
        transfer(packed, mocked)
    assert not mocked.calls


def test_upload_readback_and_pinned_download(packed, mocked, tmp_path):
    receipt = transfer(packed, mocked)
    assert receipt["revision"] == "a" * 40
    assert "secret-fixture" not in json.dumps(receipt)
    out = tmp_path / "download"
    h.download("owner/private", receipt["revision"], receipt["path"], out,
               expected_kind="hf_model", api=mocked, snapshot=mocked.snapshot,
               expected_manifest_hash=receipt["manifest_sha256"])
    assert h.files(out) == h.files(packed)
    for operation, kw in mocked.calls:
        assert kw["token"] == "secret-fixture-never-in-receipt"
        assert kw["repo_type"] == "model"
        if operation == "snapshot":
            assert kw["revision"] == "a" * 40
    with pytest.raises(ValueError, match="already exists"):
        transfer(packed, mocked)


@pytest.mark.parametrize("revision", ["main", "v1", "abc", "../main", "A" * 40])
def test_revision_refusal(mocked, tmp_path, revision):
    with pytest.raises(ValueError, match="immutable"):
        h.download("owner/private", revision, "stages/s/runs/r/checkpoints/c/hf_model",
                   tmp_path / "out", expected_kind="hf_model", api=mocked, snapshot=mocked.snapshot)
    assert not mocked.calls


def test_public_policy(packed, mocked):
    mocked.private = False
    for kw in ({}, {"rights_reviewed": True}, {"rights_reviewed": True, "repo_policy": {"rights_reviewed": True, "allow_public_repo": "other/repo"}}):
        with pytest.raises(ValueError, match="Public"):
            transfer(packed, mocked, **kw)
    receipt = transfer(packed, mocked, rights_reviewed=True,
                       repo_policy={"rights_reviewed": True, "allow_public_repo": "owner/private"})
    assert receipt["revision"] == mocked.sha


@pytest.mark.parametrize("failure", ["network", "shard", "manifest", "inventory", "revision"])
def test_failed_upload_not_success(packed, mocked, failure):
    if failure == "network":
        mocked.fail = True
    elif failure == "shard":
        mocked.corrupt = True
    elif failure == "revision":
        mocked.sha = "main"
    elif failure == "manifest":
        original = mocked.snapshot
        def corrupt(**kw):
            result = original(**kw)
            for name in kw["allow_patterns"]:
                if name.endswith(h.MANIFEST):
                    (Path(kw["local_dir"]) / name).write_text('{}')
            return result
        mocked.snapshot = corrupt
    else:
        original = mocked.list_repo_files
        def extra(**kw):
            entries = original(**kw)
            return entries + ([str(Path(entries[0]).parent / "extra")] if entries else [])
        mocked.list_repo_files = extra
    with pytest.raises((ValueError, RuntimeError)):
        transfer(packed, mocked)


def test_symlink_destination(packed, tmp_path):
    (tmp_path / "link").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="Symlink"):
        h.restore_local(packed, tmp_path / "link" / "new", expected_kind="hf_model")


def test_atomic_publication_refuses_existing(tmp_path):
    src, dest = tmp_path / "src", tmp_path / "dest"
    src.mkdir()
    dest.mkdir()
    with pytest.raises(OSError):
        h.publish(src, dest)
    assert src.exists() and dest.exists()


def test_download_failure_no_destination(packed, mocked, tmp_path):
    receipt = transfer(packed, mocked)
    target = tmp_path / "failed"
    (mocked.remote / receipt["path"] / "model-00001-of-00002.safetensors").write_bytes(b"damaged")
    with pytest.raises(ValueError):
        h.download("owner/private", receipt["revision"], receipt["path"], target,
                   expected_kind="hf_model", api=mocked, snapshot=mocked.snapshot)
    assert not target.exists()


def test_trust_and_kind_before_network(mocked, tmp_path):
    prefix = "stages/s/runs/r/checkpoints/c/torch_dist_training"
    with pytest.raises(ValueError, match="trusted_inputs"):
        h.download("owner/private", "a" * 40, prefix, tmp_path / "out",
                   expected_kind="torch_dist_training", api=mocked, snapshot=mocked.snapshot)
    with pytest.raises(ValueError, match="kind"):
        h.download("owner/private", "a" * 40, prefix, tmp_path / "out",
                   expected_kind="hf_model", api=mocked, snapshot=mocked.snapshot)
    assert not mocked.calls


def test_malicious_download_inventory(packed, mocked, tmp_path):
    receipt = transfer(packed, mocked)
    manifest = mocked.remote / receipt["path"] / h.MANIFEST
    m = json.loads(manifest.read_text())
    m["files"]["../escape"] = {"size": 1, "sha256": "a" * 64}
    manifest.write_text(json.dumps(m))
    target = tmp_path / "unsafe"
    with pytest.raises(ValueError, match="Unsafe"):
        h.download("owner/private", receipt["revision"], receipt["path"], target,
                   expected_kind="hf_model", api=mocked, snapshot=mocked.snapshot)
    assert not target.exists()


def test_offline_cli_no_sdk_or_token(packed, tmp_path, monkeypatch):
    import subprocess
    import sys
    monkeypatch.delenv("HF_TOKEN", raising=False)
    script = Path(__file__).resolve().parents[1] / "scripts" / "hf_checkpoint.py"
    result = subprocess.run([sys.executable, str(script), "upload", "--source", str(packed),
                             "--repo-id", "owner/private", "--dry-run"], capture_output=True, text=True)
    assert result.returncode == 0
    assert json.loads(result.stdout)["remote_upload_exercised"] is False
    restore = subprocess.run([sys.executable, str(script), "restore-local", "--source", str(packed),
                              "--output", str(tmp_path / "cli-restore"), "--kind", "hf_model"],
                             capture_output=True, text=True)
    assert restore.returncode == 0
    assert h.files(tmp_path / "cli-restore") == h.files(packed)


def test_cli_exception_does_not_echo_token(packed, monkeypatch, capsys):
    import importlib.util
    import sys
    script = Path(__file__).resolve().parents[1] / "scripts" / "hf_checkpoint.py"
    spec = importlib.util.spec_from_file_location("hf_checkpoint_cli", script)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load CLI test module")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    def failure(*args, **kwargs):
        raise RuntimeError("Authorization: Bearer PRIVATE_SECRET")
    monkeypatch.setattr(cli, "upload", failure)
    monkeypatch.setattr(sys, "argv", [str(script), "upload", "--source", str(packed), "--repo-id", "owner/private"])
    assert cli.main() == 1
    captured = capsys.readouterr()
    assert "PRIVATE_SECRET" not in captured.out + captured.err


@pytest.mark.parametrize("role", sorted(h.ROLES))
def test_missing_resume_role(tmp_path, role):
    # Presence checks cannot assert semantic optimizer validity, but each
    # required producer declaration must at least exist.
    src = tmp_path / "source"
    src.mkdir()
    (src / "state").write_bytes(b"opaque fixture")
    manifest = {"schema_version": 1, "kind": "torch_dist_training", "stage": "s",
                "run": "r", "checkpoint": "c", "files": h.files(src),
                "components": {r: ["state"] for r in h.ROLES if r != role}}
    with pytest.raises(ValueError, match="all resume components"):
        h.validate_contract(src, manifest)


def test_restore_output_inside_source(packed):
    with pytest.raises(ValueError, match="outside source"):
        h.restore_local(packed, packed / "nested" / "restore", expected_kind="hf_model")
    assert not (packed / "nested").exists()


def test_download_overwrite_before_network(packed, mocked):
    with pytest.raises(ValueError, match="already exists"):
        h.download("owner/private", "a" * 40, "stages/s/runs/r/checkpoints/c/hf_model", packed,
                   expected_kind="hf_model", api=mocked, snapshot=mocked.snapshot)
    assert not mocked.calls
