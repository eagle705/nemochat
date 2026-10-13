"""Private-by-default, immutable Hub transport for frozen checkpoint directories.

SHA256 inventories detect corruption, not authorship. Never deserialize untrusted
training bundles. The component declaration is the producer's resume contract,
not a proof of successful optimizer resumption on B300 hardware.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile

MANIFEST = "checkpoint_inventory.json"
KINDS = {"torch_dist_training", "hf_model"}
ROLES = {"model", "optimizer", "scheduler", "rng", "cursor", "tokenizer", "training_manifest", "environment"}
SHA = re.compile(r"[0-9a-f]{40}\Z")


def safe_name(value):
    if not isinstance(value, str) or not value or any(c in value for c in "\\\x00*?[]"):
        raise ValueError("Unsafe relative path")
    p = PurePosixPath(value)
    if p.is_absolute() or any(x in {"", ".", ".."} for x in value.split("/")):
        raise ValueError("Unsafe relative path")
    if ".git" in p.parts or any(a == ".cache" and b == "huggingface"
                                for a, b in zip(p.parts, p.parts[1:])):
        raise ValueError("Hub SDK-reserved namespace is forbidden")
    return value


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def files(root):
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Expected a real directory")
    out = {}
    for p in root.rglob("*"):
        if p.is_symlink() or (not p.is_dir() and not p.is_file()):
            raise ValueError("Symlinks and special files are forbidden")
        if p.is_file():
            name = safe_name(p.relative_to(root).as_posix())
            out[name] = {"size": p.stat().st_size, "sha256": digest(p)}
    return out


def read_json(path):
    def unique(pairs):
        d = {}
        for k, v in pairs:
            if k in d:
                raise ValueError("Duplicate JSON key")
            d[k] = v
        return d
    return json.loads(Path(path).read_text(), object_pairs_hook=unique)


def validate_contract(root, manifest):
    kind = manifest.get("kind")
    inventory = manifest.get("files")
    if manifest.get("schema_version") != 1 or kind not in KINDS or not isinstance(inventory, dict) or not inventory:
        raise ValueError("Invalid inventory schema or kind")
    for name, entry in inventory.items():
        safe_name(name)
        if name == MANIFEST or not isinstance(entry, dict) or set(entry) != {"size", "sha256"}:
            raise ValueError("Invalid file inventory")
        if type(entry["size"]) is not int or entry["size"] < 0 or not re.fullmatch(r"[0-9a-f]{64}", str(entry["sha256"])):
            raise ValueError("Invalid hash or size")
    for key in ("stage", "run", "checkpoint"):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", str(manifest.get(key, ""))):
            raise ValueError("Invalid stage/run/checkpoint identity")
    components = manifest.get("components", {})
    if kind == "torch_dist_training":
        if not isinstance(components, dict) or set(components) != ROLES:
            raise ValueError("Training bundle requires all resume components")
        for role, names in components.items():
            if not isinstance(names, list) or not names or any(n not in inventory or inventory[n]["size"] == 0 for n in names):
                raise ValueError("Missing declared resume component")
        model = components["model"]
        if not any(n.endswith(".metadata") for n in model) or not any(n.endswith(".distcp") for n in model):
            raise ValueError("torch_dist metadata and shards are required")
        for role in ("scheduler", "cursor", "training_manifest", "environment"):
            if not any(n.endswith(".json") and isinstance(read_json(Path(root) / n), dict) and read_json(Path(root) / n) for n in components[role]):
                raise ValueError("Resume JSON declarations must be nonempty objects")
    else:
        if components:
            raise ValueError("HF model export must not declare training components")
        if "config.json" not in inventory or not any(n.endswith(".safetensors") for n in inventory):
            raise ValueError("HF model export requires config and safetensors")
        if any(n.endswith((".pt", ".pth", ".bin", ".distcp", ".metadata")) or any(x in n.lower() for x in ("optimizer", "scheduler", "rng", "training_state")) for n in inventory):
            raise ValueError("Training/legacy artifacts forbidden in HF model export")
    for name in inventory:
        if name.endswith(".index.json"):
            index = read_json(Path(root) / name)
            weight_map = index.get("weight_map")
            if not isinstance(weight_map, dict) or not weight_map:
                raise ValueError("Invalid shard index")
            for shard in weight_map.values():
                safe_name(shard)
                target = str(PurePosixPath(name).parent / shard)
                if target not in inventory:
                    raise ValueError("Shard index references an unverified file")


def verify(root, expected_kind=None):
    root = Path(root)
    actual = files(root)
    if MANIFEST not in actual:
        raise ValueError("Missing inventory")
    manifest = read_json(root / MANIFEST)
    validate_contract(root, manifest)
    if expected_kind is not None and manifest["kind"] != expected_kind:
        raise ValueError("Incorrect checkpoint kind")
    actual.pop(MANIFEST)
    if actual != manifest["files"]:
        raise ValueError("Inventory does not match every checkpoint file")
    return manifest


def destination(path):
    if ".." in Path(path).parts:
        raise ValueError("Destination traversal is forbidden")
    p = Path(os.path.abspath(path))
    if p.exists() or p.is_symlink():
        raise ValueError("Destination already exists")
    for parent in p.parents:
        if parent.is_symlink():
            raise ValueError("Symlink destination ancestor")
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def publish(temp, dest):
    # Linux renameat2 NOREPLACE: atomic directory publication without a race
    # that could overwrite an empty directory created by another process.
    libc = ctypes.CDLL(None, use_errno=True)
    rename = getattr(libc, "renameat2", None)
    if rename is None:
        raise RuntimeError("Atomic no-replace publication requires Linux renameat2")
    if rename(-100, os.fsencode(temp), -100, os.fsencode(dest), 1):
        raise OSError(ctypes.get_errno(), "Atomic destination publication failed")


def pack(source, output, *, kind, stage, run, checkpoint, components=None):
    source = Path(source)
    inventory = files(source)
    if MANIFEST in inventory or not inventory:
        raise ValueError("Source must be a nonempty, un-packaged frozen directory")
    manifest = {"schema_version": 1, "kind": kind, "stage": stage, "run": run,
                "checkpoint": checkpoint, "components": components or {}, "files": inventory}
    validate_contract(source, manifest)
    if Path(os.path.abspath(output)).is_relative_to(source.resolve()):
        raise ValueError("Output must be outside source")
    dest = destination(output)
    with tempfile.TemporaryDirectory(prefix=".checkpoint-", dir=dest.parent) as tmp:
        staged = Path(tmp) / "payload"
        shutil.copytree(source, staged)
        (staged / MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        verify(staged, kind)
        publish(staged, dest)
    return manifest


def restore_local(source, output, *, expected_kind, trusted_inputs=False):
    if expected_kind not in KINDS:
        raise ValueError("Explicit valid checkpoint kind required")
    if expected_kind == "torch_dist_training" and not trusted_inputs:
        raise ValueError("Training restore requires explicit trusted_inputs")
    verify(source, expected_kind)
    if Path(os.path.abspath(output)).is_relative_to(Path(source).resolve()):
        raise ValueError("Output must be outside source")
    dest = destination(output)
    with tempfile.TemporaryDirectory(prefix=".restore-", dir=dest.parent) as tmp:
        staged = Path(tmp) / "payload"
        shutil.copytree(source, staged)
        verify(staged, expected_kind)
        publish(staged, dest)
    return str(dest)


def token():
    value = os.environ.get("HF_TOKEN")
    if not value or not value.strip():
        raise ValueError("HF_TOKEN must be explicitly provided in the environment")
    return value


def hub(api=None, snapshot=None):
    if api is None or snapshot is None:
        from huggingface_hub import HfApi, snapshot_download
        api = api or HfApi(token=token())
        snapshot = snapshot or snapshot_download
    return api, snapshot


def prefix_for(m):
    return f"stages/{m['stage']}/runs/{m['run']}/checkpoints/{m['checkpoint']}/{m['kind']}"


def check_prefix(prefix):
    safe_name(prefix)
    if not re.fullmatch(r"stages/[A-Za-z0-9_-]+/runs/[A-Za-z0-9_-]+/checkpoints/[A-Za-z0-9_-]+/(torch_dist_training|hf_model)", prefix):
        raise ValueError("Invalid checkpoint repository path")


def privacy(api, repo_id, credential, rights_reviewed, repo_policy):
    info = api.repo_info(repo_id=repo_id, repo_type="model", token=credential)
    if info.private is not True:
        if not (rights_reviewed and isinstance(repo_policy, dict) and repo_policy.get("rights_reviewed") is True and repo_policy.get("allow_public_repo") == repo_id):
            raise ValueError("Public repo requires explicit rights review and matching repository policy")
    return info


def download(repo_id, revision, prefix, output, *, expected_kind, trusted_inputs=False,
             rights_reviewed=False, repo_policy=None, api=None, snapshot=None,
             expected_manifest_hash=None):
    if not SHA.fullmatch(str(revision)):
        raise ValueError("Download requires an immutable 40-character commit SHA")
    check_prefix(prefix)
    if expected_kind not in KINDS or prefix.split("/")[-1] != expected_kind:
        raise ValueError("Incorrect checkpoint kind")
    if expected_kind == "torch_dist_training" and not trusted_inputs:
        raise ValueError("Training download requires explicit trusted_inputs")
    dest = destination(output)
    credential = token()
    api, snapshot = hub(api, snapshot)
    privacy(api, repo_id, credential, rights_reviewed, repo_policy)
    with tempfile.TemporaryDirectory(prefix=".hub-restore-", dir=dest.parent) as tmp:
        # local_dir makes real files instead of cache snapshot symlinks.
        snapshot(repo_id=repo_id, repo_type="model", revision=revision, token=credential,
                 local_dir=tmp, allow_patterns=[f"{prefix}/{MANIFEST}"])
        payload = Path(tmp) / prefix
        if payload.is_symlink() or (payload / MANIFEST).is_symlink():
            raise ValueError("Symlink manifest")
        if expected_manifest_hash and digest(payload / MANIFEST) != expected_manifest_hash:
            raise ValueError("Committed manifest hash mismatch")
        m = read_json(payload / MANIFEST)
        if m.get("schema_version") != 1 or m.get("kind") != expected_kind or prefix_for(m) != prefix:
            raise ValueError("Checkpoint manifest kind or identity mismatch")
        if not isinstance(m.get("files"), dict):
            raise ValueError("Invalid inventory")
        for name in m["files"]:
            safe_name(name)
        names = {f"{prefix}/{n}" for n in m["files"]} | {f"{prefix}/{MANIFEST}"}
        remote = {n for n in api.list_repo_files(repo_id=repo_id, repo_type="model", revision=revision, token=credential) if n.startswith(prefix + "/")}
        if names != remote:
            raise ValueError("Remote file inventory mismatch")
        snapshot(repo_id=repo_id, repo_type="model", revision=revision, token=credential,
                 local_dir=tmp, allow_patterns=sorted(names))
        verified = verify(payload, expected_kind)
        if expected_manifest_hash and digest(payload / MANIFEST) != expected_manifest_hash:
            raise ValueError("Committed manifest hash mismatch")
        if prefix_for(verified) != prefix:
            raise ValueError("Checkpoint identity mismatch")
        publish(payload, dest)
    return {"repo_id": repo_id, "revision": revision, "path": prefix,
            "kind": expected_kind, "manifest_sha256": digest(dest / MANIFEST)}


def upload(source, repo_id, *, rights_reviewed=False, repo_policy=None, api=None, snapshot=None):
    m = verify(source)
    credential = token()
    api, snapshot = hub(api, snapshot)
    info = privacy(api, repo_id, credential, rights_reviewed, repo_policy)
    prefix = prefix_for(m)
    if any(n.startswith(prefix + "/") for n in api.list_repo_files(repo_id=repo_id, repo_type="model", revision=info.sha, token=credential)):
        raise ValueError("Checkpoint repository path already exists; use a new identity")
    expected_hash = digest(Path(source) / MANIFEST)
    result = api.upload_folder(repo_id=repo_id, repo_type="model", folder_path=str(source),
                             path_in_repo=prefix, token=credential, parent_commit=info.sha,
                             allow_patterns=sorted(m["files"]) + [MANIFEST],
                             commit_message=f"Checkpoint {m['stage']}/{m['run']}/{m['checkpoint']}")
    revision = result.oid
    if not SHA.fullmatch(str(revision)):
        raise ValueError("Upload did not return an immutable commit SHA")
    # Remote commit may already exist if read-back fails: no verified receipt
    # does not mean rollback. A retry must inspect this checkpoint identity,
    # not overwrite it. Re-download and hash every shard, not just the manifest.
    with tempfile.TemporaryDirectory(prefix="checkpoint-readback-") as tmp:
        handle = download(repo_id, revision, prefix, Path(tmp) / "verified",
                          expected_kind=m["kind"], trusted_inputs=True,
                          rights_reviewed=rights_reviewed, repo_policy=repo_policy,
                          api=api, snapshot=snapshot, expected_manifest_hash=expected_hash)
    return handle
