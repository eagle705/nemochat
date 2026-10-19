"""Pinned Base teacher acquisition. Never imports repository model code."""
import fcntl
import hashlib
import json
import os
import re
import stat
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath

RAID_ROOT = Path("/raid")

REPO_ID = "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-Base-BF16"
REVISION = "4ba52e8ffa79e9693936c1a819ac89f551ceeef6"
WEIGHT_COUNT = 63
WEIGHT_BYTES = 248913061312
ALLOWED = {
    "config.json", "generation_config.json", "model.safetensors.index.json",
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "added_tokens.json", "tokenizer.model", "vocab.json", "merges.txt",
    "chat_template.jinja", "preprocessor_config.json", "processor_config.json",
    "README.md", "LICENSE", "LICENSE.md", "LICENSE.txt", "NOTICE", "NOTICE.txt",
}


def build_manifest(metadata):
    """Return the sorted data-only allowlist from the saved Hub API response."""
    if metadata.get("id") != REPO_ID or metadata.get("sha") != REVISION:
        raise ValueError("metadata repository/revision does not match pinned teacher")
    if metadata.get("modelId", REPO_ID) != REPO_ID:
        raise ValueError("metadata modelId disagrees with pinned teacher")
    manifest, seen = [], set()
    for s in metadata.get("siblings", []):
        name = s.get("rfilename")
        if (not isinstance(name, str) or not name or "\\" in name
                or any(part in {"", ".", ".."} for part in name.split("/"))
                or PurePosixPath(name).is_absolute() or ":" in name
                or any(ord(c) < 32 for c in name) or name in seen):
            raise ValueError("unsafe or duplicate metadata path")
        seen.add(name)
        if name not in ALLOWED and not re.fullmatch(r"model-\d{5}-of-00063\.safetensors", name):
            continue
        size = s.get("size")
        if type(size) is not int or size < 0:
            raise ValueError("missing/invalid file length")
        lfs = s.get("lfs")
        digest = lfs.get("sha256") if isinstance(lfs, dict) else None
        if lfs is not None or name.endswith(".safetensors"):
            if (not isinstance(lfs, dict) or not isinstance(digest, str)
                    or not re.fullmatch(r"[0-9a-f]{64}", digest) or lfs.get("size") != size):
                raise ValueError("missing/invalid authoritative LFS SHA256 or length")
        blob = s.get("blobId")
        if not digest and (not isinstance(blob, str) or not re.fullmatch(r"[0-9a-f]{40}", blob)):
            raise ValueError("missing authoritative Git blob digest")
        manifest.append({"path": name, "size": size, "sha256": digest, "git_blob": blob})
    weights = [s for s in manifest if s["path"].endswith(".safetensors")]
    expected = {f"model-{i:05d}-of-00063.safetensors" for i in range(1, 64)}
    if ({s["path"] for s in weights} != expected
            or sum(s["size"] for s in weights) != WEIGHT_BYTES):
        raise ValueError("incomplete pinned weight inventory")
    required = {"config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json", "README.md"}
    if not required <= {s["path"] for s in manifest}:
        raise ValueError("incomplete teacher support inventory")
    return sorted(manifest, key=lambda item: item["path"])


def verify_file(path, entry):
    """Stream-check length and authoritative digest, without loading tensors."""
    if path.is_symlink() or not path.is_file():
        raise ValueError("expected regular non-symlink file")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("expected regular file")
        if before.st_size != entry["size"]:
            raise ValueError("file length mismatch")
        sha = hashlib.sha256()
        git = hashlib.sha1(f"blob {entry['size']}\0".encode())
        while chunk := stream.read(8 * 1024 * 1024):
            sha.update(chunk)
            if not entry["sha256"]:
                git.update(chunk)
        after = os.fstat(stream.fileno())
    if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise ValueError("file changed during verification")
    if entry["sha256"] and sha.hexdigest() != entry["sha256"]:
        raise ValueError("SHA256 mismatch")
    if not entry["sha256"] and git.hexdigest() != entry["git_blob"]:
        raise ValueError("Git blob digest mismatch")
    return {**entry, "sha256": sha.hexdigest(), "verified": True}


def atomic_json(path, payload):
    """Publish a complete fsynced snapshot by same-directory atomic rename."""
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        temporary.unlink(missing_ok=True)


def download_teacher(metadata, output, receipt_dir, *, workers=4, attempts=3,
                     download_fn=None, sleep=time.sleep):
    """Acquire only manifest files; existing files are reverified, never trusted.

    Hub's current hf_hub_download resumes its .incomplete files automatically.
    Transport exceptions receive bounded retries; integrity mismatches fail closed.
    No exception text is persisted (upstream errors can contain credentials).
    """
    manifest = build_manifest(metadata)
    output, receipt_dir = Path(output).resolve(), Path(receipt_dir).resolve()
    if workers not in (4, 8) or type(attempts) is not int or not 1 <= attempts <= 5:
        raise ValueError("workers must be 4 or 8; attempts must be 1..5")
    for path in (output, receipt_dir):
        if not path.is_relative_to(RAID_ROOT.resolve()):
            raise ValueError("outputs and caches must be under /raid")
        path.mkdir(parents=True, exist_ok=True)
    home = output / ".hf-home"
    for root in (home, output / ".cache"):
        if root.is_symlink() or any(p.is_symlink() for p in root.rglob("*")):
            raise ValueError("cache paths must not contain symlinks")
    for key, path in {
        "HF_HOME": home, "HF_HUB_CACHE": home / "hub",
        "HUGGINGFACE_HUB_CACHE": home / "hub", "HF_XET_CACHE": home / "xet",
        "HF_ASSETS_CACHE": home / "assets", "HF_MODULES_CACHE": home / "modules",
        "XDG_CACHE_HOME": home / "xdg", "TMPDIR": home / "tmp",
    }.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    os.environ["HF_XET_HIGH_PERFORMANCE"] = "0"
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    os.environ["HF_XET_NUM_CONCURRENT_RANGE_GETS"] = "4"
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"
    os.environ["HF_HUB_ETAG_TIMEOUT"] = "30"
    if download_fn is None:
        from huggingface_hub import constants, hf_hub_download
        # Hub constants are captured at import, including the native Xet cache.
        # Do not silently accept a previously imported home-filesystem cache.
        expected_caches = {"HF_HOME": home, "HF_HUB_CACHE": home / "hub",
                           "HF_XET_CACHE": home / "xet", "HF_ASSETS_CACHE": home / "assets"}
        if (any(Path(getattr(constants, key)).resolve() != value for key, value in expected_caches.items())
                or constants.HF_HUB_ENABLE_HF_TRANSFER):
            raise RuntimeError("Hub imported with unsafe caches; run this CLI in a fresh process")
        download_fn = hf_hub_download
    run_id = f"{time.time_ns()}-{uuid.uuid4().hex}"
    lock = threading.Lock()
    complete, sequence = {}, 0

    def progress(status, event, **details):
        nonlocal sequence
        with lock:
            sequence += 1
            payload = {
                "repo_id": REPO_ID, "revision": REVISION, "run_id": run_id,
                "output": str(output), "status": status, "event": event,
                "sequence": sequence, "timestamp_ns": time.time_ns(),
                "expected_files": len(manifest), "complete_files": len(complete),
                "expected_bytes": sum(f["size"] for f in manifest),
                "verified_bytes": sum(f["size"] for f in complete.values()),
                "files": [complete[n] for n in sorted(complete)],
                "manifest": manifest, **details,
            }
            atomic_json(receipt_dir / f"progress-{run_id}-{sequence:06d}.json", payload)
            atomic_json(receipt_dir / "progress.json", payload)
            return payload

    def acquire(entry):
        path = output / entry["path"]
        if not path.exists() and not path.is_symlink():
            for attempt in range(1, attempts + 1):
                try:
                    downloaded = download_fn(
                        repo_id=REPO_ID, revision=REVISION, filename=entry["path"],
                        repo_type="model", local_dir=str(output), cache_dir=str(home / "hub"),
                        token=False, etag_timeout=30, endpoint="https://huggingface.co",
                    )
                    if Path(downloaded).absolute() != path.absolute():
                        raise ValueError("unexpected download destination")
                    break
                except ValueError:
                    raise
                except Exception as exc:
                    progress("running", "retry", path=entry["path"], attempt=attempt,
                             error_type=type(exc).__name__)
                    if attempt == attempts:
                        raise RuntimeError("teacher acquisition failed after bounded retries") from None
                    sleep(min(2 ** (attempt - 1), 30))
        verified = verify_file(path, entry)
        with lock:
            complete[entry["path"]] = verified
        progress("running", "verified", path=entry["path"])

    with (output / ".teacher-download.lock").open("a") as output_lock:
        fcntl.flock(output_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (receipt_dir / ".teacher-download.lock").open("a") as receipt_lock:
            fcntl.flock(receipt_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            final_path = receipt_dir / "receipt.json"
            if final_path.exists():
                final_path.rename(receipt_dir / f"receipt-previous-{run_id}.json")
            progress("running", "started")
            try:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    futures = [pool.submit(acquire, entry) for entry in manifest]
                    for future in as_completed(futures):
                        future.result()
                if len(complete) != len(manifest):
                    raise RuntimeError("incomplete verified file inventory")
            except BaseException as exc:
                progress("failed", "failed", error_type=type(exc).__name__)
                raise RuntimeError("teacher acquisition failed; see sanitized progress.json") from None
            result = progress("complete", "complete")
            atomic_json(final_path, result)
            return result


def main(argv=None):
    """CLI intended for a parent-owned foreground/systemd supervisor."""
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True, help="Saved complete pinned Hub metadata JSON")
    parser.add_argument("--output", type=Path, required=True, help="Teacher-only destination under /raid")
    parser.add_argument("--receipt-dir", type=Path, required=True, help="Progress and final receipt directory under /raid")
    parser.add_argument("--workers", type=int, choices=(4, 8), default=4)
    parser.add_argument("--attempts", type=int, choices=range(1, 6), default=3)
    args = parser.parse_args(argv)
    try:
        metadata = json.loads(args.metadata.read_text())
        result = download_teacher(metadata, args.output, args.receipt_dir,
                                  workers=args.workers, attempts=args.attempts)
    except Exception as exc:
        # Never expose raw upstream exception text or request URLs.
        parser.exit(1, f"teacher acquisition failed ({type(exc).__name__}); inspect sanitized progress if created\n")
    print(json.dumps({"status": result["status"], "complete_files": result["complete_files"],
                      "expected_files": result["expected_files"], "receipt": str(args.receipt_dir / "receipt.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

