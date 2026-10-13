"""Bounded CPU data acquisition. Outputs are flat uint32, NOT Megatron indexes."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import struct
import tempfile

SOURCE = {
    "repository": "karpathy/climbmix-400b-shuffle",
    "revision": "915333b4f8b8684f39aeaafea600fea6f43fb703",
    "original_repository": "nvidia/Nemotron-ClimbMix",
    "original_revision": "5eaa64b9c0c85b7f56af01d7dffdb0795816b12b",
    "license": "CC-BY-NC-4.0",
    "derivative_card_license": "MIT (does not waive upstream conditions)",
    "attribution": "NVIDIA Nemotron-ClimbMix; Karpathy seed-42 shuffled GPT-2 detokenized repackaging",
    "use": "noncommercial educational only",
}
SUPER = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"
SUPER_REVISION = "2dc98e2afe4face0e4ce40972a915c45368bd34a"
GPT2_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"


def positive(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be an explicit positive integer")
    return value


def validate_config(config):
    """Fail closed before downloads or tokenizer loading."""
    c = dict(config)
    positive(c.get("token_quota"), "token_quota")
    positive(c.get("max_shards"), "max_shards")
    if c.get("split") not in {"train", "test", "validation"}:
        raise ValueError("split must be train, test, or validation")
    if c.get("purpose") not in {"smoke", "training"}:
        raise ValueError("purpose must be smoke or training")
    if c.get("quota_policy", "prefix") not in {"prefix", "complete"}:
        raise ValueError("quota_policy must be prefix or complete")
    if c.get("rights_acknowledged") is not True:
        raise ValueError("explicit rights_acknowledged=true required")
    if c.get("mode") == "climbmix":
        if c.get("source") != SOURCE:
            raise ValueError("source must exactly match the pinned NC source manifest")
        shards = c.get("shards")
        if not isinstance(shards, list) or not shards or len(shards) > c["max_shards"]:
            raise ValueError("explicit nonempty shards within max_shards required")
        if len(set(shards)) != len(shards):
            raise ValueError("duplicate shard selection")
        for name in shards:
            if not isinstance(name, str) or not re.fullmatch(r"shard_\d{5}\.parquet", name) or int(name[6:11]) > 6542:
                raise ValueError(f"unknown shard: {name!r}")
            if c["split"] == "train" and name == "shard_06542.parquet":
                raise ValueError("validation shard forbidden in train")
            if c["split"] == "validation" and name != "shard_06542.parquet":
                raise ValueError("validation must use the reserved final shard")
        positive(c.get("max_download_bytes"), "max_download_bytes")
    elif c.get("mode") == "local_jsonl":
        files = c.get("files")
        if not isinstance(files, list) or not files or len(files) > c["max_shards"]:
            raise ValueError("explicit files within max_shards required")
        if len(set(files)) != len(files):
            raise ValueError("duplicate local files")
        if c.get("local_license") not in {"CC0-1.0", "CC-BY-NC-4.0"} or not c.get("local_attribution"):
            raise ValueError("known local_license and attribution required")
    else:
        raise ValueError("mode must be climbmix or local_jsonl")
    reservations = c.get("reserved_hashes")
    if not isinstance(reservations, dict) or set(reservations) != {"train", "test", "validation"}:
        raise ValueError("explicit reserved_hashes for train/test/validation required, even if empty")
    for hashes in reservations.values():
        if not isinstance(hashes, list) or any(not isinstance(h, str) or not re.fullmatch(r"[0-9a-f]{64}", h) for h in hashes):
            raise ValueError("reservations must be lists of SHA256 strings")
    model = c.get("tokenizer", SUPER)
    revision = c.get("tokenizer_revision", SUPER_REVISION if model == SUPER else GPT2_REVISION)
    if model not in {SUPER, "gpt2"} or (model == "gpt2" and c["purpose"] != "smoke"):
        raise ValueError("only pinned Super for training; gpt2 is smoke-only")
    if revision != (SUPER_REVISION if model == SUPER else GPT2_REVISION):
        raise ValueError("unknown tokenizer revision")
    c["tokenizer"], c["tokenizer_revision"] = model, revision
    for key, default in (("max_document_bytes", 16 * 1024 * 1024), ("max_input_bytes", 1024 * 1024 * 1024)):
        c[key] = positive(c.get(key, default), key)
    return c


def load_tokenizer(config):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(config["tokenizer"], revision=config["tokenizer_revision"], trust_remote_code=False)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def iter_records(c, cache, inputs):
    """Read selected files only; cache one remote shard at a time, with byte cap."""
    downloaded = 0
    names = c["shards"] if c["mode"] == "climbmix" else c["files"]
    for name in names:
        if c["mode"] == "local_jsonl":
            path = Path(name)
            size = path.stat().st_size
            if size > c["max_input_bytes"]:
                raise ValueError(f"local input exceeds max_input_bytes: {name}")
            inputs.append({"path": str(path.resolve()), "bytes": size, "sha256": sha256_file(path)})
            with path.open("rb") as handle:
                row = 0
                while True:
                    line = handle.readline(c["max_document_bytes"] + 1)
                    if not line:
                        break
                    if len(line) > c["max_document_bytes"]:
                        raise ValueError(f"JSONL line too large: {name}:{row}")
                    try:
                        record = json.loads(line)
                    except (ValueError, UnicodeError) as exc:
                        raise ValueError(f"malformed JSONL: {name}:{row}") from exc
                    if not isinstance(record, dict) or not isinstance(record.get("text"), str):
                        raise ValueError(f"text must be a string: {name}:{row}")
                    yield name, row, record["text"]
                    row += 1
        else:
            import requests
            import pyarrow.parquet as pq
            url = f"https://huggingface.co/datasets/{SOURCE['repository']}/resolve/{SOURCE['revision']}/{name}"
            path = cache / name
            try:
                with requests.get(url, stream=True, timeout=(15, 60)) as response:
                    response.raise_for_status()
                    length = response.headers.get("Content-Length")
                    if length and int(length) > c["max_download_bytes"] - downloaded:
                        raise ValueError("selected download exceeds byte budget")
                    with path.open("xb") as handle:
                        for block in response.iter_content(chunk_size=1024 * 1024):
                            downloaded += len(block)
                            if downloaded > c["max_download_bytes"]:
                                raise ValueError("stream exceeds max_download_bytes")
                            handle.write(block)
                inputs.append({"shard": name, "url": url, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
                parquet = pq.ParquetFile(path)
                if "text" not in parquet.schema.names:
                    raise ValueError("selected parquet has no text column")
                row = 0
                for batch in parquet.iter_batches(batch_size=64, columns=["text"]):
                    for text in batch.column(0).to_pylist():
                        if not isinstance(text, str):
                            raise ValueError(f"non-string text: {name}:{row}")
                        yield name, row, text
                        row += 1
            except Exception as exc:
                raise ValueError(f"selected-shard acquisition failed for {name}: {exc}") from exc
            finally:
                path.unlink(missing_ok=True)


def prepare_data(config, output_dir, tokenizer=None):
    """Atomically publish a fresh pack; injected tokenizers expose encode/eos_token_id."""
    c = validate_config(config)
    output = Path(output_dir).absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.with_name(output.name + ".lock")
    lock.mkdir()  # exclusive reservation across concurrent producers
    stage = None
    iterator = None
    try:
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"refusing to overwrite {output}")
        if c["token_quota"] < 2:
            raise ValueError("selection produced no accepted documents")
        stage = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
        cache = stage / "cache"
        cache.mkdir()
        tok = tokenizer if tokenizer is not None else load_tokenizer(c)
        eos = tok.eos_token_id
        if type(eos) is not int or not 0 <= eos < 2**32:
            raise ValueError("tokenizer must provide a uint32 EOS ID")
        reserved = {h for split, hashes in c["reserved_hashes"].items() if split != c["split"] for h in hashes}
        for manifest_path in c.get("reserved_manifests", []):
            with open(manifest_path, encoding="utf-8") as handle:
                manifest = json.load(handle)
            if manifest.get("split") not in {"train", "test", "validation"} or not isinstance(manifest.get("record_hashes"), list):
                raise ValueError("invalid reserved manifest")
            hashes = manifest["record_hashes"]
            if any(not isinstance(h, str) or not re.fullmatch(r"[0-9a-f]{64}", h) for h in hashes):
                raise ValueError("invalid reserved record hash")
            if manifest["split"] != c["split"]:
                reserved.update(hashes)
        counters = dict(rows_seen=0, empty=0, duplicate=0, reserved=0, oversized_for_quota=0,
                        tokenized_text_tokens=0, accepted_text_tokens=0, accepted_eos_tokens=0,
                        accepted_tokens=0, consumed_training_tokens=0, truncated_records=0)
        seen, records, inputs = set(), [], []
        iterator = iter_records(c, cache, inputs)
        with (stage / "tokens.u32").open("xb") as binary:
            for shard, row, text in iterator:
                counters["rows_seen"] += 1
                raw = text.encode("utf-8")
                if len(raw) > c["max_document_bytes"]:
                    raise ValueError(f"document exceeds max_document_bytes: {shard}:{row}")
                if not text.strip():
                    counters["empty"] += 1
                    continue
                digest = hashlib.sha256(raw).hexdigest()
                if digest in reserved:
                    counters["reserved"] += 1
                    continue
                if digest in seen:
                    counters["duplicate"] += 1
                    continue
                seen.add(digest)
                ids = list(tok.encode(text, add_special_tokens=False))
                if any(type(i) is not int or not 0 <= i < 2**32 for i in ids):
                    raise ValueError("tokenizer returned invalid uint32 token IDs")
                counters["tokenized_text_tokens"] += len(ids)
                if not ids:
                    counters["empty"] += 1
                    continue
                remaining = c["token_quota"] - counters["accepted_tokens"]
                truncated = len(ids) + 1 > remaining
                if truncated and (c.get("quota_policy", "prefix") == "complete" or remaining < 2):
                    counters["oversized_for_quota"] += 1
                    continue
                original_count = len(ids)
                if truncated:
                    ids = ids[:remaining - 1]
                    counters["truncated_records"] += 1
                packed_ids = ids + [eos]
                offset = counters["accepted_tokens"]
                payload = struct.pack(f"<{len(packed_ids)}I", *packed_ids)
                binary.write(payload)
                records.append({"source_file": shard, "row": row, "text_sha256": digest,
                                "source_utf8_bytes": len(raw), "token_offset": offset,
                                "token_count": len(packed_ids), "text_tokens": len(ids), "eos_tokens": 1,
                                "original_text_tokens": original_count, "prefix_truncated": truncated,
                                "token_content_sha256": hashlib.sha256(payload).hexdigest()})
                counters["accepted_text_tokens"] += len(ids)
                counters["accepted_eos_tokens"] += 1
                counters["accepted_tokens"] += len(packed_ids)
                # Stop before advancing the lazy iterator: every document needs text + EOS.
                if c["token_quota"] - counters["accepted_tokens"] < 2:
                    break
        iterator.close()
        iterator = None
        cache.rmdir()
        if not records:
            raise ValueError("selection produced no accepted documents")
        backend = getattr(tok, "backend_tokenizer", None)
        tokenizer_fingerprint = hashlib.sha256(backend.to_str().encode("utf-8")).hexdigest() if backend is not None else None
        manifest = {"format": "nemochat-flat-uint32-le-v1", "split": c["split"],
                    "config": c, "source": SOURCE if c["mode"] == "climbmix" else {
                        "license": c["local_license"], "attribution": c["local_attribution"]},
                    "tokenizer": {"model": c["tokenizer"], "revision": c["tokenizer_revision"],
                                  "trust_remote_code": False, "eos_token_id": eos,
                                  "backend_json_sha256": tokenizer_fingerprint,
                                  "injected_test_interface": tokenizer is not None},
                    "counters": counters, "quota_reached": counters["accepted_tokens"] == c["token_quota"],
                    "quota_includes_eos": True, "inputs": inputs,
                    "record_hashes": [r["text_sha256"] for r in records],
                    "tokens_sha256": sha256_file(stage / "tokens.u32"),
                    "dedup_scope": "exact UTF-8 text within visited selected rows and supplied reservations only"}
        for filename, value in (("records.json", records), ("manifest.json", manifest)):
            with (stage / filename).open("x", encoding="utf-8") as handle:
                json.dump(value, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
        manifest["records_sha256"] = sha256_file(stage / "records.json")
        with (stage / "manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"refusing to overwrite {output}")
        os.rename(stage, output)
        stage = None
        return manifest
    finally:
        if iterator is not None:
            iterator.close()
        if stage is not None:
            shutil.rmtree(stage)
        lock.rmdir()
