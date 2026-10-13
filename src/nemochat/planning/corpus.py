"""Explicit shard strata: read selected local files, tokenize only a bounded sample.

A reservoir is uniform only after scanning its entire declared population. Bounds
fail closed; stopping at the first rows never produces an estimate. No downloads.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import random
import statistics

from nemochat.data.climbmix import SUPER, SUPER_REVISION, GPT2_REVISION


def positive_int(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def validate_manifest(manifest):
    for key in ("max_files", "max_input_bytes", "max_document_bytes",
                "max_sample_documents", "max_sample_text_bytes", "max_sample_tokens"):
        positive_int(manifest.get(key), key)
    if type(manifest.get("seed")) is not int:
        raise ValueError("seed must be an integer")
    if manifest.get("purpose") not in {"estimate", "smoke"}:
        raise ValueError("purpose must be estimate or smoke")
    model = manifest.get("tokenizer", SUPER)
    revision = manifest.get("tokenizer_revision", SUPER_REVISION)
    allowed = {SUPER: SUPER_REVISION}
    if manifest["purpose"] == "smoke":
        allowed["gpt2"] = GPT2_REVISION
    if allowed.get(model) != revision:
        raise ValueError("use pinned Super for estimation; pinned GPT-2 is smoke-only")
    scope = manifest.get("scope")
    if not isinstance(scope, dict) or any(not isinstance(scope.get(k), str) or not scope[k].strip()
                                         for k in ("population", "shard_selection", "heldout", "filter", "dedup")):
        raise ValueError("scope must describe population, shard_selection, heldout, filter, dedup")
    strata = manifest.get("strata")
    if not isinstance(strata, list) or not strata or len(strata) > manifest["max_files"]:
        raise ValueError("nonempty strata within max_files required")
    names, paths, quota = set(), set(), 0
    for s in strata:
        if not isinstance(s, dict):
            raise ValueError("strata must be objects")
        for k in ("source", "shard", "path"):
            if not isinstance(s.get(k), str) or not s[k]:
                raise ValueError(f"stratum {k} required")
        name = (s["source"], s["shard"])
        path = str(Path(s["path"]).resolve())
        if name in names or path in paths:
            raise ValueError("duplicate stratum or file")
        names.add(name)
        paths.add(path)
        positive_int(s.get("population_documents"), "population_documents")
        n = positive_int(s.get("sample_documents"), "sample_documents")
        if n > s["population_documents"]:
            raise ValueError("sample exceeds population")
        quota += n
        if s.get("mode") not in {"reservoir", "representative"}:
            raise ValueError("mode must be reservoir or representative")
        if s["mode"] == "representative" and not s.get("sampling_provenance"):
            raise ValueError("representative samples need user sampling_provenance")
        if not isinstance(s.get("text_field", "text"), str) or not s.get("text_field", "text"):
            raise ValueError("text_field must be a nonempty string")
        if Path(path).suffix not in {".jsonl", ".parquet"}:
            raise ValueError("only local JSONL or Parquet files supported")
    if quota > manifest["max_sample_documents"]:
        raise ValueError("sample document quota exceeded")
    return model, revision


def iter_texts(path, field, max_document_bytes, max_file_bytes):
    """JSONL reads are line-bounded; Parquet uses one-row batches (decoder caveat)."""
    if path.suffix == ".jsonl":
        with path.open("rb") as handle:
            read_bytes = 0
            while True:
                # JSON escaping may use six bytes per text character. Still bound lines.
                raw = handle.readline(max_document_bytes * 6 + 4097)
                if not raw:
                    break
                read_bytes += len(raw)
                if read_bytes > max_file_bytes:
                    raise ValueError("input file grew beyond declared file byte bound")
                if len(raw) > max_document_bytes * 6 + 4096:
                    raise ValueError("JSONL record exceeds bounded line size")
                row = json.loads(raw)
                if not isinstance(row, dict) or not isinstance(row.get(field), str):
                    raise ValueError("each JSONL document needs a string text field")
                text = row[field]
                if len(text.encode("utf-8")) > max_document_bytes:
                    raise ValueError("document exceeds max_document_bytes")
                yield text
    else:
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ValueError("Parquet requires optional pyarrow") from exc
        for batch in pq.ParquetFile(path).iter_batches(batch_size=1, columns=[field]):
            for row in batch.to_pylist():
                text = row.get(field)
                if not isinstance(text, str):
                    raise ValueError("Parquet text must be a string")
                if len(text.encode("utf-8")) > max_document_bytes:
                    raise ValueError("document exceeds max_document_bytes")
                yield text


def reservoir_sample(texts, size, rng):
    """Algorithm R: equal inclusion probability size / count, without replacement."""
    positive_int(size, "reservoir size")
    sample, count = [], 0
    for count, text in enumerate(texts, 1):
        item = (count, text)
        if count <= size:
            sample.append(item)
        else:
            slot = rng.randrange(count)
            if slot < size:
                sample[slot] = item
    return sorted(sample), count


def summary(values):
    ordered = sorted(values)
    def quantile(p):
        index = (len(ordered) - 1) * p
        lo, hi = math.floor(index), math.ceil(index)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)
    return {"sum": sum(values), "mean": statistics.mean(values), "min": min(values),
            "max": max(values), "p50": quantile(.5), "p95": quantile(.95),
            "p99": quantile(.99)}


def estimate_corpus(manifest, tokenizer=None):
    """Estimate only the explicitly named shard populations, never the whole source.

    Representative manifests supply already sampled rows; their provenance is a
    user attestation, not verified random sampling. No CI is assigned to them.
    """
    model, revision = validate_manifest(manifest)
    input_bytes, file_sizes = 0, {}
    for s in manifest["strata"]:
        path = Path(s["path"])
        if not path.is_file():
            raise ValueError(f"missing local file: {path}")
        file_sizes[s["path"]] = path.stat().st_size
        input_bytes += file_sizes[s["path"]]
    if input_bytes > manifest["max_input_bytes"]:
        raise ValueError("selected files exceed max_input_bytes; no prefix estimate made")
    injected = tokenizer is not None
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model, revision=revision,
                                                  trust_remote_code=False, local_files_only=True)
    eos = tokenizer.eos_token_id
    if type(eos) is not int or not 0 <= eos < 2**32:
        raise ValueError("tokenizer must have uint32 EOS")
    results, used_bytes, used_tokens = [], 0, 0
    for s in manifest["strata"]:
        # Independent, order-stable per-stratum randomness, not Python hash().
        seed = hashlib.sha256(json.dumps([manifest["seed"], s["source"], s["shard"]], ensure_ascii=False).encode()).digest()
        rows = iter_texts(Path(s["path"]), s.get("text_field", "text"), manifest["max_document_bytes"], file_sizes[s["path"]])
        if s["mode"] == "reservoir":
            sample, count = reservoir_sample(rows, s["sample_documents"], random.Random(seed))
            if count != s["population_documents"]:
                raise ValueError("scanned population differs from declared population_documents")
        else:
            sample = []
            for count, text in enumerate(rows, 1):
                if count > s["sample_documents"]:
                    raise ValueError("representative file must contain exactly sample_documents, not a prefix")
                sample.append((count, text))
            if len(sample) != s["sample_documents"]:
                raise ValueError("representative sample count mismatch")
        lengths, byte_lengths, sampled = [], [], []
        for row, text in sample:
            raw = text.encode("utf-8")
            used_bytes += len(raw)
            if used_bytes > manifest["max_sample_text_bytes"]:
                raise ValueError("sample text byte quota exceeded")
            ids = tokenizer.encode(text, add_special_tokens=False)
            if any(type(i) is not int or not 0 <= i < 2**32 for i in ids):
                raise ValueError("invalid tokenizer token ID")
            used_tokens += len(ids) + 1
            if used_tokens > manifest["max_sample_tokens"]:
                raise ValueError("sample token quota exceeded; no truncated-document estimate")
            lengths.append(len(ids))
            byte_lengths.append(len(raw))
            sampled.append({"row": row, "text_sha256": hashlib.sha256(raw).hexdigest(),
                            "text_tokens": len(ids), "eos_tokens": 1, "utf8_bytes": len(raw)})
        n, population = len(sample), s["population_documents"]
        # Shards are separate fixed strata. Never pool rows as IID across clusters.
        variance = None
        if s["mode"] == "reservoir":
            if n == population:
                variance = 0.0
            elif n >= 2:
                variance = population**2 * (1 - n / population) * statistics.variance(lengths) / n
        results.append({"source": s["source"], "shard": s["shard"], "path": s["path"],
                        "mode": s["mode"], "sampling_provenance": s.get("sampling_provenance"),
                        "population_documents": population, "sample_documents": n,
                        "text_lengths": summary(lengths), "utf8_byte_lengths": summary(byte_lengths),
                        "sample_eos_tokens": n, "sampled_rows": sampled,
                        "estimated_text_tokens": population * statistics.mean(lengths),
                        "estimated_eos_tokens": population,
                        "estimated_utf8_bytes": population * statistics.mean(byte_lengths),
                        "total_token_variance": variance})
    total = sum(r["estimated_text_tokens"] + r["estimated_eos_tokens"] for r in results)
    variances = [r["total_token_variance"] for r in results]
    ci = None
    if all(v is not None for v in variances):
        se = math.sqrt(sum(variances))
        ci = {"standard_error": se, "normal_approx_95": [max(0, total - 1.96 * se), total + 1.96 * se],
              "method": "independent fixed-shard strata SRS with finite population correction; approximate, not exact coverage"}
    backend = getattr(tokenizer, "backend_tokenizer", None)
    return {"schema": "nemochat-corpus-estimate-v1", "purpose": manifest["purpose"],
            "seed": manifest["seed"], "scope": manifest["scope"],
            "bounds": {key: manifest[key] for key in ("max_files", "max_input_bytes", "max_document_bytes", "max_sample_documents", "max_sample_text_bytes", "max_sample_tokens")},
            "limitations": ["Only declared shard populations; no inference to unselected shards or full source.",
                            "CI is conditional on fixed selected shards; not cluster-selection uncertainty.",
                            "Representative sample provenance is user-attested; no CI without a verified design.",
                            "Raw documents, including duplicates/empty text; no filter, dedup or heldout subtraction performed.",
                            "JSONL on-disk size is bounded; Parquet decoding may expand beyond compressed file size."],
            "tokenizer": {"model": model, "revision": revision, "local_files_only": True,
                          "injected_test_interface": injected, "eos_token_id": eos,
                          "backend_json_sha256": hashlib.sha256(backend.to_str().encode()).hexdigest() if backend else None},
            "policy": {"add_special_tokens": False, "bos_tokens": 0, "eos_per_document": 1,
                       "eos_utf8_bytes": 0, "text_bytes": "UTF-8 source text only; EOS is a token, not source bytes"},
            "input_file_bytes": input_bytes, "sample_text_bytes": used_bytes,
            "sample_total_tokens": used_tokens, "strata": results,
            "estimated_population_documents": sum(r["population_documents"] for r in results),
            "estimated_text_tokens": sum(r["estimated_text_tokens"] for r in results),
            "estimated_eos_tokens": sum(r["estimated_eos_tokens"] for r in results),
            "estimated_total_tokens": total,
            "estimated_utf8_bytes": sum(r["estimated_utf8_bytes"] for r in results),
            "uncertainty": ci}
