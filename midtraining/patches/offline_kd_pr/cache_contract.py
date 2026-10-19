"""Isolated CPU reference cache. NOT the Megatron tar wire format or a trainer.

JSON deliberately avoids pickle. Hashes detect accidental corruption, not forgery.
Only continuous-no-reset samples and float32/float64 caches are supported here.
"""
import copy
import hashlib
import json
import math
from pathlib import Path
import re

import torch

META_KEYS = set("schema_version teacher_repo teacher_revision teacher_view_hash tokenizer_hash dataset_revision document_ids sample_id sample_order epoch packing_policy segment_ids position_ids loss_mask label_shift temperature vocab_size top_k dtype normalization".split())
RECORD_KEYS = set("metadata input_ids input_ids_hash prediction_positions target_ids token_ids logprobs tail_mass payload_sha256".split())


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def token_hash(input_ids):
    return _digest(input_ids)


def _integer(value, minimum=0):
    return type(value) is int and value >= minimum


def _validate_metadata(meta, input_ids):
    if not isinstance(meta, dict) or set(meta) != META_KEYS:
        raise ValueError("missing or unknown metadata fields")
    if not isinstance(input_ids, list) or len(input_ids) < 2:
        raise ValueError("input_ids must include at least one next-token target")
    if not _integer(meta["vocab_size"], 2) or not all(_integer(i) and i < meta["vocab_size"] for i in input_ids):
        raise ValueError("invalid token IDs or vocabulary")
    if type(meta["schema_version"]) is not int or meta["schema_version"] != 1 or type(meta["label_shift"]) is not int or meta["label_shift"] != 1:
        raise ValueError("unsupported schema or label shift")
    for key, length in [("teacher_revision", 40), ("teacher_view_hash", 64), ("tokenizer_hash", 64)]:
        if not isinstance(meta[key], str) or not re.fullmatch(r"[0-9a-f]{" + str(length) + r"}", meta[key]):
            raise ValueError("invalid " + key)
    for key in ("teacher_repo", "dataset_revision", "sample_id"):
        if not isinstance(meta[key], str) or not meta[key] or meta[key].strip() != meta[key]:
            raise ValueError("invalid " + key)
    if not isinstance(meta["document_ids"], list) or not meta["document_ids"] or not all(isinstance(x, str) and x for x in meta["document_ids"]):
        raise ValueError("missing document IDs")
    if not all(_integer(meta[k]) for k in ("epoch", "sample_order")):
        raise ValueError("invalid epoch or sample order")
    if meta["packing_policy"] != "continuous-no-reset":
        raise ValueError("reset packing not implemented")
    if not isinstance(meta["segment_ids"], list) or len(meta["segment_ids"]) != len(input_ids) or not all(_integer(i) for i in meta["segment_ids"]):
        raise ValueError("invalid segment alignment")
    if not isinstance(meta["position_ids"], list) or not all(_integer(i) for i in meta["position_ids"]) or meta["position_ids"] != list(range(len(input_ids) - 1)):
        raise ValueError("invalid position IDs")
    mask = meta["loss_mask"]
    if not isinstance(mask, list) or len(mask) != len(input_ids) - 1 or not all(type(x) is int and x in (0, 1) for x in mask) or not any(mask):
        raise ValueError("invalid or empty loss mask")
    t = meta["temperature"]
    if type(t) not in (int, float) or not math.isfinite(t) or t <= 0:
        raise ValueError("invalid temperature")
    if not _integer(meta["top_k"], 1) or meta["top_k"] > meta["vocab_size"]:
        raise ValueError("invalid top_k")
    if meta["dtype"] not in ("float64", "float32") or meta["normalization"] != "full-vocabulary-log-softmax-at-temperature":
        raise ValueError("unsupported precision or normalizer")


def _validate_record(record, expected_metadata, input_ids):
    _validate_metadata(expected_metadata, input_ids)
    if not isinstance(record, dict) or set(record) != RECORD_KEYS:
        raise ValueError("missing or unknown cache fields")
    _validate_metadata(record["metadata"], record["input_ids"])
    if record["metadata"] != expected_metadata:
        raise ValueError("cache lineage mismatch")
    if record["input_ids"] != input_ids or record["input_ids_hash"] != token_hash(input_ids):
        raise ValueError("input ID mismatch")
    if any(not isinstance(record[key], list) or not all(_integer(i) for i in record[key]) for key in ("target_ids", "prediction_positions")):
        raise ValueError("target IDs and prediction positions must be exact integers")
    if record["target_ids"] != input_ids[1:] or record["prediction_positions"] != list(range(len(input_ids) - 1)):
        raise ValueError("label shift or position mismatch")
    try:
        payload = {k: v for k, v in record.items() if k != "payload_sha256"}
        if _digest(payload) != record["payload_sha256"]:
            raise ValueError("cache checksum mismatch")
        ids = record["token_ids"]
        scores = torch.tensor(record["logprobs"], dtype=torch.float64, device="cpu")
        tail = torch.tensor(record["tail_mass"], dtype=torch.float64, device="cpu")
        shape = (len(input_ids) - 1, expected_metadata["top_k"])
        if tuple(scores.shape) != shape or tuple(tail.shape) != (shape[0],) or len(ids) != shape[0]:
            raise ValueError("cache tensor shape mismatch")
        for row in ids:
            if len(row) != shape[1] or len(set(row)) != shape[1] or not all(_integer(i) and i < expected_metadata["vocab_size"] for i in row):
                raise ValueError("invalid or duplicate cached token IDs")
        tolerance = 2e-6 if expected_metadata["dtype"] == "float32" else 1e-12
        mass = scores.exp().sum(-1)
        if not torch.isfinite(scores).all() or not torch.isfinite(tail).all() or (scores > 0).any() or (tail < -tolerance).any() or (tail > 1).any():
            raise ValueError("nonfinite or invalid probabilities")
        if not torch.allclose(mass + tail, torch.ones_like(mass), atol=tolerance, rtol=0):
            raise ValueError("tail mass disagrees with full-vocabulary normalization")
        if shape[1] == expected_metadata["vocab_size"] and not torch.allclose(tail, torch.zeros_like(tail), atol=tolerance, rtol=0):
            raise ValueError("full cache has nonzero omitted mass")
    except (TypeError, RuntimeError, OverflowError) as exc:
        raise ValueError("invalid cache payload") from exc


def make_record(metadata, input_ids, teacher_logits):
    _validate_metadata(metadata, input_ids)
    dtype = getattr(torch, metadata["dtype"])
    if teacher_logits.device.type != "cpu" or teacher_logits.dtype != dtype or tuple(teacher_logits.shape) != (len(input_ids) - 1, metadata["vocab_size"]) or not torch.isfinite(teacher_logits).all():
        raise ValueError("CPU logits dtype, shape or finiteness mismatch")
    scores = torch.log_softmax(teacher_logits.detach() / metadata["temperature"], dim=-1)
    values, indices = torch.topk(scores, metadata["top_k"], dim=-1)
    return _record_from_scores(metadata, input_ids, values, indices)


def _record_from_scores(metadata, input_ids, values, indices):
    record = {
        "metadata": copy.deepcopy(metadata),
        "input_ids": input_ids.copy(),
        "input_ids_hash": token_hash(input_ids),
        "prediction_positions": list(range(len(input_ids) - 1)),
        "target_ids": input_ids[1:],
        "token_ids": indices.tolist(),
        "logprobs": values.tolist(),
        "tail_mass": (1 - values.exp().sum(-1)).tolist(),
    }
    record["payload_sha256"] = _digest(record)
    _validate_record(record, metadata, input_ids)
    return record


def from_prompt_logprobs(metadata, input_ids, prompt_rows):
    """Convert plain ID->float prompt rows, NOT completion logprobs.

    Caller must extract Logprob.logprob from vLLM objects without retokenization.
    Values MUST be raw, full-vocabulary-normalized logprobs at temperature 1.
    A sparse response cannot be retempered exactly and is rejected if T != 1.
    This adapter cannot authenticate that the producer actually selected top-K.
    """
    _validate_metadata(metadata, input_ids)
    if not isinstance(prompt_rows, list) or len(prompt_rows) != len(input_ids) or prompt_rows[0] is not None:
        raise ValueError("prompt rows must include initial None and every target")
    dtype = getattr(torch, metadata["dtype"])
    full = all(isinstance(row, dict) and len(row) == metadata["vocab_size"] for row in prompt_rows[1:])
    if not full and metadata["temperature"] != 1:
        raise ValueError("sparse temperature conversion requires unavailable tail normalizer")
    values, indices = [], []
    for row in prompt_rows[1:]:
        if not isinstance(row, dict) or len(row) < metadata["top_k"] or not all(_integer(i) and i < metadata["vocab_size"] for i in row):
            raise ValueError("missing prompt position or invalid token IDs")
        if not all(type(v) in (int, float) and math.isfinite(v) and v <= 0 for v in row.values()):
            raise ValueError("prompt values must be finite logprobs, not logits")
        mass = sum(math.exp(v) for v in row.values())
        tolerance = 2e-6 if metadata["dtype"] == "float32" else 1e-12
        if mass > 1 + tolerance or (full and abs(mass - 1) > tolerance):
            raise ValueError("prompt values lack full-vocabulary normalization")
        if full:
            scores = torch.tensor([row[i] for i in range(metadata["vocab_size"])], dtype=dtype, device="cpu")
            scores = torch.log_softmax(scores / metadata["temperature"], -1)
            vals, ids = scores.topk(metadata["top_k"])
            values.append(vals)
            indices.append(ids)
        else:
            ids = sorted(row, key=lambda i: row[i], reverse=True)[:metadata["top_k"]]
            values.append(torch.tensor([row[i] for i in ids], dtype=dtype, device="cpu"))
            indices.append(torch.tensor(ids, dtype=torch.long, device="cpu"))
    return _record_from_scores(metadata, input_ids, torch.stack(values), torch.stack(indices))


def from_vllm_prompt_output(metadata, input_ids, prompt_rows):
    """Adapter for vLLM dict[int, Logprob] prompt outputs (not flat outputs).

    Select by the producer's rank so the extra ground-truth entry is not
    accidentally retained as K+1. No dependency on vLLM or model loading.
    """
    _validate_metadata(metadata, input_ids)
    if not isinstance(prompt_rows, list) or len(prompt_rows) != len(input_ids) or prompt_rows[0] is not None:
        raise ValueError("incomplete vLLM prompt output")
    converted = [None]
    for row in prompt_rows[1:]:
        if not isinstance(row, dict):
            raise ValueError("missing vLLM prompt position")
        ranks = [getattr(value, "rank", None) for value in row.values()]
        if not all(_integer(rank, 1) and rank <= metadata["vocab_size"] for rank in ranks):
            raise ValueError("missing or invalid vLLM token rank")
        selected = {i: value.logprob for i, value in row.items() if value.rank <= metadata["top_k"]}
        if len(selected) != metadata["top_k"]:
            raise ValueError("incomplete or ambiguous vLLM top-K ranks")
        # For T != 1, keep all available vocabulary entries to recompute Z(T).
        if metadata["temperature"] != 1 and len(row) == metadata["vocab_size"]:
            selected = {i: value.logprob for i, value in row.items()}
        converted.append(selected)
    return from_prompt_logprobs(metadata, input_ids, converted)


def cached_kl(student_logits, record, mode="full"):
    """Masked token-mean teacher||student KL; no implicit temperature-squared.

    sparse_global drops the teacher tail without renormalizing either support.
    conditional_topk normalizes BOTH distributions on the retained teacher IDs.
    The inspected native cached caller uses coarse_tail (one extra tail bucket),
    while the inspected ModelOpt top-K loss uses conditional_topk.
    """
    meta = record["metadata"]
    _validate_record(record, meta, record["input_ids"])
    if mode not in ("full", "sparse_global", "conditional_topk", "coarse_tail"):
        raise ValueError("unknown KL mode")
    if mode == "full" and meta["top_k"] != meta["vocab_size"]:
        raise ValueError("full KL requires all vocabulary entries")
    if student_logits.device.type != "cpu" or student_logits.dtype != getattr(torch, meta["dtype"]) or tuple(student_logits.shape) != (len(record["input_ids"]) - 1, meta["vocab_size"]) or not torch.isfinite(student_logits).all():
        raise ValueError("CPU student logits dtype, shape or finiteness mismatch")
    teacher = torch.tensor(record["logprobs"], dtype=student_logits.dtype, device="cpu")
    ids = torch.tensor(record["token_ids"], dtype=torch.long, device="cpu")
    student = torch.log_softmax(student_logits / meta["temperature"], -1).gather(-1, ids)
    if mode == "conditional_topk":
        teacher = torch.log_softmax(teacher, -1)
        student = torch.log_softmax(student, -1)
    losses = (teacher.exp() * (teacher - student)).sum(-1)
    if mode == "coarse_tail":
        # Match the inspected native helper's 1e-8 clamp, not a fabricated tail
        # distribution. A single tail bucket is still not full-vocabulary KL.
        teacher_tail = (1 - teacher.exp().sum(-1)).clamp(min=1e-8)
        student_tail = (1 - student.exp().sum(-1)).clamp(min=1e-8)
        losses = losses + teacher_tail * (teacher_tail.log() - student_tail.log())
    mask = torch.tensor(meta["loss_mask"], dtype=losses.dtype, device="cpu")
    return (losses * mask).sum() / mask.sum()


def write_cache(path, record):
    _validate_record(record, record["metadata"], record["input_ids"])
    with Path(path).open("x") as stream:
        json.dump(record, stream, allow_nan=False, sort_keys=True)
        stream.write("\n")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: " + key)
        result[key] = value
    return result


def read_cache(path, expected_metadata, input_ids):
    with Path(path).open() as stream:
        record = json.load(stream, object_pairs_hook=_unique_object)
    _validate_record(record, expected_metadata, input_ids)
    return record
