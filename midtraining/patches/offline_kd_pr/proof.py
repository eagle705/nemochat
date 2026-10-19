"""Deterministic tiny CPU teacher calculation + serialized-cache KL proof.

Timing is deliberately local/synthetic; no native Bridge/vLLM or real teacher is
loaded. CPU JSON validation can be slower than a tiny matrix multiplication.
"""
import argparse
import json
from pathlib import Path
import platform
import statistics
from time import perf_counter

import torch

from cache_contract import make_record, write_cache, read_cache, cached_kl


def run_proof(output, repeats=12):
    if type(repeats) is not int or repeats < 1:
        raise ValueError("repeats must be positive")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rng = torch.Generator(device="cpu").manual_seed(20261017)
    hidden = torch.randn(64, 32, dtype=torch.float64, generator=rng, device="cpu")
    teacher_weight = torch.randn(32, 67, dtype=torch.float64, generator=rng, device="cpu") / 4
    student_weight = torch.randn(32, 67, dtype=torch.float64, generator=rng, device="cpu") / 4
    input_ids = (torch.arange(65, device="cpu") % 67).tolist()
    meta = {
        "schema_version": 1,
        "teacher_repo": "synthetic-linear-teacher-not-the-120B-model",
        "teacher_revision": "0" * 40,
        "teacher_view_hash": "0" * 64,
        "tokenizer_hash": "0" * 64,
        "dataset_revision": "deterministic-synthetic-fixture-v1",
        "document_ids": ["synthetic-document"],
        "sample_id": "synthetic-sample-0", "sample_order": 0, "epoch": 0,
        "packing_policy": "continuous-no-reset", "segment_ids": [0] * 65,
        "position_ids": list(range(64)), "loss_mask": [0 if i % 7 == 0 else 1 for i in range(64)],
        "label_shift": 1, "temperature": 2., "vocab_size": 67, "top_k": 67,
        "dtype": "float64", "normalization": "full-vocabulary-log-softmax-at-temperature",
    }
    full_path = output / "full-cache.json"
    start = perf_counter()
    teacher = hidden @ teacher_weight
    write_cache(full_path, make_record(meta, input_ids, teacher))
    dump_seconds = perf_counter() - start
    mask = torch.tensor(meta["loss_mask"], dtype=torch.float64, device="cpu")

    def online_step():
        w = student_weight.detach().clone().requires_grad_(True)
        q = torch.log_softmax((hidden @ teacher_weight) / meta["temperature"], -1)
        p = torch.log_softmax((hidden @ w) / meta["temperature"], -1)
        per_token = (q.exp() * (q - p)).sum(-1)
        loss = (per_token * mask).sum() / mask.sum()
        grad, = torch.autograd.grad(loss, w)
        return loss.detach(), grad

    def cached_step():
        w = student_weight.detach().clone().requires_grad_(True)
        record = read_cache(full_path, meta, input_ids)
        loss = cached_kl(hidden @ w, record)
        grad, = torch.autograd.grad(loss, w)
        return loss.detach(), grad

    online_loss, online_grad = online_step()
    cached_loss, cached_grad = cached_step()
    # Both paths are warmed before timed alternating pairs; JSON read/validation
    # is included every cached step. No optimizer, GPU sync, or model loading.
    online_times, cached_times = [], []
    for i in range(repeats):
        ordered = [(online_step, online_times), (cached_step, cached_times)]
        if i % 2:
            ordered.reverse()
        for fn, times in ordered:
            start = perf_counter()
            fn()
            times.append(perf_counter() - start)
    sparse_meta = dict(meta, top_k=32)
    sparse = make_record(sparse_meta, input_ids, teacher)
    write_cache(output / "top32-cache.json", sparse)
    approx = {}
    for mode in ("sparse_global", "conditional_topk", "coarse_tail"):
        w = student_weight.detach().clone().requires_grad_(True)
        loss = cached_kl(hidden @ w, sparse, mode=mode)
        grad, = torch.autograd.grad(loss, w)
        approx[mode] = {"loss": loss.item(), "loss_delta_from_full": (loss.detach() - online_loss).item(),
                        "gradient_max_abs_delta_from_full": (grad - online_grad).abs().max().item()}
    result = {
        "scope": "synthetic-cpu-only-not-native-training",
        "python": platform.python_version(), "torch": torch.__version__,
        "device": "cpu", "dtype": "float64", "threads": torch.get_num_threads(),
        "seed": 20261017, "positions": 64, "vocab_size": 67, "temperature": 2.,
        "repeats": repeats, "full_online_loss": online_loss.item(), "full_cached_loss": cached_loss.item(),
        "full_loss_abs_error": (online_loss - cached_loss).abs().item(),
        "full_gradient_max_abs_error": (online_grad - cached_grad).abs().max().item(),
        "top32_tail_mass_mean": statistics.mean(sparse["tail_mass"]),
        "top32_gradient_delta_from_full": approx["conditional_topk"]["gradient_max_abs_delta_from_full"],
        "top32_modes": approx, "teacher_dump_seconds": dump_seconds,
        "online_step_mean_seconds": statistics.mean(online_times),
        "cached_read_step_mean_seconds": statistics.mean(cached_times),
        "online_step_seconds": online_times, "cached_read_step_seconds": cached_times,
        "full_cache_json_bytes": full_path.stat().st_size,
        "top32_cache_json_bytes": (output / "top32-cache.json").stat().st_size,
        "timing_warning": "Tiny CPU synthetic calculation, includes JSON validation; not a GPU speed prediction.",
    }
    (output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="fresh local output directory")
    parser.add_argument("--repeats", type=int, default=12)
    args = parser.parse_args()
    print(json.dumps(run_proof(args.output, args.repeats), indent=2))
