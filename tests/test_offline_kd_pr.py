"""CPU-only synthetic contract tests; not a native Bridge integration test."""
import importlib.util
import copy
import json
from pathlib import Path

import pytest
import torch

MODULE = Path(__file__).resolve().parents[1] / "midtraining/patches/offline_kd_pr/cache_contract.py"


def api():
    assert MODULE.exists(), "offline KD cache contract is not implemented"
    spec = importlib.util.spec_from_file_location("offline_kd_contract", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def metadata():
    return {
        "schema_version": 1,
        "teacher_repo": "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-Base-BF16",
        "teacher_revision": "4ba52e8ffa79e9693936c1a819ac89f551ceeef6",
        "teacher_view_hash": "a" * 64,
        "tokenizer_hash": "b" * 64,
        "dataset_revision": "synthetic-fixture-v1",
        "document_ids": ["fixture-0"],
        "sample_id": "fixture-sample-0",
        "sample_order": 0,
        "epoch": 0,
        "packing_policy": "continuous-no-reset",
        "segment_ids": [0, 0, 0, 0],
        "position_ids": [0, 1, 2],
        "loss_mask": [1, 0, 1],
        "label_shift": 1,
        "temperature": 2.0,
        "vocab_size": 5,
        "top_k": 5,
        "dtype": "float64",
        "normalization": "full-vocabulary-log-softmax-at-temperature",
    }


def teacher_logits():
    return torch.tensor([[2., -1., 0., 4., 1.], [0., 1., 2., 3., 4.], [3., 1., -2., 0., 2.]], dtype=torch.float64)


def test_full_vocabulary_cache_roundtrip(tmp_path):
    m = api()
    record = m.make_record(metadata(), [1, 2, 3, 4], teacher_logits())
    path = tmp_path / "cache.json"
    m.write_cache(path, record)
    loaded = m.read_cache(path, metadata(), [1, 2, 3, 4])
    assert loaded == record
    assert record["input_ids_hash"] == m.token_hash([1, 2, 3, 4])
    assert record["prediction_positions"] == [0, 1, 2]
    assert record["target_ids"] == [2, 3, 4]
    assert torch.allclose(torch.tensor(record["tail_mass"]), torch.zeros(3), atol=1e-14)
    with pytest.raises(FileExistsError):
        m.write_cache(path, record)


@pytest.mark.parametrize("field", list(metadata()))
def test_reader_rejects_changed_or_missing_metadata(tmp_path, field):
    m = api()
    path = tmp_path / "cache.json"
    record = m.make_record(metadata(), [1, 2, 3, 4], teacher_logits())
    del record["metadata"][field]
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        m.read_cache(path, metadata(), [1, 2, 3, 4])
    record = m.make_record(metadata(), [1, 2, 3, 4], teacher_logits())
    record["metadata"][field] = "WRONG"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        m.read_cache(path, metadata(), [1, 2, 3, 4])


@pytest.mark.parametrize("kind", ["tokens", "hash", "target", "shift", "duplicate", "nonfinite", "mass", "position", "missing", "extra"])
def test_reader_rejects_corrupt_or_misaligned_record(tmp_path, kind):
    m = api()
    path = tmp_path / "cache.json"
    ids = [1, 2, 3, 4]
    record = m.make_record(metadata(), ids, teacher_logits())
    if kind == "tokens":
        ids = [1, 3, 2, 4]
    elif kind == "hash":
        record["input_ids_hash"] = "bad"
    elif kind == "target":
        record["target_ids"][0] = 4
    elif kind == "shift":
        record["metadata"]["label_shift"] = 0
    elif kind == "duplicate":
        record["token_ids"][0][0] = record["token_ids"][0][1]
    elif kind == "nonfinite":
        record["logprobs"][0][0] = float("nan")
    elif kind == "mass":
        record["logprobs"][0][0] = 1.
    elif kind == "position":
        record["prediction_positions"] = [2, 1, 0]
    elif kind == "missing":
        del record["tail_mass"]
    else:
        record["extra"] = 1
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        m.read_cache(path, metadata(), ids)


@pytest.mark.parametrize("field,value", [("teacher_revision", "not-a-revision"), ("tokenizer_hash", "x" * 64), ("temperature", 0.), ("dtype", "bfloat16"), ("label_shift", 0), ("top_k", 6), ("position_ids", [0]), ("loss_mask", [0, 0, 0]), ("sample_order", -1), ("epoch", -1), ("normalization", "topk-renormalized")])
def test_writer_rejects_invalid_contract(tmp_path, field, value):
    m = api()
    meta = metadata()
    meta[field] = value
    with pytest.raises(ValueError):
        m.make_record(meta, [1, 2, 3, 4], teacher_logits())


@pytest.mark.parametrize("mode,k", [("full", 5), ("sparse_global", 2), ("conditional_topk", 2)])
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_cached_loss_and_gradient_match_online_reference(tmp_path, mode, k, dtype):
    m = api()
    meta = metadata()
    meta.update(top_k=k, dtype=str(dtype).split(".")[-1])
    teacher = teacher_logits().to(dtype)
    path = tmp_path / "cache.json"
    m.write_cache(path, m.make_record(meta, [1, 2, 3, 4], teacher))
    record = m.read_cache(path, meta, [1, 2, 3, 4])
    student = torch.tensor([[1., 2., -1., 0., 3.], [3., 0., 1., 2., -1.], [1., -3., 2., 0., 4.]], dtype=dtype, requires_grad=True)
    other = student.detach().clone().requires_grad_(True)
    online_teacher = torch.log_softmax(teacher / meta["temperature"], -1)
    online_student = torch.log_softmax(student / meta["temperature"], -1)
    if mode != "full":
        online_teacher, ids = online_teacher.topk(k, -1)
        online_student = online_student.gather(-1, ids)
        if mode == "conditional_topk":
            online_teacher = torch.log_softmax(online_teacher, -1)
            online_student = torch.log_softmax(online_student, -1)
    per_token = (online_teacher.exp() * (online_teacher - online_student)).sum(-1)
    mask = torch.tensor(meta["loss_mask"], dtype=dtype)
    reference = (per_token * mask).sum() / mask.sum()
    cached = m.cached_kl(other, record, mode=mode)
    tol = 2e-6 if dtype == torch.float32 else 1e-12
    torch.testing.assert_close(cached, reference, atol=tol, rtol=tol)
    g1, = torch.autograd.grad(reference, student)
    g2, = torch.autograd.grad(cached, other)
    torch.testing.assert_close(g1, g2, atol=tol, rtol=tol)
    assert torch.equal(g2[1], torch.zeros_like(g2[1]))
    assert torch.isfinite(g2).all() and g2.abs().sum() > 0


def test_topk_approximations_are_not_full_kl():
    m = api()
    meta = metadata()
    meta["top_k"] = 2
    record = m.make_record(meta, [1, 2, 3, 4], teacher_logits())
    student = -teacher_logits().clone().requires_grad_(True)
    sparse = m.cached_kl(student, record, mode="sparse_global")
    conditional = m.cached_kl(student, record, mode="conditional_topk")
    assert min(record["tail_mass"]) > 0
    assert not torch.isclose(sparse, conditional)
    with pytest.raises(ValueError, match="full"):
        m.cached_kl(student, record, mode="full")
    with pytest.raises(ValueError):
        m.cached_kl(student, record, mode="mystery")


@pytest.mark.parametrize("k,temp", [(5, 2.), (2, 1.)])
def test_vllm_prompt_rows_cover_every_shifted_position(k, temp):
    m = api()
    meta = metadata()
    meta.update(top_k=k, temperature=temp)
    logits = teacher_logits()
    full = torch.log_softmax(logits, -1)
    # vLLM prompt row zero is None; each subsequent dictionary contains
    # top-K plus the actual target, which may be outside top-K.
    rows = [None]
    for row, target in zip(full, [2, 3, 4]):
        ids = row.topk(k).indices.tolist()
        rows.append({i: float(row[i]) for i in sorted(set(ids + [target]))})
    record = m.from_prompt_logprobs(meta, [1, 2, 3, 4], rows)
    expected = m.make_record(meta, [1, 2, 3, 4], logits)
    assert record["token_ids"] == expected["token_ids"]
    torch.testing.assert_close(torch.tensor(record["logprobs"]), torch.tensor(expected["logprobs"]))
    assert record["prediction_positions"] == [0, 1, 2]


@pytest.mark.parametrize("kind", ["missing_position", "no_first_none", "temperature", "invalid_ids", "not_normalized"])
def test_vllm_prompt_adapter_rejects_incomplete_or_ambiguous_rows(kind):
    m = api()
    meta = metadata()
    meta.update(top_k=2, temperature=1.)
    full = torch.log_softmax(teacher_logits(), -1)
    rows = [None] + [{i: float(row[i]) for i in row.topk(2).indices.tolist()} for row in full]
    if kind == "missing_position":
        rows[2] = None
    elif kind == "no_first_none":
        rows[0] = {}
    elif kind == "temperature":
        meta["temperature"] = 2.
    elif kind == "invalid_ids":
        rows[1] = {True: -0.5, 2: -2.}
    else:
        rows[1] = {0: 0.5, 2: -1.}
    with pytest.raises(ValueError):
        m.from_prompt_logprobs(meta, [1, 2, 3, 4], rows)


@pytest.mark.parametrize("mode,k", [("full", 5), ("conditional_topk", 2), ("sparse_global", 2), ("coarse_tail", 2)])
def test_exact_upstream_source_methods_cpu_tp1(mode, k):
    m = api()
    spec = importlib.util.spec_from_file_location("source_controls", MODULE.with_name("source_controls.py"))
    assert MODULE.with_name("source_controls.py").exists(), "source-method harness is not implemented"
    controls = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(controls)
    meta = metadata()
    meta.update(dtype="float32", top_k=k, temperature=1.)
    teacher = teacher_logits().float()
    student = (-teacher).clone().requires_grad_(True)
    other = student.detach().clone().requires_grad_(True)
    record = m.make_record(meta, [1, 2, 3, 4], teacher)
    per_token = controls.upstream_token_loss(student, teacher, mode=mode, top_k=k)
    mask = torch.tensor(meta["loss_mask"])
    online = (per_token * mask).sum() / mask.sum()
    cached = m.cached_kl(other, record, mode=mode)
    torch.testing.assert_close(cached, online, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(torch.autograd.grad(cached, other)[0], torch.autograd.grad(online, student)[0], atol=2e-6, rtol=2e-6)


def test_current_modelopt_detached_magnitude_balancer_is_not_raw_mixture():
    spec = importlib.util.spec_from_file_location("source_controls", MODULE.with_name("source_controls.py"))
    assert MODULE.with_name("source_controls.py").exists(), "source-method harness is not implemented"
    controls = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(controls)
    lm = torch.tensor(3., requires_grad=True)
    kd = torch.tensor(2., requires_grad=True)
    total = controls.upstream_balanced_loss(lm, kd, scale=.25)
    torch.testing.assert_close(total, torch.tensor(3.75))
    grad_lm, grad_kd = torch.autograd.grad(total, (lm, kd))
    torch.testing.assert_close(grad_lm, torch.tensor(1.))
    torch.testing.assert_close(grad_kd, torch.tensor(.375))
    assert total != .75 * lm + .25 * kd



def test_proof_produces_measured_synthetic_parity_and_timing_artifacts(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(MODULE.parent))
    path = MODULE.with_name("proof.py")
    assert path.exists(), "CPU proof runner is not implemented"
    spec = importlib.util.spec_from_file_location("offline_proof", path)
    proof = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(proof)
    result = proof.run_proof(tmp_path / "proof", repeats=3)
    assert result["scope"] == "synthetic-cpu-only-not-native-training"
    assert result["full_loss_abs_error"] < 1e-12
    assert result["full_gradient_max_abs_error"] < 1e-12
    assert result["top32_tail_mass_mean"] > 0
    assert result["top32_gradient_delta_from_full"] > 0
    for key in ("teacher_dump_seconds", "online_step_mean_seconds", "cached_read_step_mean_seconds"):
        assert result[key] > 0
    assert json.loads((tmp_path / "proof/result.json").read_text()) == result
    assert (tmp_path / "proof/full-cache.json").exists()
    assert (tmp_path / "proof/top32-cache.json").exists()



def test_current_full_kl_balanced_objective_and_gradients_match_cached(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(MODULE.parent))
    import source_controls as controls
    m = api()
    meta = metadata()
    meta.update(dtype="float32", temperature=1.)
    teacher = teacher_logits().float()
    path = tmp_path / "cache.json"
    m.write_cache(path, m.make_record(meta, [1, 2, 3, 4], teacher))
    record = m.read_cache(path, meta, [1, 2, 3, 4])
    a = (-teacher).clone().requires_grad_(True)
    b = a.detach().clone().requires_grad_(True)
    mask = torch.tensor(meta["loss_mask"])
    targets = torch.tensor(record["target_ids"])
    def lm(x):
        return (torch.nn.functional.cross_entropy(x, targets, reduction="none") * mask).sum() / mask.sum()
    native_kl = (controls.upstream_token_loss(a, teacher) * mask).sum() / mask.sum()
    online = controls.upstream_balanced_loss(lm(a), native_kl, scale=.35)
    cached = controls.upstream_balanced_loss(lm(b), m.cached_kl(b, record), scale=.35)
    torch.testing.assert_close(online, cached, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(torch.autograd.grad(online, a)[0], torch.autograd.grad(cached, b)[0], atol=2e-6, rtol=2e-6)


def test_teacher_entropy_constant_changes_detached_balancer_gradients(monkeypatch):
    monkeypatch.syspath_prepend(str(MODULE.parent))
    import source_controls as controls
    teacher = teacher_logits().float()
    q = torch.softmax(teacher, -1)
    a = (-teacher).clone().requires_grad_(True)
    b = a.detach().clone().requires_grad_(True)
    def objective(x, use_ce):
        logp = torch.log_softmax(x, -1)
        kd = -(q * logp).sum(-1).mean() if use_ce else (q * (q.log() - logp)).sum(-1).mean()
        lm = torch.nn.functional.cross_entropy(x, torch.tensor([2, 3, 4]))
        return controls.upstream_balanced_loss(lm, kd, scale=.35)
    with_ce, with_kl = objective(a, True), objective(b, False)
    # Scalar equality is particularly misleading: detached magnitude scaling
    # makes both total values LM*(1+scale), but their KD gradients differ.
    torch.testing.assert_close(with_ce, with_kl)
    ga, = torch.autograd.grad(with_ce, a)
    gb, = torch.autograd.grad(with_kl, b)
    assert (ga - gb).abs().max() > 1e-4


@pytest.mark.parametrize("kind", ["duplicate_json_key", "missing_file", "truncated_file"])
def test_reader_never_repairs_invalid_json(tmp_path, kind):
    m = api()
    path = tmp_path / "cache.json"
    if kind == "missing_file":
        with pytest.raises(FileNotFoundError):
            m.read_cache(path, metadata(), [1, 2, 3, 4])
        return
    record = m.make_record(metadata(), [1, 2, 3, 4], teacher_logits())
    serialized = json.dumps(record)
    if kind == "duplicate_json_key":
        serialized = '{"metadata": null,' + serialized[1:]
    else:
        serialized = serialized[:-3]
    path.write_text(serialized)
    with pytest.raises(ValueError):
        m.read_cache(path, metadata(), [1, 2, 3, 4])



def test_vllm_objects_filter_actual_target_by_rank_not_dictionary_order():
    from types import SimpleNamespace
    m = api()
    meta = metadata()
    meta.update(top_k=2, temperature=1.)
    rows = [None]
    for row, target in zip(torch.log_softmax(teacher_logits(), -1), [2, 3, 4]):
        ordered = row.argsort(descending=True).tolist()
        # Put the actual prompt target first, just like an API is allowed to.
        ids = list(dict.fromkeys([target] + ordered[:2]))
        rows.append({i: SimpleNamespace(logprob=float(row[i]), rank=ordered.index(i) + 1) for i in ids})
    record = m.from_vllm_prompt_output(meta, [1, 2, 3, 4], rows)
    expected = m.make_record(meta, [1, 2, 3, 4], teacher_logits())
    assert record["token_ids"] == expected["token_ids"]
    assert all(len(row) == 2 for row in record["token_ids"])
    rows[1][next(iter(rows[1]))].rank = None
    with pytest.raises(ValueError):
        m.from_vllm_prompt_output(meta, [1, 2, 3, 4], rows)



@pytest.mark.parametrize("field,value", [("label_shift", True), ("label_shift", 1.), ("position_ids", [0., 1., 2.])])
def test_contract_does_not_coerce_numeric_token_position_fields(field, value):
    m = api()
    meta = metadata()
    meta[field] = value
    with pytest.raises(ValueError):
        m.make_record(meta, [1, 2, 3, 4], teacher_logits())



@pytest.mark.parametrize("field", ["input_ids", "prediction_positions", "target_ids", "metadata"])
def test_rehashed_cache_still_requires_exact_contract_types(tmp_path, field):
    m = api()
    record = m.make_record(metadata(), [1, 2, 3, 4], teacher_logits())
    if field == "metadata":
        record[field]["label_shift"] = 1.
    else:
        record[field] = [float(i) for i in record[field]]
    record["payload_sha256"] = m._digest({k: v for k, v in record.items() if k != "payload_sha256"})
    path = tmp_path / "cache.json"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        m.read_cache(path, metadata(), [1, 2, 3, 4])
