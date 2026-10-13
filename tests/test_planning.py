"""Planning arithmetic/sampling fixtures; no hardware throughput measurements."""
import copy
import json
import random
import subprocess
import sys
from pathlib import Path

import pytest

from nemochat.planning.corpus import estimate_corpus, reservoir_sample, SUPER_REVISION
from nemochat.planning.budget import plan_training


class Tokenizer:
    eos_token_id = 9

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return list(range(len(text)))


def manifest(tmp_path, n=20, sample=5):
    path = tmp_path / "rows.jsonl"
    path.write_text("".join(json.dumps({"text": "a" * (i + 1)}) + "\n" for i in range(n)))
    return {"purpose": "estimate", "seed": 42, "max_files": 2, "max_input_bytes": 10000,
            "max_document_bytes": 1000, "max_sample_documents": 100,
            "max_sample_text_bytes": 10000, "max_sample_tokens": 10000,
            "scope": dict(population="local shards only", shard_selection="fixed", heldout="none", filter="none", dedup="none"),
            "strata": [dict(source="local", shard="a", path=str(path), population_documents=n,
                            sample_documents=sample, mode="reservoir")]}


def plan():
    return {"data_parallel": 2, "micro_batch_size": 2, "gradient_accumulation_steps": 3,
            "nodes": 1, "gpus_per_node": 8, "price_unit": "node", "price_per_hour": 36,
            "remaining_setup_seconds": 10, "time_limit_seconds": 110,
            "stages": [{"name": "short", "sequence_length": 10, "target_tokens": 1200,
                        "tokens_per_update": 120, "step_seconds": [2, 2],
                        "measurement_provenance": "unit fixture", "timing_scope": "core_compute",
                        "utilization": .5, "checkpoint_every_updates": 5, "checkpoint_seconds": 3,
                        "eval_every_updates": 10, "evaluation_seconds": 4}]}


def test_sampling_seed_uniform_and_not_prefix():
    sample, count = reservoir_sample(map(str, range(100)), 10, random.Random(42))
    assert count == 100 and len(sample) == 10
    assert sample == reservoir_sample(map(str, range(100)), 10, random.Random(42))[0]
    assert sample != reservoir_sample(map(str, range(100)), 10, random.Random(43))[0]
    assert any(row > 10 for row, text in sample)
    # Fixed statistical regression: each position's inclusion count near 10%.
    included = [0] * 20
    for seed in range(2000):
        for row, _ in reservoir_sample(range(20), 2, random.Random(seed))[0]:
            included[row - 1] += 1
    assert all(150 < count < 250 for count in included)


def test_census_text_eos_bytes(tmp_path):
    report = estimate_corpus(manifest(tmp_path, 3, 3), Tokenizer())
    assert report["estimated_text_tokens"] == 6
    assert report["estimated_eos_tokens"] == 3
    assert report["estimated_total_tokens"] == 9
    assert report["estimated_utf8_bytes"] == 6
    assert report["uncertainty"]["standard_error"] == 0
    assert report["tokenizer"]["injected_test_interface"]
    assert report["tokenizer"]["revision"] == SUPER_REVISION


def test_variance_is_shard_stratified(tmp_path):
    c = manifest(tmp_path, 20, 5)
    r = estimate_corpus(c, Tokenizer())
    s = r["strata"][0]
    import statistics
    variance = 20**2 * (1 - 5 / 20) * statistics.variance([x["text_tokens"] for x in s["sampled_rows"]]) / 5
    assert s["total_token_variance"] == pytest.approx(variance)
    assert r["uncertainty"]["standard_error"] == pytest.approx(variance**.5)
    assert r == estimate_corpus(c, Tokenizer())


def test_multiple_fixed_shards_no_pooled_iid(tmp_path):
    c = manifest(tmp_path, 20, 5)
    other = tmp_path / "other.jsonl"
    other.write_text("".join(json.dumps({"text": "b" * 100}) + "\n" for _ in range(20)))
    c["strata"].append(dict(source="local", shard="b", path=str(other),
                            population_documents=20, sample_documents=5, mode="reservoir"))
    r = estimate_corpus(c, Tokenizer())
    assert r["strata"][1]["total_token_variance"] == 0
    assert r["uncertainty"]["standard_error"]**2 == pytest.approx(r["strata"][0]["total_token_variance"])
    c["strata"].reverse()
    reordered = estimate_corpus(c, Tokenizer())
    assert {s["shard"]: s["sampled_rows"] for s in r["strata"]} == {s["shard"]: s["sampled_rows"] for s in reordered["strata"]}


def test_representative_no_fake_ci(tmp_path):
    c = manifest(tmp_path, 5, 5)
    c["strata"][0].update(mode="representative", population_documents=1000, sampling_provenance="user-selected external sample; clusters unknown")
    r = estimate_corpus(c, Tokenizer())
    assert r["estimated_text_tokens"] == 3000
    assert r["uncertainty"] is None


@pytest.mark.parametrize("key,value", [("max_files", 0), ("max_input_bytes", 1),
    ("max_document_bytes", 1), ("max_sample_documents", 1),
    ("max_sample_text_bytes", 1), ("max_sample_tokens", 1), ("seed", True),
    ("tokenizer_revision", "main"), ("tokenizer", "gpt2")])
def test_corpus_rejects_invalid_and_quotas(tmp_path, key, value):
    c = manifest(tmp_path)
    c[key] = value
    with pytest.raises(ValueError):
        estimate_corpus(c, Tokenizer())


def test_population_mismatch_no_prefix(tmp_path):
    c = manifest(tmp_path)
    c["strata"][0]["population_documents"] = 100
    with pytest.raises(ValueError, match="scanned population"):
        estimate_corpus(c, Tokenizer())
    c["strata"][0].update(mode="representative", sampling_provenance="user attested")
    with pytest.raises(ValueError, match="exactly sample_documents"):
        estimate_corpus(c, Tokenizer())


def test_duplicate_strata_and_single_sample_ci(tmp_path):
    c = manifest(tmp_path, 20, 1)
    assert estimate_corpus(c, Tokenizer())["uncertainty"] is None
    c["strata"].append(copy.deepcopy(c["strata"][0]))
    with pytest.raises(ValueError, match="duplicate"):
        estimate_corpus(c, Tokenizer())


def test_parquet_actual_local(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    c = manifest(tmp_path, 3, 3)
    path = tmp_path / "rows.parquet"
    pq.write_table(pa.table({"text": ["a", "bb", "ccc"]}), path)
    c["strata"][0]["path"] = str(path)
    assert estimate_corpus(c, Tokenizer())["estimated_total_tokens"] == 9


def test_gbs_overhead_price_and_scenarios():
    r = plan_training(plan())
    assert r["global_batch_size"] == 12
    assert r["hourly_cluster_price"] == 36
    base = r["scenarios"]["base"]
    assert base["remaining_seconds"] == 60  # 10 setup + 40 compute + 6 ckpt + 4 eval
    assert base["remaining_cost"] == .6
    assert base["capacity_tokens"] == 1200
    assert base["fits_limits"]
    assert r["scenarios"]["conservative"]["remaining_seconds"] > 60
    assert r["scenarios"]["optimistic"]["remaining_seconds"] < 60
    c = plan()
    c["price_unit"] = "gpu"
    assert plan_training(c)["scenarios"]["base"]["remaining_cost"] == 4.8


def test_limits_progress_capacity_and_retention():
    c = plan()
    c.update(elapsed_seconds=20, time_limit_seconds=60, budget_limit=100)
    c["storage"] = dict(checkpoint_bytes=100, retain_checkpoints=2, existing_checkpoints=1, eval_artifact_bytes=10)
    r = plan_training(c)
    assert r["remaining_time_seconds"] == 40
    assert r["scenarios"]["base"]["capacity_tokens"] == 720  # 6 steps=24+3 within 30s
    assert not r["scenarios"]["base"]["fits_limits"]
    assert r["scenarios"]["base"]["storage"]["retained_bytes"] == 210
    assert r["scenarios"]["base"]["storage"]["peak_bytes_upper_bound"] == 310
    c["stages"][0]["completed_tokens"] = 600
    assert plan_training(c)["scenarios"]["base"]["stages"][0]["updates"] == 5
    c["remaining_setup_seconds"] = 1000
    assert plan_training(c)["scenarios"]["base"]["capacity_tokens"] == 0


def test_stage_specific_rate_and_sequential_capacity():
    c = plan()
    c["stages"].append(dict(name="long", sequence_length=100, target_tokens=2400,
                            tokens_per_second=100, measurement_provenance="fixture"))
    c["time_limit_seconds"] = 65
    r = plan_training(c)["scenarios"]["base"]
    assert r["remaining_seconds"] == 84
    assert r["capacity_tokens"] == 1200
    assert r["stages"][1]["effective_tokens_per_second"] == 100


def test_budget_exhaustion_and_free_price():
    c = plan()
    c.update(budget_limit=.3, spent_cost=.2)
    r = plan_training(c)["scenarios"]["base"]
    assert r["capacity_tokens"] == 0  # only setup fits
    c["price_per_hour"] = 0
    assert plan_training(c)["scenarios"]["base"]["capacity_tokens"] == 1200
    c.update(time_limit_seconds=0)
    assert plan_training(c)["scenarios"]["base"]["capacity_tokens"] == 0


def test_unknown_b300_no_prediction():
    c = plan()
    c["hardware_label"] = "B300"
    del c["stages"][0]["step_seconds"]
    r = plan_training(c)
    assert all(s["status"] == "measurement_required" and s["remaining_cost"] is None and s["capacity_tokens"] is None for s in r["scenarios"].values())


@pytest.mark.parametrize("key,value", [("data_parallel", 0), ("price_per_hour", -1),
    ("price_unit", "hour"), ("elapsed_seconds", float("nan")), ("global_batch_size", 9)])
def test_bad_plan_inputs(key, value):
    c = plan()
    c[key] = value
    with pytest.raises(ValueError):
        plan_training(c)


@pytest.mark.parametrize("changes", [{"tokens_per_second": 10}, {"step_seconds": [0]},
    {"timing_scope": "effective_wall"}, {"utilization": 1.1}, {"target_tokens": 0},
    {"tokens_per_update": 121}, {"checkpoint_every_updates": 0}, {"completed_tokens": -1}])
def test_bad_stage_inputs(changes):
    c = plan()
    c["stages"][0].update(changes)
    with pytest.raises(ValueError):
        plan_training(c)


def test_cli_real_planner_and_overwrite(tmp_path):
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "report.json"
    command = [sys.executable, str(root / "scripts/plan_training.py"), "--config",
               str(root / "configs/planning/training_measured_example.json"), "--output", str(output)]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text())["scenarios"]["base"]["status"] == "measured_input_projection"
    assert subprocess.run(command, capture_output=True).returncode == 1
