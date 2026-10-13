"""Synthetic fixtures test import plumbing, NEVER model capability or Gym scores."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from nemochat.evaluation.gym_benchmarks import (
    GYM_REPOSITORY, GYM_REVISION, compare_manifest, numeric_metric,
)


def save(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def synthetic_manifest(tmp_path):
    (tmp_path / "raw.bin").write_bytes(b"SYNTHETIC RAW WEIGHTS, NOT A MODEL")
    (tmp_path / "merged.bin").write_bytes(b"SYNTHETIC MERGED WEIGHTS, NOT A MODEL")
    (tmp_path / "tokenizer.json").write_text("SYNTHETIC TOKENIZER")
    (tmp_path / "snapshot.yaml").write_text("# SYNTHETIC PROTOCOL SNAPSHOT")
    save(tmp_path / "protocol.json", {"generation": {"temperature": 0, "seed": 0},
        "context_length": 128, "prompt_mode": "text_completion", "scorer": "fixture only",
        "scope": "synthetic fixture", "runtime": "no inference", "wsm": {"window": 2}})
    (tmp_path / "input.jsonl").write_text('{"question": "SYNTHETIC"}\n')
    runs = []
    for variant, score in (("raw", 0.25), ("merged", 0.75)):
        (tmp_path / f"{variant}.jsonl").write_text(json.dumps({
            "_ng_task_index": 0, "_ng_rollout_index": 0, "reward": score}) + "\n")
        save(tmp_path / f"{variant}_metrics.json", [{"agent_ref": {"name": "synthetic"}, "agent_metrics": {"mean/reward": score},
            "key_metrics": {}, "group_level_metrics": [], "repeat_level_metrics": []}])
        runs.append({"checkpoint_step": 10, "variant": variant, "task": "ruler_pretrain",
            "benchmark_config": "benchmarks/ruler/config_pretrain.yaml",
            "checkpoint": f"{variant}.bin", "tokenizer": "tokenizer.json",
            "input_jsonl": "input.jsonl", "rollouts": f"{variant}.jsonl",
            "metrics": f"{variant}_metrics.json", "protocol": "protocol.json",
            "protocol_files": ["snapshot.yaml"], "sample_count": 1, "num_repeats": 1,
            "metric_pointer": "/0/agent_metrics/mean~1reward"})
    manifest = {"source": {"repository": GYM_REPOSITORY, "revision": GYM_REVISION},
                "fixture": True, "tasks": ["ruler_pretrain"], "runs": runs}
    save(tmp_path / "manifest.json", manifest)
    return tmp_path / "manifest.json"


def test_fixture_pair_and_cli(synthetic_manifest):
    result = compare_manifest(synthetic_manifest, allow_fixtures=True)
    assert result["fixture"] and result["pairs"][0]["merged_minus_raw"] == 0.5
    root = Path(__file__).resolve().parents[1]
    output = synthetic_manifest.parent / "comparison.json"
    command = [sys.executable, str(root / "scripts/gym_eval.py"), "--manifest",
               str(synthetic_manifest), "--allow-fixtures", "--output", str(output)]
    env = {**os.environ, "PYTHONPATH": str(root / "src")}
    run = subprocess.run(command, env=env, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    assert json.loads(output.read_text())["measurement"] == "synthetic fixture only"
    assert subprocess.run(command, env=env, capture_output=True).returncode != 0


def test_fixture_requires_opt_in(synthetic_manifest):
    with pytest.raises(ValueError, match="allow-fixtures"):
        compare_manifest(synthetic_manifest)


@pytest.mark.parametrize("change,match", [
    ({"mask_sample": True}, "Masked"), ({"error": {}}, "Failed"),
    ({"reward": None}, "reward"), ({"reward": float("nan")}, "reward"),
    ({"_ng_task_index": True}, "identity"), ({"_ng_rollout_index": 1}, "identity"),
])
def test_invalid_rollouts(synthetic_manifest, change, match):
    p = synthetic_manifest.parent / "raw.jsonl"
    row = json.loads(p.read_text())
    row.update(change)
    p.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match=match):
        compare_manifest(synthetic_manifest, allow_fixtures=True)


def test_duplicate_rollout(synthetic_manifest):
    p = synthetic_manifest.parent / "raw.jsonl"
    p.write_text(p.read_text() * 2)
    with pytest.raises(ValueError, match="Duplicate"):
        compare_manifest(synthetic_manifest, allow_fixtures=True)


def test_protocol_mismatch(synthetic_manifest):
    manifest = json.loads(synthetic_manifest.read_text())
    protocol = json.loads((synthetic_manifest.parent / "protocol.json").read_text())
    protocol["generation"]["temperature"] = 1
    save(synthetic_manifest.parent / "different.json", protocol)
    manifest["runs"][1]["protocol"] = "different.json"
    save(synthetic_manifest, manifest)
    with pytest.raises(ValueError, match="Incomparable"):
        compare_manifest(synthetic_manifest, allow_fixtures=True)


@pytest.mark.parametrize("mutation,match", [
    ("missing_pair", "requires raw and merged"), ("pin", "source"),
    ("missing_task", "declared task set"), ("real", "gym-root"),
])
def test_manifest_guards(synthetic_manifest, mutation, match):
    manifest = json.loads(synthetic_manifest.read_text())
    if mutation == "missing_pair": manifest["runs"].pop()
    if mutation == "pin": manifest["source"]["revision"] = "main"
    if mutation == "missing_task": manifest["tasks"].append("mmlu")
    if mutation == "real": manifest["fixture"] = False
    save(synthetic_manifest, manifest)
    with pytest.raises(ValueError, match=match):
        compare_manifest(synthetic_manifest, allow_fixtures=True)


def test_incomplete_grid(synthetic_manifest):
    manifest = json.loads(synthetic_manifest.read_text())
    for run in manifest["runs"]:
        run["num_repeats"] = 2
    save(synthetic_manifest, manifest)
    with pytest.raises(ValueError, match="Incomplete rollout grid"):
        compare_manifest(synthetic_manifest, allow_fixtures=True)


def test_dataset_statistics_are_not_model_scores(synthetic_manifest):
    save(synthetic_manifest.parent / "raw_metrics.json", {"Number of examples": 1})
    with pytest.raises(ValueError, match="official aggregation"):
        compare_manifest(synthetic_manifest, allow_fixtures=True)


def test_aggregation_error(synthetic_manifest):
    save(synthetic_manifest.parent / "raw_metrics.json",
         [{"agent_metrics": {"mean/reward": 0}, "aggregation_error": {}}])
    with pytest.raises(ValueError, match="aggregation failed"):
        compare_manifest(synthetic_manifest, allow_fixtures=True)


@pytest.mark.parametrize("value", [True, None, float("inf"), "0.5"])
def test_bad_metric(value):
    with pytest.raises(ValueError):
        numeric_metric({"score": value}, "/score")
