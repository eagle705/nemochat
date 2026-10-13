"""Import official Gym scores without implementing or replacing its scorers."""

import hashlib
import json
import math
import subprocess
from pathlib import Path

GYM_REPOSITORY = "https://github.com/NVIDIA-NeMo/Gym"
GYM_REVISION = "804c12a7d56a151a98ae150a352350e3c856ba47"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_rows(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError("JSONL must contain nonempty object rows")
    return rows


def reject_metric_errors(value):
    if isinstance(value, dict):
        if "aggregation_error" in value or "_ng_aggregation_error" in value:
            raise ValueError("Official aggregation failed; cannot import a score")
        for child in value.values():
            reject_metric_errors(child)
    elif isinstance(value, list):
        for child in value:
            reject_metric_errors(child)


def numeric_metric(metrics, pointer):
    """Read an explicit RFC 6901 pointer; never infer an official metric name."""
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        raise ValueError("metric_pointer must be an explicit JSON pointer")
    value = metrics
    for token in pointer[1:].split("/"):
        token = token.replace("~1", "/").replace("~0", "~")
        value = value[int(token)] if isinstance(value, list) else value[token]
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("Official metric must be a finite number, not a boolean")
    return value


def check_checkout(root):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()
    if git("rev-parse", "HEAD") != GYM_REVISION:
        raise ValueError("Gym checkout does not match the pinned revision")
    if git("diff", "HEAD", "--name-only"):
        raise ValueError("Gym tracked source is modified; use a clean pinned checkout")


def validate_rollouts(rows, count, repeats):
    expected = {(task, repeat) for task in range(count) for repeat in range(repeats)}
    observed = set()
    for row in rows:
        key = (row.get("_ng_task_index"), row.get("_ng_rollout_index"))
        if any(type(index) is not int for index in key) or key not in expected or key in observed:
            raise ValueError("Duplicate, invalid or unexpected rollout identity")
        observed.add(key)
        if row.get("mask_sample", False) is not False:
            raise ValueError("Masked rollout cannot enter a complete paired comparison")
        # Presence, not truthiness: an empty error payload is still an error.
        if any(field in row and row[field] is not None for field in
               ("error", "failure", "_ng_failure_class", "aggregation_error")):
            raise ValueError("Failed rollout cannot be treated as a model score")
        reward = row.get("reward")
        if type(reward) not in (int, float) or not math.isfinite(reward):
            raise ValueError("Missing or nonfinite official reward")
    if observed != expected:
        raise ValueError("Incomplete rollout grid; missing attempts are not zero scores")


def compare_manifest(manifest_path, *, gym_root=None, allow_fixtures=False):
    """Pair task scores at each training step. Metadata remains user-attested."""
    manifest_path = Path(manifest_path).resolve()
    manifest = read_json(manifest_path)
    if manifest.get("source") != {"repository": GYM_REPOSITORY, "revision": GYM_REVISION}:
        raise ValueError("Manifest source must match the official repository and revision")
    fixture = manifest.get("fixture") is True
    if fixture and not allow_fixtures:
        raise ValueError("Synthetic fixture requires --allow-fixtures")
    if not fixture:
        if gym_root is None:
            raise ValueError("Real imports require --gym-root pointing to a clean pinned Gym checkout")
        check_checkout(gym_root)
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list) or not tasks or any(not isinstance(t, str) for t in tasks) or len(set(tasks)) != len(tasks):
        raise ValueError("Declare a nonempty unique task list")
    runs = manifest.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("No runs supplied; no benchmark score has been measured")
    groups = {}
    checkpoint_hashes = {}
    for run in runs:
        step, variant, task = run["checkpoint_step"], run["variant"], run["task"]
        if type(step) is not int or step < 0 or variant not in ("raw", "merged") or task not in tasks:
            raise ValueError("Invalid checkpoint step, variant or task")
        def path(field):
            return manifest_path.parent / run[field]
        config = run["benchmark_config"]
        config_path = Path(config)
        if config_path.is_absolute() or ".." in config_path.parts or not config.startswith("benchmarks/") or config_path.suffix != ".yaml":
            raise ValueError("benchmark_config must identify an official benchmarks YAML")
        if not fixture and not (Path(gym_root) / config).is_file():
            raise ValueError("Benchmark config not found at pinned source")
        count, repeats = run["sample_count"], run["num_repeats"]
        if type(count) is not int or type(repeats) is not int or count < 1 or repeats < 1:
            raise ValueError("Positive integer sample_count and num_repeats required")
        if len(read_rows(path("input_jsonl"))) != count:
            raise ValueError("Input row count differs from declared sample_count")
        validate_rollouts(read_rows(path("rollouts")), count, repeats)
        protocol = read_json(path("protocol"))
        for field in ("generation", "context_length", "prompt_mode", "scorer", "scope", "runtime", "wsm"):
            if field not in protocol:
                raise ValueError(f"Protocol missing {field}")
        if not isinstance(protocol["wsm"], dict):
            raise ValueError("Protocol wsm must declare merge policy shared by the pair")
        metrics = read_json(path("metrics"))
        reject_metric_errors(metrics)
        if not isinstance(metrics, list) or len(metrics) != 1 or not isinstance(metrics[0], dict):
            raise ValueError("Use a single-task, single-agent official aggregation artifact")
        entry = metrics[0]
        if any(field not in entry for field in ("agent_ref", "agent_metrics", "key_metrics",
                                                "group_level_metrics", "repeat_level_metrics")):
            raise ValueError("Missing official aggregation fields; dataset statistics are not scores")
        if not run["metric_pointer"].startswith(("/0/agent_metrics/", "/0/key_metrics/")):
            raise ValueError("Select an official agent_metrics or key_metrics score")
        score = numeric_metric(metrics, run["metric_pointer"])
        evidence: dict = {field + "_sha256": digest(path(field)) for field in
                    ("checkpoint", "tokenizer", "input_jsonl", "protocol", "metrics", "rollouts")}
        checkpoint_key = (step, variant)
        checkpoint_hash = evidence["checkpoint_sha256"]
        if checkpoint_hashes.setdefault(checkpoint_key, checkpoint_hash) != checkpoint_hash:
            raise ValueError("Checkpoint weights differ across tasks at the same step/variant")
        protocol_files = run["protocol_files"]
        if not isinstance(protocol_files, list) or not protocol_files:
            raise ValueError("Provide prompt/scorer/serving config snapshot protocol_files")
        evidence["protocol_files_sha256"] = [digest(manifest_path.parent / p) for p in protocol_files]
        evidence["benchmark_config_sha256"] = None if fixture else digest(Path(gym_root) / config)
        item = {"checkpoint_step": step, "variant": variant, "task": task,
                "benchmark_config": config, "sample_count": count, "num_repeats": repeats,
                "metric_pointer": run["metric_pointer"], "score": score, "protocol": protocol,
                "artifacts": {field: str(path(field).resolve()) for field in
                              ("checkpoint", "tokenizer", "input_jsonl", "protocol", "metrics", "rollouts")},
                **evidence}
        key = (step, task)
        pair = groups.setdefault(key, {})
        if variant in pair:
            raise ValueError("Duplicate checkpoint/task/variant")
        pair[variant] = item
    steps = sorted({step for step, _ in groups})
    if set(groups) != {(step, task) for step in steps for task in tasks}:
        raise ValueError("Every checkpoint must cover the declared task set")
    results = []
    for (step, task), pair in sorted(groups.items()):
        if set(pair) != {"raw", "merged"}:
            raise ValueError("Each checkpoint/task requires raw and merged results")
        raw, merged = pair["raw"], pair["merged"]
        for field in ("benchmark_config", "benchmark_config_sha256", "sample_count", "num_repeats",
                      "metric_pointer", "tokenizer_sha256", "input_jsonl_sha256", "protocol_sha256",
                      "protocol_files_sha256"):
            if raw[field] != merged[field]:
                raise ValueError(f"Incomparable raw/merged pair: {field}")
        delta = merged["score"] - raw["score"]
        if not math.isfinite(delta):
            raise ValueError("Nonfinite score difference")
        results.append({"checkpoint_step": step, "task": task, "raw": raw, "merged": merged,
                        "merged_minus_raw": delta})
    return {"source": manifest["source"], "fixture": fixture,
            "measurement": "synthetic fixture only" if fixture else "imported official metrics; provenance user-attested",
            "warning": "No inference or scoring performed by this importer; no cross-task composite score.",
            "pairs": results}
