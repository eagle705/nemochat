#!/usr/bin/env python3
"""Execute the unmodified official CPU DCP merge on trusted synthetic fixtures.

This is wiring evidence only, not a model-quality or optimizer-resume test.
Run with the isolated upstream interpreter (Python 3.12+).
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nemochat.checkpoints.wsm import REVISION, validate_upstream


def require(condition: bool, message: str) -> None:
    """Verification guards must remain active under python -O."""
    if not condition:
        raise AssertionError(message)


def validate_observed_state(loaded: dict, common: dict, wrapper: dict) -> int:
    tensor_iterations = loaded["model.tensor._extra_state"].tolist()
    require(tensor_iterations == [30], "Tensor extra-state must come from target iteration 30")
    byte_key = next(k for k in loaded if k.startswith("model.bytes._extra_state"))
    require(loaded[byte_key] == [{"marker": 30}], "Byte extra-state must come from target iteration 30")
    require(not any(k.startswith("optimizer") for k in loaded), "Optimizer entries must be removed")
    require(common["iteration"] == 30, "Common state must come from target iteration 30")
    provenance = common["weighted_merge_provenance"]
    require(provenance["merge_style"] == "minus-sqrt", "Unexpected merge style")
    require(provenance["ignore_non_model_state"] is True, "Non-model-state opt-in missing")
    require(wrapper["revision"] == REVISION and len(wrapper["tool_sha256"]) == 64,
            "Invalid wrapper source provenance")
    return tensor_iterations[0]


def verify(upstream: Path, work: Path) -> dict:
    validate_upstream(upstream)
    if work.exists():
        raise ValueError("Fixture work directory must not exist")
    work.mkdir(parents=True)
    sys.path.insert(0, str(upstream.resolve()))
    # Keep optional upstream imports after source validation. Megatron package
    # initialization has its own Python/dependency requirements even for CPU DCP.
    import torch
    import torch.distributed as dist
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.metadata import TensorStorageMetadata
    from megatron.core.dist_checkpointing.core import CheckpointingConfig, save_config
    from megatron.core.dist_checkpointing.strategies.common import save_common

    dist.init_process_group("gloo", init_method=f"file://{work / 'rendezvous'}", rank=0, world_size=1)
    inputs = work / "inputs"
    inputs.mkdir()
    for iteration, value in [(10, 1.), (20, 3.), (30, 9.)]:
        path = inputs / f"iter_{iteration:07d}"
        path.mkdir()
        # Public DCP writes the exact tensor FQNs and Megatron object shard format.
        # Megatron's training save strategy calls CUDA synchronize even on CPU;
        # fixture construction intentionally uses the public synchronous DCP writer.
        state = {"model.weight": torch.full((2, 3), value),
                 "model.tensor._extra_state": torch.tensor([iteration], dtype=torch.int64),
                 "model.bytes._extra_state/shard_0_1": [{"marker": iteration}],
                 "optimizer.moment": torch.full((2, 3), 99.)}
        dcp.save(state, checkpoint_id=path, planner=dcp.DefaultSavePlanner(flatten_state_dict=False))
        save_common({"iteration": iteration, "consumed_train_samples": iteration * 2}, str(path))
        save_config(CheckpointingConfig("torch_dist", 1), str(path))
    dist.destroy_process_group()
    launcher = Path(__file__).with_name("merge_checkpoints.py")
    base = [sys.executable, str(launcher), "--upstream", str(upstream),
            "--checkpoint-root", str(inputs), "--target-iteration", "30",
            "--window-iterations", "20", "--min-iteration-interval", "1", "--trusted-inputs"]
    checks = []
    def launch(extra: list[str], ok: bool, label: str) -> subprocess.CompletedProcess:
        result = subprocess.run(base + extra, capture_output=True, text=True)
        (work / f"{label}.log").write_text(result.stdout + result.stderr)
        if (result.returncode == 0) != ok:
            raise AssertionError(f"{label}: unexpected exit {result.returncode}\n{result.stdout}\n{result.stderr}")
        checks.append(label)
        return result
    launch(["--output", str(work / "reject-optimizer")], False, "optimizer_rejected_by_default")
    launch(["--output", str(work / "dry"), "--ignore-non-model-state", "--dry-run"], True, "dry_run")
    require(not (work / "dry").exists(), "Dry-run must not create output")
    output = work / "merged"
    launch(["--output", str(output), "--ignore-non-model-state"], True, "official_merge")
    checkpoint = output / "iter_0000030"
    # Independently deserialize with public PyTorch DCP, not the merge implementation.
    metadata = dcp.FileSystemReader(checkpoint).read_metadata()
    loaded = {key: torch.empty(entry.size, dtype=entry.properties.dtype) if isinstance(entry, TensorStorageMetadata) else None
              for key, entry in metadata.state_dict_metadata.items()}
    dcp.load(loaded, checkpoint_id=checkpoint)
    n = 3
    weights = [1 / n ** 0.5, (2 / n) ** 0.5 - (1 / n) ** 0.5, 1 - (2 / n) ** 0.5]
    expected = sum(w * v for w, v in zip(weights, [1., 3., 9.]))
    torch.testing.assert_close(loaded["model.weight"], torch.full((2, 3), expected))
    common = torch.load(checkpoint / "common.pt", map_location="cpu", weights_only=False)
    provenance = common["weighted_merge_provenance"]
    torch.testing.assert_close(torch.tensor(provenance["weights"], dtype=torch.float64),
                               torch.tensor(weights, dtype=torch.float64), rtol=1e-12, atol=1e-12)
    wrapper = json.loads((output / "nemochat_wsm_provenance.json").read_text())
    extra_state_iteration = validate_observed_state(loaded, common, wrapper)
    launch(["--output", str(output), "--ignore-non-model-state"], False, "overwrite_rejected")
    launch(["--output", str(work / "missing"), "--target-iteration", "31", "--ignore-non-model-state"], False, "missing_target_rejected")
    launch(["--output", str(work / "too-few"), "--min-checkpoints", "4", "--ignore-non-model-state"], False, "too_few_rejected")
    filtered = work / "filtered"
    launch(["--output", str(filtered), "--min-iteration-interval", "15", "--ignore-non-model-state"], True, "backward_interval_preserves_target")
    filtered_common = torch.load(filtered / "iter_0000030" / "common.pt", weights_only=False)
    require([Path(x).name for x in filtered_common["weighted_merge_provenance"]["input_paths"]]
            == ["iter_0000010", "iter_0000030"], "Backward interval selection must preserve target")
    # Layout mismatch must be rejected by actual official metadata validation.
    bad = inputs / "iter_0000025"
    bad.mkdir()
    dcp.save({"model.weight": torch.ones(4)}, checkpoint_id=bad)
    (bad / "metadata.json").write_text((inputs / "iter_0000020" / "metadata.json").read_text())
    torch.save({"iteration": 25}, bad / "common.pt")
    launch(["--output", str(work / "bad-layout"), "--ignore-non-model-state", "--dry-run"], False, "layout_mismatch_rejected")
    report = {"scope": "trusted synthetic CPU DCP wiring only; no model-quality claim", "revision": REVISION,
              "python": sys.version, "torch": torch.__version__, "weights": weights,
              "expected_weight": expected, "actual_weight": loaded["model.weight"].tolist(),
              "extra_state_source_iteration": extra_state_iteration, "checks": checks,
              "wrapper_provenance": wrapper, "upstream_provenance": provenance}
    (work / "verification.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"status": "PASS", "report": str(work / "verification.json"), "checks": checks}, indent=2))
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--upstream", required=True, type=Path)
    p.add_argument("--work-dir", required=True, type=Path)
    args = p.parse_args()
    verify(args.upstream.resolve(), args.work_dir.resolve())
