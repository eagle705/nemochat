"""Readable CPU next-token loop to exercise update/save/resume mechanics.

This is a tiny HF control, NOT training NemoChat or a Bridge replacement.
Do not use the hand-written smoke corpus as a model capability benchmark.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nemochat.tracking import add_tracking_args, from_args, hf_reference
from nemochat.training.wsd import WSDConfig, learning_rate_for_update


CHECKPOINT_FORMAT = "cpu-hf-sha256-v1"


def checkpoint_artifacts(path):
    """HF saves flat files. Reject links/directories rather than following them."""
    artifacts = {}
    for entry in path.iterdir():
        if entry.is_symlink() or not entry.is_file():
            raise ValueError(f"Unsafe checkpoint artifact: {entry.name}")
        if entry.name != "run.json":
            artifacts[entry.name] = entry
    return artifacts


def artifact_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_checkpoint_inventory(path, manifest, allow_legacy=False):
    """Detect accidental changes, not malicious rewriting of the manifest."""
    inventory = manifest.get("artifacts_sha256")
    if "checkpoint_format" not in manifest and "artifacts_sha256" not in manifest:
        if not allow_legacy:
            raise ValueError("Unverified legacy resume requires --allow-unverified-legacy-resume")
        # Even explicitly unverified legacy checkpoints must not follow file links.
        artifacts = checkpoint_artifacts(path)
        for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
            if name in artifacts:
                weight_map = json.loads(artifacts[name].read_text()).get("weight_map")
                if (not isinstance(weight_map, dict) or not weight_map
                        or any(not isinstance(shard, str) or shard not in artifacts
                               for shard in weight_map.values())):
                    raise ValueError("Unsafe or incomplete checkpoint shard index")
        print("WARNING: unverified legacy resume; checkpoint identity is not verified", file=sys.stderr)
        return
    if manifest.get("checkpoint_format") != CHECKPOINT_FORMAT:
        raise ValueError("Unsupported checkpoint inventory format")
    if not isinstance(inventory, dict) or not inventory:
        raise ValueError("Missing or incomplete SHA256 inventory")
    for name, digest in inventory.items():
        if (not isinstance(name, str) or not name or name in (".", "..", "run.json")
                or "/" in name or "\\" in name or Path(name).is_absolute()):
            raise ValueError("Unsafe checkpoint inventory path")
        if (not isinstance(digest, str) or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)):
            raise ValueError("Invalid SHA256 inventory digest")
    artifacts = checkpoint_artifacts(path)
    if set(inventory) != set(artifacts):
        raise ValueError("SHA256 inventory has missing or unexpected checkpoint artifacts")
    required = {"config.json", "tokenizer_config.json", "optimizer_rng.pt"}
    tokenizer_present = (bool(set(inventory) & {"tokenizer.json", "tokenizer.model", "spiece.model", "vocab.txt"})
                         or {"vocab.json", "merges.txt"} <= set(inventory))
    weights = {name for name in inventory if name.endswith(".safetensors")
               or (name.startswith("pytorch_model") and name.endswith(".bin"))}
    if not required <= set(inventory) or not tokenizer_present or not weights:
        raise ValueError("Incomplete SHA256 inventory: model/tokenizer/optimizer required")
    for name, entry in artifacts.items():
        if artifact_sha256(entry) != inventory[name]:
            raise ValueError(f"SHA256 mismatch for checkpoint artifact: {name}")
    # HF weight indexes must not redirect loaders outside the verified files.
    for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        if name in artifacts:
            index = json.loads(artifacts[name].read_text())
            weight_map = index.get("weight_map")
            if (not isinstance(weight_map, dict) or not weight_map
                    or any(not isinstance(shard, str) or shard not in weights
                           for shard in weight_map.values())):
                raise ValueError("Unsafe or incomplete checkpoint shard index")


def save_checkpoint(path, model, tokenizer, optimizer, contract, step, cursor,
                    tokens_seen, parent, history):
    """Capture an update boundary without advancing RNG or overwriting a path."""
    import torch

    path.mkdir(parents=True, exist_ok=False)
    rng = torch.get_rng_state().clone()
    try:
        model.save_pretrained(path, safe_serialization=True)
        tokenizer.save_pretrained(path)
        torch.save({"optimizer": optimizer.state_dict(), "torch_rng": rng},
                   path / "optimizer_rng.pt")
        manifest = {"kind": "CPU HF control, not NemoChat/Bridge", "contract": contract,
                    "step": step, "cursor": cursor, "loss_tokens_seen": tokens_seen,
                    "parent": str(parent) if parent else None,
                    "history_this_invocation": list(history),
                    "checkpoint_format": CHECKPOINT_FORMAT,
                    "artifacts_sha256": {name: artifact_sha256(entry) for name, entry
                                         in sorted(checkpoint_artifacts(path).items())}}
        (path / "run.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    finally:
        torch.set_rng_state(rng)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="sshleifer/tiny-gpt2")
    parser.add_argument("--revision", default="5f91d94bd9cd7190a9f3216ff93cd1dd95f2c7be")
    parser.add_argument("--texts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--allow-unverified-legacy-resume", action="store_true",
                        help="Explicitly accept historical checkpoints without an integrity inventory")
    parser.add_argument("--steps", type=int, default=3, help="Additional updates in this invocation")
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lr-schedule", choices=("constant", "wsd"), default="constant")
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--stable-steps", type=int)
    parser.add_argument("--decay-steps", type=int)
    parser.add_argument("--schedule-total-steps", type=int,
                        help="Global WSD horizon, independent of invocation --steps")
    parser.add_argument("--min-lr-ratio", type=float)
    parser.add_argument("--checkpoint-every", type=int,
                        help="Save after global updates divisible by this positive interval")
    parser.add_argument("--checkpoint-dir", type=Path,
                        help="Periodic checkpoint root; child step-NNNNNNNN paths must not exist")
    add_tracking_args(parser)
    args = parser.parse_args(argv)
    if (args.steps <= 0 or args.context_length < 2 or
            not math.isfinite(args.learning_rate) or args.learning_rate <= 0):
        parser.error("steps/lr must be positive and lr finite; context-length >= 2")
    if args.output.exists():
        parser.error("Output exists; choose a new checkpoint directory")
    schedule = None
    phase_counts = (args.warmup_steps, args.stable_steps, args.decay_steps, args.schedule_total_steps)
    if args.lr_schedule == "wsd":
        if any(value is None for value in phase_counts):
            parser.error("WSD requires warmup/stable/decay steps and schedule-total-steps")
        try:
            schedule = WSDConfig(*phase_counts, min_lr_ratio=(
                0.0 if args.min_lr_ratio is None else args.min_lr_ratio))
        except ValueError as exc:
            parser.error(str(exc))
    elif any(value is not None for value in phase_counts) or args.min_lr_ratio is not None:
        parser.error("WSD settings require --lr-schedule wsd")
    if (args.checkpoint_every is None) != (args.checkpoint_dir is None):
        parser.error("checkpoint-every and checkpoint-dir must be supplied together")
    if args.checkpoint_every is not None and args.checkpoint_every <= 0:
        parser.error("checkpoint-every must be positive")

    data_hash = hashlib.sha256(args.texts.read_bytes()).hexdigest()
    # Keep the original constant-LR contract byte-for-byte compatible in structure.
    contract = {"texts_sha256": data_hash, "context_length": args.context_length,
                "learning_rate": args.learning_rate, "model": args.model,
                "revision": args.revision}
    if schedule:
        contract["lr_schedule"] = schedule.to_dict()
    prior = None
    step = cursor = tokens_seen = 0
    if args.resume:
        if (args.resume / "run.json").is_symlink():
            raise ValueError("Unsafe checkpoint manifest link")
        prior = json.loads((args.resume / "run.json").read_text())
        verify_checkpoint_inventory(args.resume, prior, args.allow_unverified_legacy_resume)
        if prior["contract"] != contract:
            raise ValueError("Resume input data/config differs from original run")
        step, cursor, tokens_seen = prior["step"], prior["cursor"], prior["loss_tokens_seen"]
        for value in (step, cursor, tokens_seen):
            if type(value) is not int or value < 0:
                raise ValueError("Invalid saved step/cursor/token count")
    if schedule and step + args.steps > schedule.total_steps:
        parser.error("Requested updates exceed the global WSD horizon")
    periodic_paths = {}
    if args.checkpoint_every:
        root = args.checkpoint_dir.resolve()
        output = args.output.resolve()
        if root == output or root in output.parents or output in root.parents:
            parser.error("Periodic checkpoint root and final output must not overlap")
        if root.exists() and not root.is_dir():
            parser.error("Periodic checkpoint root is not a directory")
        for update in range(step + 1, step + args.steps + 1):
            if update % args.checkpoint_every == 0:
                path = root / f"step-{update:08d}"
                if path.exists():
                    parser.error(f"Periodic checkpoint exists: {path}")
                periodic_paths[update] = path

    with from_args(args, "pretrain", run_type="stage", config=vars(args)) as tracking:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.set_num_threads(2)
        torch.manual_seed(1234)
        model_source = str(args.resume) if args.resume else args.model
        revision = None if args.resume else args.revision
        tokenizer = AutoTokenizer.from_pretrained(model_source, revision=revision,
                                                 trust_remote_code=False)
        model = AutoModelForCausalLM.from_pretrained(model_source, revision=revision,
                                                    trust_remote_code=False, weights_only=True)
        model.to("cpu").train()
        limit = getattr(model.config, "max_position_embeddings", None)
        if limit is not None and args.context_length > limit:
            parser.error("Requested context exceeds model positional capacity")
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        if prior:
            state = torch.load(args.resume / "optimizer_rng.pt", map_location="cpu", weights_only=True)
            optimizer.load_state_dict(state["optimizer"])
            torch.set_rng_state(state["torch_rng"])
        prefix = tokenizer.bos_token_id
        if prefix is None:
            prefix = tokenizer.eos_token_id
        if prefix is None:
            raise ValueError("Need a declared reset-prefix token")
        chunks = []
        with args.texts.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                text = json.loads(line)["text"]
                ids = tokenizer.encode(text, add_special_tokens=False)
                for start in range(0, len(ids), args.context_length - 1):
                    targets = ids[start:start + args.context_length - 1]
                    chunks.append([prefix] + targets)
        if not chunks:
            raise ValueError("No training chunks")
        tracking.update_config({"evidence": "measured CPU tiny-HF control, not NemoChat",
                                "hf_model": args.model, "immutable_revision": args.revision,
                                "resume_checkpoint": str(args.resume) if args.resume else None})
        if tracking.active and not Path(args.model).exists():
            tracking.artifact_metadata("source-" + tracking.id, kind="hf-reference",
                                       metadata=hf_reference(args.model, args.revision))
        history = []
        consumed_this_invocation = 0
        loss_this_invocation = 0
        for _ in range(args.steps):
            update_started = time.monotonic()
            lr, phase = (learning_rate_for_update(schedule, args.learning_rate, step + 1)
                         if schedule else (args.learning_rate, "constant"))
            for group in optimizer.param_groups:
                group["lr"] = lr
            ids = torch.tensor([chunks[cursor % len(chunks)]], dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            # HF performs the causal shift internally when labels=input_ids.
            loss = model(input_ids=ids, labels=ids, use_cache=False).loss
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            update_seconds = time.monotonic() - update_started
            step += 1
            cursor += 1
            tokens_seen += ids.shape[1] - 1
            record = {"step": step, "loss_tokens_seen": tokens_seen, "lm_ce": float(loss.detach()),
                      "learning_rate": lr, "lr_phase": phase}
            consumed_this_invocation += ids.numel()
            loss_this_invocation += ids.shape[1] - 1
            tracking.log({**record, "consumed_input_tokens_this_invocation": consumed_this_invocation,
                          "loss_tokens_this_invocation": loss_this_invocation,
                          "global_update": step, "update_seconds": update_seconds,
                          "loss_tokens_per_update_second": (ids.shape[1] - 1) / update_seconds}, step=step)
            history.append(record)
            print(json.dumps(record, allow_nan=False))
            if step in periodic_paths:
                save_checkpoint(periodic_paths[step], model, tokenizer, optimizer, contract,
                                step, cursor, tokens_seen, args.resume, history)
        save_checkpoint(args.output, model, tokenizer, optimizer, contract,
                        step, cursor, tokens_seen, args.resume, history)
        tracking.artifact_metadata("checkpoint-" + tracking.id + "-" + str(step), kind="checkpoint-metadata",
            metadata={"path": str(args.output.resolve()), "global_update": step,
                      "resume_checkpoint": str(args.resume) if args.resume else None,
                      "hf_model": args.model, "immutable_revision": args.revision,
                      "weights_uploaded": False})

if __name__ == "__main__":
    main()
