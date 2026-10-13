"""Evaluate a real local/HF causal LM on CPU; no mock model scores."""

import argparse
import json
import re
from pathlib import Path

from nemochat.tracking import add_tracking_args, from_args
from nemochat.eval.bundle import BUNDLE_SHA256, read_core_bundle
from nemochat.eval.core import evaluate_core_tasks
from nemochat.eval.hf_eval import evaluate_texts
from nemochat.eval.token_bytes import token_byte_lengths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="HF ID or local export; no remote code")
    parser.add_argument("--revision", help="Immutable revision for hosted model and tokenizer")
    parser.add_argument("--texts", type=Path, help="Local JSONL with one text field per row")
    parser.add_argument("--core", action="store_true")
    parser.add_argument("--core-tasks", nargs="+", help="Explicit subset: score is labeled partial")
    parser.add_argument("--max-per-task", type=int, default=100)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--max-documents", type=int, default=20)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    add_tracking_args(parser)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a new run directory")
    if not args.texts and not args.core:
        parser.error("Choose --texts, --core, or both")
    if args.max_documents <= 0:
        parser.error("max-documents must be positive")
    if not Path(args.model).exists() and not re.fullmatch(r"[0-9a-f]{40}", args.revision or ""):
        parser.error("Hosted model requires --revision as an immutable 40-character commit SHA")

    with from_args(args, "pretrain", run_type="evaluation", config=vars(args)) as tracking:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.set_num_threads(2)
        tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision,
                                                 trust_remote_code=False)
        model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision,
                                                    trust_remote_code=False)
        model.to("cpu").eval()
        declared_limit = getattr(model.config, "max_position_embeddings", None)
        if declared_limit is not None and args.context_length > declared_limit:
            parser.error(f"Context exceeds model limit {declared_limit}")
        result = {"model": args.model, "revision": args.revision, "device": "cpu",
                  "measured_inference": True,
                  "warning": "CPU HF evaluator; not MCore distributed or NemoChat training"}
        if args.texts:
            texts = []
            with args.texts.open(encoding="utf-8") as stream:
                for line in stream:
                    if line.strip():
                        record = json.loads(line)
                        if not isinstance(record.get("text"), str):
                            raise ValueError("Every input row must contain a string text")
                        texts.append(record["text"])
                    if len(texts) >= args.max_documents:
                        break
            result["bpb"] = evaluate_texts(model, tokenizer, texts,
                                            token_byte_lengths(tokenizer),
                                            context_length=args.context_length)
        if args.core:
            tasks, data, baselines = read_core_bundle(args.bundle, labels=args.core_tasks)
            result["core"] = evaluate_core_tasks(
                model, tokenizer, tasks, data, baselines,
                max_per_task=args.max_per_task, context_length=args.context_length,
            )
            result["core"]["bundle_sha256"] = BUNDLE_SHA256
            result["core"]["selected_tasks"] = [task["label"] for task in tasks]
            result["core"]["score_scope"] = (
                "partial task subset" if len(tasks) != 22 else
                "full tasks, capped examples" if args.max_per_task != -1 else "full bundle"
            )
        args.output.mkdir(parents=True, exist_ok=False)
        path = args.output / "metrics.json"
        path.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                        encoding="utf-8")
        print(json.dumps({"output": str(path),
                          "bpb": result.get("bpb", {}).get("aggregate"),
                          "core": {k: v for k, v in result.get("core", {}).items()
                                   if k not in ("raw_results",)}}, indent=2))
        tracking.update_config({"evidence": "measured CPU HF evaluation"})
        tracking.log({"evaluation": result})
        tracking.artifact_metadata("evaluation-" + tracking.id, kind="evaluation-metadata",
            metadata={"path": str(path.resolve()), "checkpoint": args.model,
                      "revision": args.revision, "measured_inference": True})

if __name__ == "__main__":
    main()
