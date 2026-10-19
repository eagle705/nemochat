#!/usr/bin/env python3
"""Export/publish only the explicitly authorized private pretrained base artifact.

Run from the repository root with its existing .venv. No GPU, native byte audit,
optimizer state upload, tokenizer migration, git operation, or public publication.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from nemochat.hub.base_checkpoint import (
    REPO_ID, WAIVER, export_model_dcp, initialize_private_repo, verify_remote,
)

SOURCE_REL = Path("results/pretraining/nemochat_baby/training/runs/baby-main-01/checkpoint/iter_0019074")
RESULT_REL = Path("results/pretraining/nemochat_baby/hf_base_publication")
EXPORT_REL = Path("checkpoints/exports/nemochat_baby_base_bf16")
TOKENIZER_REL = Path("checkpoints/teachers/hf-cache/models--nvidia--NVIDIA-Nemotron-3-Super-120B-A12B-BF16/snapshots/2dc98e2afe4face0e4ce40972a915c45368bd34a")
IMAGE = "nvcr.io/nvidia/nemo@sha256:fdd6e9c7929b76c8624ddfea939dda345beaf67695d9c3994994c18f9e79b9bf"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["export", "upload", "verify"])
    parser.add_argument("--revision", help="full immutable SHA, required for verify")
    args = parser.parse_args()
    output, result = ROOT / EXPORT_REL, ROOT / RESULT_REL
    result.mkdir(parents=True, exist_ok=True)
    if args.action == "export":
        import yaml
        if output.exists():
            raise FileExistsError("export destination is not fresh")
        config = yaml.safe_load((ROOT / "results/pretraining/nemochat_baby/training/train.yaml").read_text())
        # The only large read is the selected-record serialization requested by the user.
        receipt = export_model_dcp(ROOT / SOURCE_REL, output / "native")
        if receipt["tensor_dtypes"] != {"torch.bfloat16": 9790967416, "torch.float32": 2944}:
            raise ValueError("export does not match the frozen native model census")
        (output / "tokenizer").mkdir()
        for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
            (output / "tokenizer" / name).write_bytes((ROOT / TOKENIZER_REL / name).read_bytes())
        (output / "README.md").write_bytes((result / "MODEL_CARD.md").read_bytes())
        (output / "USE_CONSTRAINTS.md").write_bytes((result / "USE_CONSTRAINTS.md").read_bytes())
        for name in ("native_loader.py", "load_tensors_cpu.py"):
            (output / name).write_bytes((result / name).read_bytes())
        (output / "model_config.json").write_text(json.dumps(config["model"], indent=2) + "\n")
        manifest = {
            "repo_id": REPO_ID, "private_required": True,
            "stage": "pretrained_base_not_sft_or_rlvr", "checkpoint_update": 19074,
            "source_checkpoint": "baby-main-01/iter_0019074", "checkpoint_integrity": WAIVER,
            "training_positions": 10000269312, "total_learned_parameters": 9790967416,
            "active_parameter_topk_convention": 964054648, "runtime_image": IMAGE,
            "data": {"repo": "karpathy/climbmix-400b-shuffle", "revision": "915333b4f8b8684f39aeaafea600fea6f43fb703",
                     "upstream": "nvidia/Nemotron-ClimbMix", "upstream_revision": "5eaa64b9c0c85b7f56af01d7dffdb0795816b12b",
                     "upstream_license": "CC-BY-NC-4.0", "intended_use": "noncommercial educational research"},
            "tokenizer": {"repo": "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16", "revision": "2dc98e2afe4face0e4ce40972a915c45368bd34a",
                          "original_aliases_preserved": True, "eod_id": 11, "embedding_row_changes": False},
            "export": receipt,
            "operational_export_full_model_reload": "not_exercised_no_independent_weight_reread",
            "cpu_controls": "synthetic_DCP_model_expert_shapes_buffers_extra_state_restore",
        }
        (output / "publication_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (result / "export_receipt.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps(receipt), flush=True)
        return
    # Runtime-only credential parsing: never execute .env or print/persist values.
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        for line in (ROOT / ".env").read_text().splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                if key.strip() in {"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"}:
                    token = value.strip().strip("\"'")
                    if token:
                        break
    if not token:
        raise ValueError("runtime HF credential unavailable")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi(token=token)
    if args.action == "upload":
        manifest = json.loads((output / "publication_manifest.json").read_text())
        if manifest["repo_id"] != REPO_ID or manifest["checkpoint_integrity"] != WAIVER:
            raise ValueError("unexpected export contract")
        parent = initialize_private_repo(api)
        (result / "remote_preflight.json").write_text(json.dumps({"repo_id": REPO_ID, "private": True, "parent_commit": parent}, indent=2))
        print("PRIVATE_TARGET_CONFIRMED; uploading complete model-only payload", flush=True)
        commit = api.upload_folder(repo_id=REPO_ID, repo_type="model", folder_path=str(output),
                                   commit_message="Add final pretrained BF16 base model-only native DCP and original tokenizer",
                                   parent_commit=parent)
        revision = commit.oid
        (result / "commit_returned.json").write_text(json.dumps({"repo_id": REPO_ID, "revision": revision}, indent=2))
    else:
        if not args.revision:
            parser.error("verify requires --revision")
        revision = args.revision
    verified = verify_remote(api, output, revision)
    # Only the small card is downloaded, never the weights.
    card = hf_hub_download(repo_id=REPO_ID, repo_type="model", filename="README.md", revision=revision,
                           token=token, local_dir=str(result / "small_remote_readback"))
    if Path(card).read_bytes() != (output / "README.md").read_bytes():
        raise ValueError("immutable remote model card differs")
    verified["model_card_exact_readback"] = True
    (result / "publication_receipt.json").write_text(json.dumps(verified, indent=2) + "\n")
    print(json.dumps(verified, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # Hub exception text/tracebacks can contain sensitive request details.
        status = getattr(getattr(error, "response", None), "status_code", None)
        print(f"BASE_PUBLICATION_FAILED type={type(error).__name__} http_status={status}", file=sys.stderr)
        sys.exit(1)
