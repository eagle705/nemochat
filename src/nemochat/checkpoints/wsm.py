"""Safe launcher for the official, immutable Megatron WSM implementation.

No checkpoint averaging is implemented here. Only trusted torch_dist inputs
with identical metadata layouts are supported by the upstream tool.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import subprocess
import socket
import sys
import warnings

REVISION = "ca4826e9555032b8710dfb4a4e5693c46c1613dd"
TOOL = "tools/checkpoint/weighted_merge.py"


def validate_upstream(root: Path) -> str:
    """Reject wrong revisions and locally changed dependency/launcher code."""
    def git(*args: str) -> str:
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()
    if git("rev-parse", "HEAD") != REVISION:
        raise ValueError(f"Megatron HEAD must be {REVISION}")
    if git("status", "--porcelain", "--untracked-files=all", "--", "megatron", "tools/checkpoint"):
        raise ValueError("Megatron checkpoint tool/dependencies must be unmodified")
    expected = subprocess.check_output(["git", "-C", str(root), "show", f"{REVISION}:{TOOL}"])
    actual = (root / TOOL).read_bytes()
    if actual != expected:
        raise ValueError("Official weighted_merge.py does not match pinned revision")
    return hashlib.sha256(actual).hexdigest()


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--upstream", required=True, type=Path, help="External pinned sparse Megatron checkout")
    p.add_argument("--python", default=sys.executable, help="Isolated Python 3.12+ with upstream dependencies")
    p.add_argument("--checkpoint-root", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path, help="New output root; must not exist")
    p.add_argument("--target-iteration", required=True, type=int)
    p.add_argument("--window-iterations", required=True, type=int, help="Inclusive [max(0,target-window),target] range")
    p.add_argument("--min-iteration-interval", default=1, type=int)
    p.add_argument("--min-checkpoints", default=2, type=int)
    p.add_argument("--ignore-non-model-state", action="store_true", help="Explicit upstream opt-in; discards distributed optimizer/non-model entries")
    p.add_argument("--trusted-inputs", action="store_true", help="Acknowledge pickle/code-execution risk, including metadata in dry-run")
    p.add_argument("--dry-run", action="store_true", help="Execute upstream metadata/layout validation; no output is created")
    return p


def run(args: argparse.Namespace) -> dict:
    if not args.trusted_inputs:
        raise ValueError("Use --trusted-inputs only for trusted checkpoints: DCP metadata and torch objects may execute code")
    warnings.warn("Trusted-input operation: torch/DCP metadata can execute pickle code. Never merge downloaded untrusted checkpoints.")
    if args.target_iteration < 0 or args.window_iterations <= 0:
        raise ValueError("Target must be nonnegative and window must be positive")
    if args.min_iteration_interval < 1 or args.min_checkpoints < 1:
        raise ValueError("Minimum interval/count must be positive")
    root, output, upstream = args.checkpoint_root.resolve(), args.output.resolve(), args.upstream.resolve()
    # Upstream treats iter_<digits> as a checkpoint path and writes the tracker
    # in its parent, outside the requested output root. Reject before invocation.
    if re.fullmatch(r"iter_(\d+)", output.name):
        raise ValueError("Output root basename must not match iter_<digits>; upstream would write the parent tracker")
    if args.output.exists() or args.output.is_symlink():
        raise ValueError("Refusing to overwrite existing output root")
    if output == root or root in output.parents or output in root.parents:
        raise ValueError("Input and output roots must not overlap")
    if not root.is_dir():
        raise ValueError("Checkpoint root must be a directory")
    source_hash = validate_upstream(upstream)
    start = max(0, args.target_iteration - args.window_iterations)
    command = [args.python, str(upstream / TOOL), "--merge-inputs", str(root),
               "--merge-output", str(output), "--start-checkpoint", str(start),
               "--end-checkpoint", str(args.target_iteration), "--merge-style", "minus-sqrt",
               "--min-iteration-interval", str(args.min_iteration_interval),
               "--min-checkpoints", str(args.min_checkpoints),
               "--common-state-checkpoint", str(root / f"iter_{args.target_iteration:07d}"),
               "--merge-save-dtype", "same"]
    if args.ignore_non_model_state:
        command.append("--merge-ignore-non-model-state")
    if args.dry_run:
        command.append("--dry-run")
    # Single-process CPU launcher: do not inherit a training job's rank environment.
    env = os.environ.copy()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = str(sock.getsockname()[1])
    env.update(CUDA_VISIBLE_DEVICES="", RANK="0", WORLD_SIZE="1", LOCAL_RANK="0",
               MASTER_ADDR="127.0.0.1", MASTER_PORT=port)
    env.pop("PYTHONPATH", None)
    runtime = json.loads(subprocess.check_output(
        [args.python, "-c", "import sys,json,importlib.metadata as m; print(json.dumps({'python':sys.version,'packages':{k:m.version(k) for k in ['torch','numpy','packaging','PyYAML','triton']}}))"],
        env=env, text=True))
    result = subprocess.run(command, env=env, text=True, capture_output=True)
    print(result.stdout, end="")
    print(result.stderr, end="", file=sys.stderr)
    if result.returncode:
        raise RuntimeError(f"Official weighted_merge.py failed (exit {result.returncode}); no substitute merge was used")
    record = {"revision": REVISION, "tool": TOOL, "tool_sha256": source_hash, "runtime": runtime,
              "command": command, "checkpoint_root": str(root), "output": str(output),
              "start_iteration": start, "target_iteration": args.target_iteration,
              "window_iterations": args.window_iterations, "schedule": "minus-sqrt",
              "common_and_extra_state_source": "target checkpoint",
              "ignore_non_model_state": args.ignore_non_model_state,
              "dry_run": args.dry_run, "evidence_scope": "merge wiring, not quality or resumable optimizer validation"}
    if not args.dry_run:
        with (output / "nemochat_wsm_provenance.json").open("x", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
    return record


def main() -> None:
    p = parser()
    try:
        run(p.parse_args())
    except (ValueError, RuntimeError, subprocess.CalledProcessError, OSError) as exc:
        p.exit(2, f"WSM error: {exc}\n")
