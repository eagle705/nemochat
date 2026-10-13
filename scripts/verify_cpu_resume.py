"""Compare real CPU control checkpoints from split and uninterrupted training.

Run manually after the train_cpu.py commands in the README. No downloads.
This verifies model, optimizer, RNG, cursor, and consumed-token state;
it is not a distributed checkpoint parity test.
"""

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file


def assert_same(left, right, name):
    # Explicit checks stay active even if Python is started with -O.
    if isinstance(left, torch.Tensor):
        if not isinstance(right, torch.Tensor):
            raise AssertionError(f"Type differs: {name}")
        if left.dtype != right.dtype or left.shape != right.shape or not torch.equal(left, right):
            raise AssertionError(f"Tensor differs: {name}")
    elif isinstance(left, dict):
        if not isinstance(right, dict) or left.keys() != right.keys():
            raise AssertionError(f"Keys differ: {name}")
        for key in left:
            assert_same(left[key], right[key], f"{name}.{key}")
    elif isinstance(left, (list, tuple)):
        if type(left) is not type(right) or len(left) != len(right):
            raise AssertionError(f"Sequence differs: {name}")
        for index, (a, b) in enumerate(zip(left, right)):
            assert_same(a, b, f"{name}[{index}]")
    elif left != right:
        raise AssertionError(f"Value differs: {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split", type=Path)
    parser.add_argument("uninterrupted", type=Path)
    args = parser.parse_args()
    assert_same(load_file(str(args.split / "model.safetensors")),
                load_file(str(args.uninterrupted / "model.safetensors")), "model")
    assert_same(torch.load(args.split / "optimizer_rng.pt", map_location="cpu", weights_only=True),
                torch.load(args.uninterrupted / "optimizer_rng.pt", map_location="cpu", weights_only=True),
                "training_state")
    left_run = json.loads((args.split / "run.json").read_text())
    right_run = json.loads((args.uninterrupted / "run.json").read_text())
    for field in ("contract", "step", "cursor", "loss_tokens_seen"):
        assert_same(left_run[field], right_run[field], f"run.{field}")
    print("PASS: exact model, optimizer/RNG, and run cursor/token-count equality (CPU control only)")


if __name__ == "__main__":
    main()
