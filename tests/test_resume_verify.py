"""Checkpoint-verifier fixtures, not model training or performance evidence."""

import json
import sys

import pytest
import torch
from safetensors.torch import save_file
from scripts import verify_cpu_resume


@pytest.fixture
def checkpoints(tmp_path):
    left, right = tmp_path / "split", tmp_path / "continuous"
    for path in (left, right):
        path.mkdir()
        save_file({"weight": torch.ones(2)}, str(path / "model.safetensors"))
        torch.save({"optimizer": {"step": torch.tensor(4)},
                    "torch_rng": torch.tensor([1, 2], dtype=torch.uint8)},
                   path / "optimizer_rng.pt")
        (path / "run.json").write_text(json.dumps({
            "contract": {"revision": "fixture"}, "step": 4,
            "cursor": 4, "loss_tokens_seen": 98,
        }))
    return left, right


def invoke(monkeypatch, checkpoints):
    monkeypatch.setattr(sys, "argv", ["verify_cpu_resume.py", *map(str, checkpoints)])
    verify_cpu_resume.main()


def test_equal_checkpoint_passes(monkeypatch, checkpoints, capsys):
    invoke(monkeypatch, checkpoints)
    assert "PASS:" in capsys.readouterr().out


@pytest.mark.parametrize("field,value", [
    ("contract", {"revision": "different"}), ("step", 5),
    ("cursor", 3), ("loss_tokens_seen", 99),
])
def test_run_metadata_mismatch_fails(monkeypatch, checkpoints, field, value):
    path = checkpoints[1] / "run.json"
    metadata = json.loads(path.read_text())
    metadata[field] = value
    path.write_text(json.dumps(metadata))
    with pytest.raises(AssertionError, match=f"run.{field}"):
        invoke(monkeypatch, checkpoints)


def test_optimizer_rng_mismatch_fails(monkeypatch, checkpoints):
    path = checkpoints[1] / "optimizer_rng.pt"
    state = torch.load(path, weights_only=True)
    state["torch_rng"][0] = 3
    torch.save(state, path)
    with pytest.raises(AssertionError, match="training_state.torch_rng"):
        invoke(monkeypatch, checkpoints)


def test_weight_mismatch_fails(monkeypatch, checkpoints):
    save_file({"weight": torch.zeros(2)}, str(checkpoints[1] / "model.safetensors"))
    with pytest.raises(AssertionError, match="model.weight"):
        invoke(monkeypatch, checkpoints)
