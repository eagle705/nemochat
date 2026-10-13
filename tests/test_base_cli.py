"""CLI reproducibility guards must fail before loading any hosted model."""

from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("revision", [None, "main", "v1.0"])
def test_hosted_model_requires_immutable_revision(revision, tmp_path):
    root = Path(__file__).resolve().parents[1]
    command = [sys.executable, str(root / "scripts/base_eval.py"),
               "--model", "sshleifer/tiny-gpt2", "--texts", "unused.jsonl",
               "--output", str(tmp_path / "unused")]
    if revision is not None:
        command += ["--revision", revision]
    result = subprocess.run(command, cwd=root, text=True, capture_output=True, timeout=15)
    assert result.returncode == 2
    assert "immutable 40-character commit SHA" in result.stderr
    assert result.stdout == ""
