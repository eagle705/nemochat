#!/usr/bin/env bash
# Explicitly authorized online W&B authentication preflight; no training.
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
exec .venv/bin/python results/bridge_wandb/wandb_preflight.py "$@"
