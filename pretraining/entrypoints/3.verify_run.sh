#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
NAME="${1:?usage: 3.verify_run.sh FRESH_RUN_NAME}"
# Requires stopped successful exact container and all-rank receipts.
exec "$ROOT/.venv/bin/python" results/pretrain_followup/post_exit_verify.py --name "$NAME"
