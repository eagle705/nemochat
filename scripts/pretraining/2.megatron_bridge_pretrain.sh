#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
# Fresh names only. Runtime key is parsed securely by the Python launcher.
NAME="${1:?usage: 2.megatron_bridge_pretrain.sh FRESH_RUN_NAME}"
exec "$ROOT/.venv/bin/python" results/pretrain_followup/online_launcher.py --name "$NAME" --timeout "${2:-1800}"
