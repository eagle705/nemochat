#!/usr/bin/env bash
set -euo pipefail
cd -- /raid/yoon/nemochat
python=/raid/yoon/nemochat/.venv/bin/python
entry=/raid/yoon/nemochat/midtraining/sft_after_kd/controller.py
case "${1:---preview}" in
  --preview) exec "$python" "$entry" --preview ;;
  --start) exec "$python" "$entry" --verify-start ;;
  *) printf '%s\n' 'Only --preview or --start (fail-closed until reviewed) are accepted.' >&2; exit 64 ;;
esac
