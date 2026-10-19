#!/usr/bin/env bash
set -euo pipefail
ROOT=/raid/yoon/nemochat
CONTROLLER="$ROOT/midtraining/sft_after_kd/runtime05/controller.py"
PYTHON="$ROOT/.venv/bin/python"
cd "$ROOT"
if (($# == 0)); then
  exec "$PYTHON" "$CONTROLLER" --preview calibration
fi
case "$1" in
  --calibration)
    shift
    exec "$PYTHON" "$CONTROLLER" --start calibration "$@"
    ;;
  --run|--start|--preview|--help)
    exec "$PYTHON" "$CONTROLLER" "$@"
    ;;
  *)
    printf '%s\n' 'Use --preview calibration|main, --calibration, or --run main; --release PATH binds parent acceptance.' >&2
    exit 64
    ;;
esac
