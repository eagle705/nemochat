#!/usr/bin/env bash
# Default is inert. Parent reviews source-freeze.json before --start.
set -euo pipefail
ROOT=/raid/yoon/nemochat
CODE="$ROOT/midtraining/offline2b/ablation_auto"
OUT="$ROOT/results/midtraining/offline2b-ablation-auto-01"
PY="$ROOT/.venv/bin/python"
UNIT=nemochat-offline2b-ablation-auto-01.service
cd "$ROOT"
if [[ ${1:-preview} == preview && $# -le 1 ]]; then
  exec "$PY" "$CODE/controller.py" preview
fi
if [[ $# != 1 || $1 != --start ]]; then
  printf 'Usage: %s [preview|--start]\n' "$0" >&2
  exit 2
fi
"$PY" "$CODE/controller.py" preflight > "$OUT/start-preflight.json"
systemd-run --user --unit="$UNIT" \
  --property=Type=exec --property=Restart=no \
  --property=RuntimeMaxSec=23h --property=TimeoutStopSec=180 \
  --property=KillMode=control-group \
  --property="WorkingDirectory=$ROOT" \
  --property="ExecStopPost=$PY $CODE/controller.py cleanup" \
  --property="StandardOutput=append:$OUT/controller.log" \
  --property="StandardError=append:$OUT/controller.log" \
  --setenv=PYTHONDONTWRITEBYTECODE=1 \
  "$PY" -u "$CODE/controller.py" run
systemctl --user show "$UNIT" -p MainPID -p ActiveState -p SubState -p ExecMainStatus -p Restart -p RuntimeMaxUSec
