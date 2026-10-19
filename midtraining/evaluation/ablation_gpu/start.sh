#!/bin/bash
set -euo pipefail
ROOT=/raid/yoon/nemochat
HERE=$ROOT/midtraining/evaluation/ablation_gpu
OUT=$ROOT/results/midtraining/offline2b-ablation-gpu-01
[[ ${1:-} == --start ]] || { printf 'Use --start for the authorized GPU run\n'; exit 2; }
cd "$ROOT"
"$ROOT/.venv/bin/python" "$HERE/controller.py" freeze
systemd-run --user --unit=nemochat-offline2b-ablation-gpu-01.service --property=Restart=no --property=RuntimeMaxSec=21600 --property=TimeoutStopSec=120 --property=KillMode=control-group --property=MemoryMax=8G --property="WorkingDirectory=$ROOT" --property="StandardOutput=append:$OUT/controller.log" --property="StandardError=append:$OUT/controller.log" --property="ExecStopPost=$ROOT/.venv/bin/python $HERE/controller.py cleanup" "$ROOT/.venv/bin/python" -u "$HERE/controller.py" run
systemctl --user show nemochat-offline2b-ablation-gpu-01.service -p ActiveState -p SubState -p MainPID -p Restart -p RuntimeMaxUSec
