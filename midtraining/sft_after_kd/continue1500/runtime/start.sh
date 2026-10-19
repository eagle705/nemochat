#!/bin/sh
set -eu
cd /raid/yoon/nemochat
exec /raid/yoon/nemochat/.venv/bin/python /raid/yoon/nemochat/midtraining/sft_after_kd/continue1500/runtime/controller.py --start "$@"
