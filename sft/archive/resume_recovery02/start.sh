#!/bin/sh
set -eu
cd /raid/yoon/nemochat
exec /raid/yoon/nemochat/.venv/bin/python /raid/yoon/nemochat/sft/archive/resume_recovery02/controller.py --start "$@"
