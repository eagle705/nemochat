#!/bin/sh
set -eu
cd /raid/yoon/nemochat
exec /raid/yoon/nemochat/.venv/bin/python /raid/yoon/nemochat/sft/resume/runtime/controller.py --start "$@"
