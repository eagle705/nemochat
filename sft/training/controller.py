#!/usr/bin/env python3
"""Inert SFT continuation preview. Parent must bind the REAL GPU producer first."""
import argparse
import json
from pathlib import Path
import sys
from controls import DEADLINE,SOURCE,SUITE_SHA

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'results/midtraining/sft-after-kd-01'
UNIT = 'nemochat-sft-after-kd-01.service'
# Exact finished producer is now implemented in benchmark_gate.py and was read
# back once after the parent reported clean prerequisite service exit.
PRODUCER_CONTRACT_BOUND = True


def supervisor_argv():
    return ['systemd-run','--user','--unit='+UNIT,'--service-type=exec',
            '--working-directory='+str(ROOT),'--property=Restart=no',
            '--property=KillMode=control-group','--property=TimeoutStopSec=180',
            '--property=NoNewPrivileges=yes','--property=RuntimeMaxSec=39600',
            '--setenv=PYTHONPATH=src',str(ROOT/'.venv/bin/python'),'-u',
            str(Path(__file__).resolve()),'--execute']


def preview():
    readiness = json.loads((OUT/'cpu-readiness.json').read_text())
    return dict(status='INERT_PARENT_REVIEW_REQUIRED',gpu_launches=0,
                hard_deadline=DEADLINE.isoformat(),source_hf=SOURCE,
                source_native='results/midtraining/offline2b-prefixtrain-1000-01/checkpoint/iter_0001000',
                producer_contract_bound=PRODUCER_CONTRACT_BOUND,
                required_benchmark_root='results/midtraining/offline2b-ablation-gpu-01',
                expected_suite_sha256=SUITE_SHA,
                prerequisites=['exact benchmark supervisor PID/UID/argv/starttime -> pidfd exit',
                    'parent-reviewed producer contract + immutable complete marker published LAST',
                    'all A/C/B real reports: same 400 question IDs, 64 windows, 65536 targets and backend',
                    'all three clean exits, comparison PNGs decoded, cooperative GPU lock released',
                    'stable approved full SFT data completion proof (currently missing)',
                    'new C corpus and SFT reserved-ID alias audit (old scan does not cover C)',
                    'fresh C 16K EP8 GB128 calibration: actual restore, finite positive update, resumable save, clean exit',
                    'budget from actual warm timings, remaining clock and >=1800s save/eval/export buffer',
                    'fresh C model-only main (not calibration weights); one finite unique epoch maximum'],
                current_data_status=readiness['status'],
                native_entrypoint='sft/training/native_sft_v1.py',
                versioned_recipe='sft/configs/recipe-v1.yaml',
                native_command_preview=['torchrun','--nnodes=1','--nproc-per-node=8',
                    '--master-addr=127.0.0.1','--master-port=29619',
                    'sft/training/native_sft_v1.py','--config','PARENT_FROZEN_CONFIG',
                    '--manifest',readiness['manifest'],'--output',str(OUT/'main'),
                    '--release',str(OUT/'parent-release.json')],
                supervisor_argv=supervisor_argv(),auto_restart=False,
                actual_gpu_config_construction='SDK config constructed in pinned no-device CPU container; weights/optimizer unrun',
                execution_implemented=False,
                note='Exact GPU benchmark contract accepted; preview cannot schedule until stable data/tokenizer audits and parent launcher integration are complete.')


def verify_start():
    if not PRODUCER_CONTRACT_BOUND:
        raise PermissionError('producer contract unbound; parent review of actual GPU completion required; no service started')
    ready=ROOT/'results/pretraining/nemochat_baby/nemotron_sft/cpu-full-03/completion-ready.json'
    if not ready.is_file():
        raise PermissionError('stable SFT data completion proof missing; prior stable-file hash failure remains unresolved; no service started')
    raise PermissionError('parent source-freeze, new-corpus reserved-ID audit and calibration launcher integration required before scheduling')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    choice=parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--preview',action='store_true')
    choice.add_argument('--verify-start',action='store_true')
    choice.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    if args.preview:
        print(json.dumps(preview(),indent=2))
    else:
        try:
            verify_start()
        except PermissionError as exc:
            print(str(exc),file=sys.stderr)
            raise SystemExit(78)


if __name__=='__main__':
    main()
