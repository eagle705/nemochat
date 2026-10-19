"""Fail-closed CPU controls for the parent-reviewed C-only SFT branch."""
from datetime import datetime
from zoneinfo import ZoneInfo
import math
import statistics
import os
from pathlib import Path
import select


def process_identity(pid):
    proc = Path('/proc')/str(pid)
    stat = (proc/'stat').read_text().rsplit(')',1)[1].split()
    return dict(pid=pid, uid=proc.stat().st_uid, starttime=int(stat[19]),
                argv=(proc/'cmdline').read_bytes().rstrip(b'\0').decode().split('\0'),
                state=stat[0])


def wait_exact(expected, timeout):
    def check():
        actual = process_identity(expected['pid'])
        if any(actual[k] != expected[k] for k in ('pid','uid','starttime')):
            raise ValueError('prerequisite process identity changed')
        if actual['state'] != 'Z' and actual['argv'] != expected['argv']:
            raise ValueError('prerequisite argv identity changed')
        return actual
    check()
    fd = os.pidfd_open(expected['pid'])
    try:
        check()
        poll = select.poll()
        poll.register(fd, select.POLLIN)
        if not poll.poll(max(0,math.ceil(timeout*1000))):
            raise TimeoutError('prerequisite did not exit before finite timeout')
    finally:
        os.close(fd)

DEADLINE = datetime(2026, 10, 19, 12, tzinfo=ZoneInfo('Asia/Seoul'))
SOURCE = 'checkpoints/exports/nemochat_mid_super35_distill_hf_bf16_01'
SUITE_SHA = '07609a0cf53e4a63d555a3e50938adfed1cb05471d02b37fb93461e7879af19e'


def validate_source(source):
    if source != SOURCE:
        raise ValueError('C LM+KD1000 model-only source required')


def accept_benchmark(proof, producer_contract):
    """Normalized proof ONLY; actual producer-to-proof adapter awaits parent review.

    This cannot accept an arbitrary existing complete.json as a launch marker.
    The separate release contract must authenticate producer/marker/score bytes.
    """
    if not producer_contract or producer_contract.get('reviewed') is not True:
        raise ValueError('parent-reviewed producer contract required')
    if set(proof.get('models', {})) != {'A','C','B'}:
        raise ValueError('all three scored models required')
    if (proof.get('suite_sha256') != SUITE_SHA or not proof.get('backend') or
            proof.get('clean_exit') is not True or proof.get('gpu_lock_released') is not True):
        raise ValueError('matching suite/backend/clean exit/released GPU lock required')
    reference = None
    for key in ('A','C','B'):
        row = proof['models'][key]
        q, w = row.get('question_ids', []), row.get('heldout_window_ids', [])
        if (len(q) != 400 or len(set(map(str,q))) != 400 or len(w) != 64 or
                len(set(map(str,w))) != 64 or row.get('heldout_targets') != 65536 or
                row.get('clean_exit') is not True):
            raise ValueError('exact benchmark coverage/clean exit required')
        if reference is None:
            reference = (q,w)
        elif reference != (q,w):
            raise ValueError('exact matching coverage IDs required')
        if any(not isinstance(row.get(k),(int,float)) or not math.isfinite(row[k])
               for k in ('score','heldout_nll')):
            raise ValueError('actual finite scores required; no quality threshold')
    return dict(models=['A','C','B'], coverage='400 questions / 64 windows / 65536 targets each')


def budget(now, warm_seconds, unique_dialogues, *, buffer_seconds=1800):
    if now >= DEADLINE.timestamp()-buffer_seconds:
        raise ValueError('deadline has no training/save/eval margin')
    if len(warm_seconds) < 10 or any(not math.isfinite(x) or x <= 0 for x in warm_seconds):
        raise ValueError('ten finite positive fresh native warm timings required')
    if buffer_seconds < 1800 or type(unique_dialogues) is not int or unique_dialogues < 128:
        raise ValueError('explicit save/eval buffer and finite unique pool required')
    mean = statistics.mean(warm_seconds)
    # Conservative observed slowest step + 10% overhead; projection, not guarantee.
    seconds = max(warm_seconds)/.9
    updates = min(math.ceil(unique_dialogues/128), int((DEADLINE.timestamp()-now-buffer_seconds)/seconds))
    if updates < 2:
        raise ValueError('deadline budget too short')
    return dict(updates=updates, unique_capacity_updates=math.ceil(unique_dialogues/128),
                warm_mean_seconds=mean, conservative_iteration_seconds=seconds,
                training_stop_timestamp=DEADLINE.timestamp()-buffer_seconds,
                hard_deadline=DEADLINE.isoformat(), save_eval_buffer_seconds=buffer_seconds,
                epochs=1, allow_repetition=False)
