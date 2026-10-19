"""Explicit Bridge-to-Core dependency binding, never megatron.training globals.

A binding is process-local, non-reentrant and synchronous. Native DataLoader
workers inherit the bound dataset identity under Linux fork; spawn is rejected
by the integration adapter until worker context transport is implemented.
"""
from contextlib import contextmanager
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Callable


@dataclass(frozen=True)
class CoreRuntime:
    seed: int
    seq_length: int
    train_samples: int
    global_batch_size: int
    micro_batch_size: int
    blend: object
    blend_per_split: object
    iteration: Callable[[], int]
    num_microbatches: Callable[[], int]
    mtp_num_layers: int = 0
    tensorboard_writer: object = None
    data_parallel_size: int = 1
    expert_parallel_size: int = 1
    dataset_identity: str | None = None


def validate_topology(*, tp, cp, dp, pp, ep, gbs, mbs, microbatches):
    """Only single-rank or DP8/EP8 layouts; preserve native Core sharding."""
    if (tp, cp, dp, pp, ep) not in ((1, 1, 1, 1, 1), (1, 1, 8, 1, 8)):
        raise NotImplementedError('only TP1/CP1/PP1 DP1/EP1 or DP8/EP8 are supported')
    if min(gbs, mbs, microbatches) < 1 or gbs != dp * mbs * microbatches:
        raise ValueError('fixed GBS = DP * MBS * native microbatches required')
    return dp


_ACTIVE = None


def current_runtime():
    if _ACTIVE is None:
        raise RuntimeError('Core cached KD runtime is not bound')
    return _ACTIVE


@contextmanager
def bind_runtime(runtime):
    global _ACTIVE
    if _ACTIVE is not None:
        raise RuntimeError('nested Core cached KD binding is unsupported')
    _ACTIVE = runtime
    try:
        yield runtime
    finally:
        _ACTIVE = None


def get_args():
    r = current_runtime()
    return SimpleNamespace(seed=r.seed, seq_length=r.seq_length,
                           train_samples=r.train_samples,
                           global_batch_size=r.global_batch_size,
                           micro_batch_size=r.micro_batch_size,
                           mtp_num_layers=r.mtp_num_layers,
                           curr_iteration=r.iteration())


def get_blend_and_blend_per_split(args):
    r = current_runtime()
    return r.blend, r.blend_per_split


def get_num_microbatches():
    return current_runtime().num_microbatches()


def get_tensorboard_writer():
    return current_runtime().tensorboard_writer


def print_rank_0(message):
    import torch.distributed as dist
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(message, flush=True)
