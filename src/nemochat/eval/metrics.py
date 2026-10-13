"""Metric arithmetic with no model or distributed runtime dependency."""

import math
from dataclasses import dataclass


@dataclass
class LossTotals:
    """Sum first, divide once; never average per-batch BPB values."""

    nll_nats: float = 0.0
    target_tokens: int = 0
    target_bytes: int = 0

    def add(self, losses, byte_lengths):
        if len(losses) != len(byte_lengths):
            raise ValueError("Losses and byte lengths must have the same length")
        for loss, size in zip(losses, byte_lengths):
            if not isinstance(size, int) or isinstance(size, bool) or size < 0:
                raise ValueError("Byte lengths must be nonnegative integers")
            # Special/masked tokens have size zero and contribute to neither sum.
            if size == 0:
                continue
            if not math.isfinite(loss) or loss < 0:
                raise ValueError("Eligible NLL must be finite and nonnegative")
            self.nll_nats += float(loss)
            self.target_tokens += 1
            self.target_bytes += size

    def merge(self, other):
        self.nll_nats += other.nll_nats
        self.target_tokens += other.target_tokens
        self.target_bytes += other.target_bytes

    def report(self):
        if self.target_tokens == 0 or self.target_bytes == 0:
            raise ValueError("Evaluation has no eligible text targets")
        ce = self.nll_nats / self.target_tokens
        return {
            "nll_nats": self.nll_nats,
            "target_tokens": self.target_tokens,
            "target_bytes": self.target_bytes,
            "ce_nats_per_token": ce,
            "perplexity": math.exp(ce) if ce < 709 else None,
            "bpb": self.nll_nats / (math.log(2) * self.target_bytes),
        }


def centered_accuracy(accuracy, baseline):
    if not math.isfinite(accuracy) or not 0 <= accuracy <= 1:
        raise ValueError("Accuracy must be a fraction in [0, 1]")
    if not math.isfinite(baseline) or not 0 <= baseline < 1:
        raise ValueError("Baseline must be a fraction in [0, 1)")
    return (accuracy - baseline) / (1 - baseline)


def core_score(task_accuracies, task_baselines):
    if not task_accuracies or set(task_accuracies) != set(task_baselines):
        raise ValueError("Task accuracies and baselines must be nonempty and match")
    centered = {
        task: centered_accuracy(acc, task_baselines[task])
        for task, acc in task_accuracies.items()
    }
    return {"core_metric": sum(centered.values()) / len(centered),
            "accuracies": dict(task_accuracies), "centered": centered}
