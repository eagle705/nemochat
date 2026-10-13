"""Stateless warmup/stable/linear-decay schedule, indexed by global update.

This is a learning-rate policy, not a trainer or weight-merging algorithm.
"""

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class WSDConfig:
    warmup_steps: int
    stable_steps: int
    decay_steps: int
    total_steps: int
    min_lr_ratio: float = 0.0

    def __post_init__(self):
        for name in ("warmup_steps", "stable_steps", "decay_steps", "total_steps"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.stable_steps < 1:
            raise ValueError("stable_steps must be positive")
        if self.total_steps != self.warmup_steps + self.stable_steps + self.decay_steps:
            raise ValueError("total_steps must equal warmup_steps + stable_steps + decay_steps")
        if not math.isfinite(self.min_lr_ratio) or not 0 <= self.min_lr_ratio <= 1:
            raise ValueError("min_lr_ratio must be finite and in [0, 1]")

    def to_dict(self):
        return {"kind": "wsd", "decay_shape": "linear", **asdict(self)}


def learning_rate_for_update(config, peak_lr, update):
    """Return (LR, phase) applied BEFORE the one-based global optimizer update.

    Warmup updates 1..W use peak*u/W. Stable updates W+1..W+S
    use peak. Decay updates W+S+1..T reach the floor on update T.
    Zero warmup or decay skips that phase. No updates beyond T are allowed.
    """
    if not math.isfinite(peak_lr) or peak_lr <= 0:
        raise ValueError("peak_lr must be finite and positive")
    if type(update) is not int or not 1 <= update <= config.total_steps:
        raise ValueError("update must be in [1, total_steps]")
    if update <= config.warmup_steps:
        return peak_lr * update / config.warmup_steps, "warmup"
    if update <= config.warmup_steps + config.stable_steps:
        return peak_lr, "stable"
    progress = (update - config.warmup_steps - config.stable_steps) / config.decay_steps
    ratio = config.min_lr_ratio + (1 - config.min_lr_ratio) * (1 - progress)
    return peak_lr * ratio, "decay"
