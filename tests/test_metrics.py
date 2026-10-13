"""Metric/byte/CORE fixtures. These are unit tests, not benchmark results."""

import math
import pytest

from nemochat.eval.metrics import LossTotals, centered_accuracy, core_score
from nemochat.eval.token_bytes import bytelevel_alphabet, token_byte_lengths
from nemochat.models.dimensions import dimensions, parameter_estimate


def test_bpb_sum_not_batch_mean():
    a = LossTotals()
    a.add([1.0, 2.0, float("nan")], [1, 4, 0])
    b = LossTotals()
    b.add([4.0], [10])
    a.merge(b)
    out = a.report()
    assert out["target_tokens"] == 3
    assert out["target_bytes"] == 15
    assert out["bpb"] == pytest.approx(7 / (math.log(2) * 15))
    assert out["perplexity"] == pytest.approx(math.exp(7 / 3))


def test_invalid_metric_inputs():
    with pytest.raises(ValueError):
        LossTotals().report()
    with pytest.raises(ValueError):
        LossTotals().add([1], [1, 2])
    with pytest.raises(ValueError):
        LossTotals().add([float("nan")], [1])
    with pytest.raises(ValueError):
        LossTotals().add([1], [-1])


def test_core_baseline_and_equal_task_weight():
    assert centered_accuracy(.6, .25) == pytest.approx(.4666666667)
    assert centered_accuracy(.2, .5) == pytest.approx(-.6)
    result = core_score({"small": 1., "large": .25}, {"small": .5, "large": .25})
    assert result["core_metric"] == .5
    with pytest.raises(ValueError):
        centered_accuracy(.5, 1.)
    with pytest.raises(ValueError):
        core_score({"a": .5}, {"b": .25})


def test_bytelevel_alphabet_is_bijective():
    inverse = bytelevel_alphabet()
    assert len(inverse) == 256
    assert set(inverse.values()) == set(range(256))
    assert inverse["Ġ"] == 32


def test_fail_closed_unsupported_tokenizer():
    class Unsupported:
        is_fast = False
    with pytest.raises(ValueError, match="fast"):
        token_byte_lengths(Unsupported())


def test_dimensions_and_counts():
    a, b = parameter_estimate(dimensions("A")), parameter_estimate(dimensions("B"))
    assert a["blocks"] == {"mamba": 23, "moe": 23, "attention": 6}
    assert a["total_billions"] == pytest.approx(13.7727, abs=.0001)
    assert a["active_billions"] == pytest.approx(2.0035, abs=.0001)
    assert b["total_billions"] > a["total_billions"]
    assert a["components"]["all_routed_experts"] == b["components"]["all_routed_experts"]
    with pytest.raises(ValueError):
        dimensions("C")
