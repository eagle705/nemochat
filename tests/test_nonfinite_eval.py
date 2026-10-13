"""Real forward-pass regressions for nonfinite inference in CORE scoring."""

from types import SimpleNamespace

import pytest
import torch

from nemochat.eval.core import evaluate_item
from nemochat.eval.hf_eval import next_token_outputs


class TinyModel(torch.nn.Module):
    def __init__(self, output_kind="valid"):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        logits = torch.tensor([0., 5., 0., 0., 0.])
        if output_kind in {"nan", "inf", "negative_inf"}:
            # Even a non-target vocabulary entry must be finite.
            logits[0] = {"nan": float("nan"), "inf": float("inf"),
                         "negative_inf": -float("inf")}[output_kind]
        elif output_kind == "overflow_nll":
            # Finite float32 logits can still produce infinite cross-entropy.
            logits[0], logits[1] = 3e38, -3e38
        self.register_buffer("token_logits", logits)

    def forward(self, input_ids, use_cache=False):
        logits = self.token_logits.expand(*input_ids.shape, -1) + self.anchor
        return SimpleNamespace(logits=logits)


class TinyTokenizer:
    bos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [{"a": 1, "b": 2, "Q": 3, " ": 4}[char] for char in text]


INVALID_OUTPUTS = ["nan", "inf", "negative_inf", "overflow_nll"]


def score_item(model, task_type):
    if task_type == "multiple_choice":
        item = {"query": "Q", "choices": ["a", "b"], "gold": 0}
    else:
        item = {"context": "Q", "continuation": "a"}
    return evaluate_item(model, TinyTokenizer(), item, task_type, delimiter="")


@pytest.mark.parametrize("output_kind", INVALID_OUTPUTS)
def test_forward_rejects_nonfinite_outputs_before_argmax(output_kind, monkeypatch):
    def forbidden_argmax(*args, **kwargs):
        pytest.fail("Invalid inference reached argmax")

    monkeypatch.setattr(torch.Tensor, "argmax", forbidden_argmax)
    with pytest.raises(ValueError, match="Nonfinite.*(logits|NLL)"):
        next_token_outputs(TinyModel(output_kind), [0, 1])


@pytest.mark.parametrize("output_kind", INVALID_OUTPUTS)
@pytest.mark.parametrize("task_type", ["multiple_choice", "language_modeling"])
def test_core_rejects_nonfinite_inference(output_kind, task_type):
    # Gold option zero reproduced a false correct result with NaN losses.
    with pytest.raises(ValueError, match="Nonfinite.*(logits|NLL)"):
        score_item(TinyModel(output_kind), task_type)


def test_valid_forward_outputs():
    model = TinyModel()
    losses, predictions = next_token_outputs(model, [0, 1, 2])
    expected = torch.nn.functional.cross_entropy(
        model.token_logits.expand(2, -1), torch.tensor([1, 2]), reduction="none")
    assert losses == pytest.approx(expected.tolist())
    assert predictions == [1, 1]


@pytest.mark.parametrize("task_type", ["multiple_choice", "language_modeling"])
def test_valid_core_scoring(task_type):
    result = score_item(TinyModel(), task_type)
    assert result["correct"] is True
    assert all(torch.isfinite(torch.tensor(result["mean_token_nll"])))
    if task_type == "multiple_choice":
        assert result["predicted_option"] == 0
        assert result["mean_token_nll"][0] < result["mean_token_nll"][1]
    else:
        assert result["predicted_option"] is None
