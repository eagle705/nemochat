"""Deterministic task-scoring fixtures, not model capability measurements."""

import pytest

from nemochat.eval.core import common_length, render_prompts, tokenize_spans, evaluate_item


class CharacterTokenizer:
    bos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [ord(char) + 1 for char in text]


def test_prompt_rendering():
    example = {"query": "2+2?", "choices": ["3", "4"], "gold": 1}
    item = {"query": "3+3?", "choices": ["5", "6"], "gold": 1}
    assert render_prompts(item, "multiple_choice", "\nAnswer: ", [example]) == [
        "2+2?\nAnswer: 4\n\n3+3?\nAnswer: 5",
        "2+2?\nAnswer: 4\n\n3+3?\nAnswer: 6",
    ]


def test_common_prefix_suffix():
    assert common_length([[1, 2, 3], [1, 2, 4]]) == 2
    assert common_length([[1, 2, 3], [4, 2, 3]], suffix=True) == 2


def test_teacher_forced_shift(monkeypatch):
    tokenizer = CharacterTokenizer()
    monkeypatch.setattr("nemochat.eval.core.next_token_outputs",
                        lambda model, ids: ([.2] * (len(ids) - 1), ids[1:]))
    result = evaluate_item(None, tokenizer,
                           {"context": "Question?", "continuation": "answer"},
                           "language_modeling", delimiter="\nAnswer: ")
    assert result["correct"] is True


def test_mean_not_sum_candidate_loss(monkeypatch):
    tokenizer = CharacterTokenizer()

    def outputs(model, ids):
        losses = [0.] * (len(ids) - 1)
        # 'bbbb' has a better mean (1) but worse sum (4) than 'a' (2).
        answer = "".join(chr(t - 1) for t in ids[1:]).split(" ")[-1]
        for i in range(len(answer)):
            losses[-1 - i] = 2. if answer == "a" else 1.
        return losses, [0] * len(losses)

    monkeypatch.setattr("nemochat.eval.core.next_token_outputs", outputs)
    result = evaluate_item(None, tokenizer, {"query": "Q", "choices": ["a", "bbbb"],
                                             "gold": 1}, "multiple_choice")
    assert result["predicted_option"] == 1
    assert result["mean_token_nll"] == [2., 1.]


def test_truncation_does_not_remove_answer():
    tokenizer = CharacterTokenizer()
    with pytest.raises(ValueError, match="truncation"):
        evaluate_item(None, tokenizer, {"context": "Q", "continuation": "long answer"},
                      "language_modeling", context_length=3)


def test_empty_continuation_rejected():
    with pytest.raises(ValueError, match="nonempty"):
        tokenize_spans(CharacterTokenizer(), ["Question", "Question"], "language_modeling")
