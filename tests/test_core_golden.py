"""Additional golden fixtures for nanochat's pinned schema and few-shot rules."""

import pytest
from nemochat.eval.core import tokenize_spans, render_prompts, evaluate_core_tasks


class CharacterTokenizer:
    bos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [ord(char) + 1 for char in text]


def test_schema_golden_render_and_common_suffix():
    item = {"context_options": ["Alice ran", "Bob ran"], "continuation": " fast", "gold": 0}
    prompts = render_prompts(item, "schema", delimiter="")
    assert prompts == ["Alice ran fast", "Bob ran fast"]
    sequences, spans = tokenize_spans(CharacterTokenizer(), prompts, "schema")
    # Upstream uses the entire common token suffix, not only the literal answer.
    assert ["".join(chr(t - 1) for t in seq[start:end])
            for seq, (start, end) in zip(sequences, spans)] == [" ran fast", " ran fast"]


def test_merged_token_boundary_fails_explicitly():
    class MergingTokenizer:
        bos_token_id = 0

        def encode(self, text, add_special_tokens=False):
            return {"A": [1], "AB": [2]}[text]

    with pytest.raises(ValueError, match="token prefix"):
        tokenize_spans(MergingTokenizer(), ["A", "AB"], "language_modeling")


def test_capped_fewshot_golden_order(monkeypatch):
    class ModelState:
        training = True

        def eval(self):
            self.training = False

        def train(self, mode):
            self.training = mode

    seen = []

    def fake_evaluate(model, tokenizer, item, task_type, **kwargs):
        seen.append((item["id"], kwargs["examples"][0]["id"]))
        return {"correct": True}

    monkeypatch.setattr("nemochat.eval.core.evaluate_item", fake_evaluate)
    model = ModelState()
    result = evaluate_core_tasks(
        model, None,
        [{"label": "fixture", "num_fewshot": [1], "icl_task_type": "multiple_choice"}],
        {"fixture": [{"id": i} for i in range(5)]}, {"fixture": .5}, max_per_task=3,
    )
    assert seen == [(1, 3), (0, 3), (3, 0)]
    assert result["counts"] == {"fixture": 3}
    assert model.training is True
