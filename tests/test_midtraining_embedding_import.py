"""Native HF import must not silently omit the student's learned embedding."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_singular_embedding_alias_uses_native_mapping_without_weight_edits():
    index = json.loads((ROOT/'checkpoints/exports/nemochat_baby_base_hf_bf16_02/model.safetensors.index.json').read_text())['weight_map']
    assert 'backbone.embedding.weight' in index and 'backbone.embeddings.weight' not in index
    path = ROOT/'midtraining/patches/student_embedding_import.py'
    assert path.exists(), 'native import alias repair missing'
    spec=importlib.util.spec_from_file_location('student_embedding_import',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    class Registry:
        def __init__(self,*mappings): self.mappings=list(mappings)
    factory=lambda **kwargs:SimpleNamespace(**kwargs)
    embedding=factory(megatron_param='embedding.word_embeddings.weight',hf_param='backbone.embeddings.weight')
    other=factory(megatron_param='output_layer.weight',hf_param='lm_head.weight')
    original=Registry(embedding,other)
    result=module.rewrite_registry(original,set(index),factory)
    assert result.mappings[0].hf_param=='backbone.embedding.weight'
    assert result.mappings[0].megatron_param=='embedding.word_embeddings.weight'
    assert original.mappings[0].hf_param=='backbone.embeddings.weight'
    assert result.mappings[1] is other
    assert module.rewrite_registry(original,{'backbone.embeddings.weight'},factory) is original
    with pytest.raises(ValueError,match='ambiguous'):
        module.rewrite_registry(original,{'backbone.embedding.weight','backbone.embeddings.weight'},factory)
