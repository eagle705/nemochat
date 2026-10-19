"""Compatibility adapters stay separate from the native Bridge trainer."""
import importlib.util
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_reserved_alias_gate_reuses_exact_accepted_migration(tmp_path):
    path = ROOT / 'midtraining/patches/tokenizer_alias_gate.py'
    assert path.exists(), 'explicit tokenizer-alias compatibility gate is missing'
    spec = importlib.util.spec_from_file_location('tokenizer_alias_gate', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    root = tmp_path
    original = json.loads((ROOT / 'checkpoints/exports/nemochat_baby_base_hf_bf16_02/tokenizer.json').read_text())
    adopted = json.loads((ROOT / 'datasets/tokenizers/nemotron_super35/fc067b34f6b514014d2f8bddba855d2e4bfdfba6/tokenizer.json').read_text())
    (root / 'student').mkdir(); (root / 'teacher').mkdir(); (root / 'accepted').mkdir()
    for folder, payload in [('student', original), ('teacher', adopted), ('accepted', adopted)]:
        (root / folder / 'tokenizer.json').write_text(json.dumps(payload))
    manifest = json.loads((ROOT / 'results/pretraining/nemochat_baby/tokenizer_migration/manifest.json').read_text())
    manifest['tokenizer_path'] = 'accepted'
    (root / 'manifest.json').write_text(json.dumps(manifest))
    result = module.check(root / 'student', root / 'teacher', root / 'manifest.json', root)
    assert result['changed_reserved_ids'] == list(range(18, 30))
    assert result['token_id_semantics_equal'] is False
    assert result['migration_reused'] is True
    assert result['ordinary_text_ids_equal'] is True
    assert result['weight_permutation_performed'] is False
    native_spec = importlib.util.spec_from_file_location('native_kd', ROOT / 'midtraining/native_kd.py')
    native = importlib.util.module_from_spec(native_spec); native_spec.loader.exec_module(native)
    assert native.tokenizer_gate(root / 'student', root / 'teacher')['migration_reused']
    adopted['model']['vocab']['a'], adopted['model']['vocab']['b'] = adopted['model']['vocab']['b'], adopted['model']['vocab']['a']
    (root / 'teacher/tokenizer.json').write_text(json.dumps(adopted))
    with pytest.raises(ValueError, match='accepted'):
        module.check(root / 'student', root / 'teacher', root / 'manifest.json', root)
