"""A text-only view must preserve actual teacher tensor bytes."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_text_view_preserves_language_weights_and_excludes_vision(tmp_path):
    path = ROOT / 'midtraining/patches/teacher_text_view.py'
    assert path.is_file(), 'text-backbone conversion is not implemented'
    spec = importlib.util.spec_from_file_location('teacher_text_view', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    import torch
    from safetensors.torch import save_file, load_file
    source = tmp_path / 'source'; source.mkdir()
    values = {'language_model.backbone.embeddings.weight': torch.arange(12).reshape(3, 4),
              'language_model.lm_head.weight': torch.ones(3, 4),
              'language_model.mtp.layers.0.weight': torch.ones(4),
              'vision_model.weight': torch.zeros(4)}
    save_file(values, str(source / 'model.safetensors'))
    (source / 'config.json').write_text(json.dumps({'model_type': 'nemotron_h_omni',
        'llm_config': {'model_type': 'nemotron_h', 'architectures': ['NemotronHForCausalLM'],
                       'num_nextn_predict_layers': 1, 'vocab_size': 131072}}))
    (source / 'model.safetensors.index.json').write_text(json.dumps({'weight_map':
        {key: 'model.safetensors' for key in values}}))
    output = tmp_path / 'text'
    receipt = module.convert(source, output)
    loaded = load_file(str(output / 'model.safetensors'))
    assert set(loaded) == {'backbone.embeddings.weight', 'lm_head.weight'}
    assert torch.equal(loaded['backbone.embeddings.weight'], values['language_model.backbone.embeddings.weight'])
    assert torch.equal(loaded['lm_head.weight'], values['language_model.lm_head.weight'])
    assert receipt['language_tensors'] == 2
    config = json.loads((output / 'config.json').read_text())
    assert config['num_nextn_predict_layers'] == 0
    assert config['model_type'] == 'nemotron_h'
    assert 'auto_map' not in config
