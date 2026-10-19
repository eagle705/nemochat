from pathlib import Path
import importlib.util
import json
import pytest

ROOT = Path('/raid/yoon/nemochat')


def module():
    path = ROOT / 'midtraining/publication/hf_kd1000/publish.py'
    assert path.exists(), 'Guarded HF publication is not implemented'
    spec = importlib.util.spec_from_file_location('kd_publication', path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


def test_actual_validation_and_card_contract():
    value = module()
    real = json.loads((ROOT / 'results/midtraining/hf-kd1000-conversion-01/validation_result.json').read_text())
    value.validate(real)
    bad = dict(real, source_step=16)
    with pytest.raises(AssertionError): value.validate(bad)
    text = value.card(ROOT, real)
    for required in ['524,288,000', '1,000', 'CachedKDConfig', 'msc_prefetch_depth=0', 'not** stock', '0.5', '1.0 LM', 'graphs/lr_loss.png', 'graphs/validation_snapshots.png', '13fa979be2e7f7e62913eee0ec5e97c8fd6e24af']:
        assert required in text
    assert 'temperature=1' not in text, 'No nonexistent temperature dataclass field'
