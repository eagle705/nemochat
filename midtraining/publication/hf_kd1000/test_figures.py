from pathlib import Path
import importlib.util

ROOT = Path('/raid/yoon/nemochat')


def test_render_only_observed_complete_run_without_replacing_outputs(tmp_path):
    code = ROOT / 'midtraining/publication/hf_kd1000/figures.py'
    assert code.is_file(), 'The observed-run card figure renderer is not implemented'
    spec = importlib.util.spec_from_file_location('card_figures', code)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = tmp_path / 'figures'
    receipt = module.render(ROOT, output)
    assert receipt['calls'] == 1000
    assert len(receipt['validation_observations']) == 5
    end = [v for v in receipt['validation_observations'] if v['step'] == 1000]
    assert {v['phase'] for v in end} == {'interval', 'end_of_training'}
    assert len(receipt['files']) == 3
    for name in receipt['files']:
        assert (output / name).stat().st_size > 1000
    import pytest
    with pytest.raises(FileExistsError):
        module.render(ROOT, output)
