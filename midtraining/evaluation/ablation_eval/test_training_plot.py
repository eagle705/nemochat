from pathlib import Path
import importlib.util

ROOT = Path(__file__).resolve().parents[3]


def test_same_step_training_comparison_excludes_unobserved_control_steps(tmp_path):
    source = ROOT / 'midtraining/evaluation/ablation_eval/render_training.py'
    assert source.exists(), 'Matched-step training comparison renderer is missing'
    spec = importlib.util.spec_from_file_location('training_comparison', source)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    data = module.snapshot(ROOT)
    assert 1 <= data['calls'] <= 1000
    assert len(data['B']) == len(data['C']) == data['calls']
    assert data['max_lr_difference'] == 0
    assert data['input_digest_rows_verified'] == data['calls'] * 128
    output = tmp_path / 'comparison'
    report = module.render(data, output)
    assert (output / 'lr_lm_comparison.png').is_file()
    assert (output / 'observed.csv').is_file()
    assert report['last16']['B']['lm_loss'] > 0
    assert report['last16']['C']['lm_loss'] > 0
