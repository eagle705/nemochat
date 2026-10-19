"""Opt-in observations must not repair or replace native forward outputs."""
import importlib.util
import json
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]


def test_probe_records_finite_and_nonfinite_without_replacing_outputs(tmp_path):
    path = ROOT / 'midtraining/patches/kd_finite_probe.py'
    assert path.exists(), 'bounded finite-value diagnostic is missing'
    spec = importlib.util.spec_from_file_location('kd_finite_probe', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.output_layer = torch.nn.Identity()
        def forward(self, values):
            return self.output_layer(values)
    model = Model(); log = tmp_path / 'events.jsonl'
    handles = module.install(model, 'student', log)
    value = torch.tensor([2., 3.], requires_grad=True)
    assert model(value) is value
    value.sum().backward(); assert value.grad.tolist() == [1., 1.]
    bad = torch.tensor([float('nan'), 1.])
    assert model(bad) is bad
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert rows[0]['model'] == 'student'
    assert rows[0]['nonfinite_elements'] == 0
    assert rows[-1]['nonfinite_elements'] == 1
    assert all(row['observation_only'] for row in rows)
    assert handles
