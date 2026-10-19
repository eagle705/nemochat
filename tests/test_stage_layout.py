"""Portable contracts for stage-owned source, not historical GPU qualification."""
from pathlib import Path
import subprocess
import sys
import os

ROOT = Path(__file__).resolve().parents[1]


def test_stage_source_ownership():
    for stage in ('pretraining', 'midtraining', 'sft', 'rl'):
        assert (ROOT / stage / 'README.md').is_file(), stage
    assert not (ROOT / 'midtraining/sft_after_kd').exists()
    assert not (ROOT / 'midtraining/publication/hf_sft1200').exists()
    assert (ROOT / 'sft/training/native_sft_v1.py').is_file()
    assert (ROOT / 'sft/resume/runtime/resume_sft.py').is_file()
    assert (ROOT / 'sft/publication/hf_sft1200/prepare.py').is_file()
    assert (ROOT / 'midtraining/training/offline_kd/native.py').is_file()
    assert (ROOT / 'midtraining/data/prepare_corpus.py').is_file()
    assert (ROOT / 'rl/evaluation/gym_eval.py').is_file()


def test_stage_entrypoints_offline_help():
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1',
               TRANSFORMERS_OFFLINE='1', WANDB_MODE='disabled',
               PYTHONPATH=str(ROOT / 'src'))
    for entry in ('sft/data/nemotron_sft.py', 'sft/data/prepare_tokenizer.py',
                  'sft/training/launch_nemotron_sft.py', 'rl/evaluation/gym_eval.py',
                  'midtraining/data/download_teacher.py'):
        result = subprocess.run([sys.executable, str(ROOT / entry), '--help'],
                                cwd=ROOT, env=env, text=True, capture_output=True, timeout=30)
        assert result.returncode == 0, (entry, result.stderr)
        assert 'usage:' in result.stdout.lower(), entry


def test_no_legacy_source_import_paths():
    forbidden = ('midtraining/sft_after_kd', 'scripts/posttraining',
                 'midtraining/publication/hf_sft1200')
    for stage in ('pretraining', 'midtraining', 'sft', 'rl'):
        for path in (ROOT / stage).rglob('*.py'):
            if 'evidence' in path.parts or '__pycache__' in path.parts:
                continue
            text = path.read_text()
            assert not any(old in text for old in forbidden), str(path.relative_to(ROOT))


def test_stage_source_directory_constants_resolve():
    """Catch composed HERE/CODE directories, not just whole-file strings."""
    import ast
    for stage in ('pretraining', 'midtraining', 'sft', 'rl'):
        for path in (ROOT / stage).rglob('*.py'):
            if 'evidence' in path.parts or path.name.startswith('test_'):
                continue
            for node in ast.parse(path.read_text()).body:
                if not isinstance(node, ast.Assign):
                    continue
                expr = node.value
                if not (isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Div)
                        and isinstance(expr.left, ast.Name) and expr.left.id in ('ROOT', 'REPO')
                        and isinstance(expr.right, ast.Constant) and isinstance(expr.right.value, str)):
                    continue
                relative = Path(expr.right.value)
                # JSON request/acceptance records may be deliberately excluded
                # historical runtime evidence, even when a source references them.
                if (relative.parts and relative.parts[0] in ('pretraining', 'midtraining', 'sft', 'rl')
                        and relative.suffix in ('', '.py', '.yaml', '.sh')):
                    assert (ROOT / relative).exists(), (path.relative_to(ROOT), relative)


def test_production_corpus_default_preview_is_offline_and_inert(tmp_path):
    """Exercise the real default CLI, without downloading or preparing a corpus."""
    import json
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1',
               TRANSFORMERS_OFFLINE='1', WANDB_MODE='disabled')
    result = subprocess.run(
        [sys.executable, str(ROOT / 'midtraining/data/production_corpus.py'), '--preview'],
        cwd=tmp_path, env=env, text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    preview = json.loads(result.stdout)
    assert preview['execute'] is False
    assert preview['acquisition_bytes'] == 0
    assert not list(tmp_path.iterdir())


def test_stage_config_defaults_resolve():
    import yaml
    import importlib.util
    for name in ('midtraining/data/prepare_corpus.py',
                 'midtraining/data/recover_corpus.py',
                 'midtraining/training/native_kd.py'):
        spec = importlib.util.spec_from_file_location('stage_config_probe', ROOT / name)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert module.DEFAULT_CONFIG.is_file()
        assert isinstance(yaml.safe_load(module.DEFAULT_CONFIG.read_text()), dict)
    for name in ('sft/configs/recipe-v1.yaml', 'sft/configs/tokenizer_super35.yaml',
                 'sft/configs/nemotron_dataset_settings.yaml'):
        assert isinstance(yaml.safe_load((ROOT / name).read_text()), dict)
