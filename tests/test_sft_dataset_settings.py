"""CPU-only registry controls; no model loading or training acceptance claims."""
import importlib.util
import copy
import json
import pytest

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_missing_local_evidence_is_explicit_not_a_readiness_pass(tmp_path):
    from nemochat.data import sft_dataset_settings as mod
    settings = mod.load_settings()
    with pytest.raises(ValueError, match='local evidence unavailable'):
        mod.validate_settings(settings, root=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_registry_has_five_pinned_datasets_and_explicit_stage_membership():
    assert importlib.util.find_spec('nemochat.data.sft_dataset_settings') is not None, 'consuming settings module missing'
    from nemochat.data.sft_dataset_settings import load_settings
    settings = load_settings()
    assert len(settings['datasets']) == 5
    stages = [row['stage'] for row in settings['datasets']]
    assert stages.count('current_sft') == 3
    assert stages.count('deferred_long_context') == 1
    assert stages.count('reserved_rl_seed') == 1
    assert sum(len(row['sources']) for row in settings['datasets']) == 8


def test_validation_crosschecks_actual_discovery_runtime_reservations_and_long_plan():
    from nemochat.data import sft_dataset_settings as mod
    assert hasattr(mod, 'validate_settings'), 'strict validation missing'
    result = mod.validate_settings(mod.load_settings())
    assert result['status'] == 'passed'
    assert result['counts'] == {'datasets': 5, 'current_sft': 3, 'deferred_long_context': 1,
                                'reserved_rl_seed': 1, 'source_files': 8, 'labeled_source_files': 7}
    assert result['reserved_prompt_groups'] == 14
    assert result['kmmlu_reserved_questions'] == 34871
    assert result['schema_rejected_records'] == 39
    assert result['runtime_crosschecks'] is True
    assert result['native_128k_launch_ready'] is False


@pytest.mark.parametrize('mutation', [
    lambda s: s['datasets'].pop(),
    lambda s: s['datasets'][0].update(revision='0'*40),
    lambda s: s['datasets'][0].update(stage='deferred_long_context'),
    lambda s: s['datasets'][0].update(train='true'),
    lambda s: s['datasets'][0].update(sources=['data/not-pinned.jsonl']),
    lambda s: s['datasets'][-1].update(train=True),
    lambda s: s['datasets'][3].update(sequence_length=16384),
    lambda s: s['stages']['current_sft'].update(context_parallel_size=2),
    lambda s: s['stages']['deferred_long_context'].update(auto_launch=True),
    lambda s: s['stages']['deferred_long_context'].update(initial_context_parallel_size=2),
    lambda s: s['preparation'].update(truncate=True),
    lambda s: s['preparation'].update(selection='fixed_mixture'),
    lambda s: s['tokenizer'].update(revision='0'*40),
    lambda s: s['datasets'][1].update(train_turns_mask='assistant_subsequence'),
    lambda s: s['preparation'].update(tool_policy='flatten_to_text'),
    lambda s: s['preparation'].update(reasoning_content='include_thinking'),
    lambda s: s['preparation'].update(mask_policy='all_tokens'),
    lambda s: s['preparation'].update(inactive_excluded_source_groups='force_reinclude_Long'),
    lambda s: s['preparation'].update(parity='skip'),
    lambda s: s['stages']['deferred_long_context'].update(native_cp_validation='passed'),
])
def test_validation_rejects_policy_pin_stage_and_runtime_drift(mutation):
    from nemochat.data import sft_dataset_settings as mod
    assert hasattr(mod, 'validate_settings'), 'strict validation missing'
    settings = copy.deepcopy(mod.load_settings())
    mutation(settings)
    with pytest.raises(ValueError):
        mod.validate_settings(settings)


@pytest.mark.parametrize('stage,length,active,excluded', [
    ('current_sft', 16384, 3, ['nvidia/Nemotron-SFT-Long-Context-Transduction-v1']),
    ('deferred_long_context', 131072, 1, ['nvidia/Nemotron-SFT-Agentic-v4', 'nvidia/Nemotron-SFT-BBQ-v1',
      'nvidia/Nemotron-SFT-Instruction-Following-Chat-v4']),
])
def test_compiler_consumes_stage_and_emits_real_prepare_parser_flags(stage, length, active, excluded):
    from nemochat.data import sft_dataset_settings as mod
    import subprocess
    import sys
    assert hasattr(mod, 'compile_plan'), 'stage plan compiler missing'
    output = 'results/pretraining/nemochat_baby/nemotron_sft/dataset_settings_code/not-created-by-preview'
    plan = mod.compile_plan(mod.load_settings(), stage, raw='datasets/nemotron_sft/nemochat_baby/raw-full-01', output=output)
    argv = plan['prepare_argv']
    assert argv[:3] == [sys.executable, str(ROOT / 'sft/data/nemotron_sft.py'), 'prepare']
    assert argv[argv.index('--sequence-length')+1] == str(length)
    assert len(plan['active_datasets']) == active
    assert plan['excluded_datasets'] == excluded
    assert [argv[i+1] for i, arg in enumerate(argv) if arg == '--exclude-dataset'] == excluded
    assert int(argv[argv.index('--quota-per-file')+1]) > 54283343615  # physical byte bound, not a mixture quota
    assert '--reserve-heldout' in argv and '--kmmlu-exclusions' in argv
    assert not (ROOT/output).exists()
    help_result = subprocess.run(argv[:3]+['--help'], capture_output=True, text=True, check=True)
    for arg in argv[3:]:
        if arg.startswith('--'):
            assert arg in help_result.stdout
    assert plan['native_training_argv'] is None
    assert plan['execution_environment'] == {'CUDA_VISIBLE_DEVICES': '', 'TOKENIZERS_PARALLELISM': 'false'}


def test_reserved_stage_has_no_preparation_command():
    from nemochat.data import sft_dataset_settings as mod
    assert hasattr(mod, 'compile_plan'), 'stage plan compiler missing'
    plan = mod.compile_plan(mod.load_settings(), 'reserved_rl_seed')
    assert plan['prepare_argv'] is None
    assert plan['active_datasets'] == []


@pytest.mark.parametrize('output', ['../escape', '/tmp/escape', 'sft/configs/nemotron_dataset_settings.yaml'])
def test_compile_rejects_unsafe_or_existing_output(output):
    from nemochat.data import sft_dataset_settings as mod
    assert hasattr(mod, 'compile_plan'), 'stage plan compiler missing'
    with pytest.raises(ValueError):
        mod.compile_plan(mod.load_settings(), 'current_sft', raw='datasets', output=output)


def test_actual_wrapper_default_check_json_and_preview_do_not_write():
    import subprocess
    import sys
    script = ROOT/'sft/data/nemotron_dataset_settings.py'
    assert script.exists(), 'consuming wrapper CLI missing'
    from nemochat.data.sft_dataset_settings import load_settings
    discovery_available = (ROOT / load_settings()['discovery']).is_dir()
    for args in ([], ['--check', '--json'], ['--preview', '--stage', 'deferred_long_context', '--json']):
        result = subprocess.run([sys.executable, str(script), *args], cwd=ROOT, capture_output=True, text=True)
        if not discovery_available:
            # A source-only checkout must reject absent historical evidence, not
            # import an editable original checkout and falsely report readiness.
            assert result.returncode != 0
            assert 'local evidence unavailable' in result.stderr
            assert not result.stdout.strip()
            continue
        assert result.returncode == 0, result.stderr
        if '--json' in args:
            parsed = json.loads(result.stdout)
            assert parsed['validation']['status'] == 'passed'
            assert parsed['native_training_argv'] is None
    help_result = subprocess.run([sys.executable, str(script), '--help'], capture_output=True, text=True, check=True)
    assert '--execute-prepare' in help_result.stdout


def test_wrapper_plan_write_is_fresh_and_cpu_execution_guard_is_fail_closed():
    import subprocess
    import sys
    import tempfile
    script = ROOT/'sft/data/nemotron_dataset_settings.py'
    assert script.exists(), 'consuming wrapper CLI missing'
    parent = ROOT/'results/pretraining/nemochat_baby/nemotron_sft/dataset_settings_code'
    with tempfile.TemporaryDirectory(dir=parent) as temporary:
        path = Path(temporary)/'plan.json'
        command = [sys.executable, str(script), '--write-plan', str(path), '--json']
        first = subprocess.run(command, capture_output=True, text=True)
        assert first.returncode == 0, first.stderr
        original = path.read_bytes()
        assert json.loads(original)['stage'] == 'current_sft'
        second = subprocess.run(command, capture_output=True, text=True)
        assert second.returncode != 0 and 'fresh' in second.stderr
        assert path.read_bytes() == original
        output = Path(temporary)/'never-prepared'
        result = subprocess.run([sys.executable, str(script), '--execute-prepare', '--raw', 'datasets',
                                 '--output', str(output)], capture_output=True, text=True)
        assert result.returncode != 0
        assert not output.exists()
        assert 'runtime' in result.stderr.lower() or 'receipt' in result.stderr.lower()


def test_safe_path_rejects_symlink_escape_and_existing_output(tmp_path):
    from nemochat.data.sft_dataset_settings import repository_path
    link = tmp_path/'escape'
    link.symlink_to('/tmp', target_is_directory=True)
    with pytest.raises(ValueError):
        repository_path(link/'new.json', root=tmp_path, fresh=True)
    existing = tmp_path/'existing'
    existing.mkdir()
    with pytest.raises(ValueError):
        repository_path(existing, root=tmp_path, fresh=True)
    dangling = tmp_path/'dangling'
    dangling.symlink_to(tmp_path/'not-created')
    with pytest.raises(ValueError):
        repository_path('dangling', root=tmp_path, fresh=True)


def test_actual_cli_rejects_mutated_temporary_registry_without_writes():
    import subprocess
    import sys
    import tempfile
    import yaml
    from nemochat.data.sft_dataset_settings import load_settings
    parent = ROOT/'results/pretraining/nemochat_baby/nemotron_sft/dataset_settings_code'
    with tempfile.TemporaryDirectory(dir=parent) as temporary:
        path = Path(temporary)/'wrong.yaml'
        settings = load_settings()
        settings['datasets'][3]['stage'] = 'current_sft'
        path.write_text(yaml.safe_dump(settings))
        plan = Path(temporary)/'not-created.json'
        result = subprocess.run([sys.executable, str(ROOT/'sft/data/nemotron_dataset_settings.py'),
                                 '--config', str(path), '--check', '--write-plan', str(plan)],
                                capture_output=True, text=True)
        assert result.returncode != 0 and 'stage mismatch' in result.stderr
        assert not plan.exists()
