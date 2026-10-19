"""CPU protocol controls, not native training evidence."""
import json
from pathlib import Path
import runpy
import numpy as np
import pytest
import yaml

NATIVE = runpy.run_path('sft/training/native_nemotron_sft.py')
LAUNCH = runpy.run_path('sft/training/launch_nemotron_sft.py')


def config(**kwargs):
    c = yaml.safe_load(Path('sft/configs/nemochat_baby_nemotron_sft.yaml').read_text())
    c.update(execution_mode='longrun', epochs=3, max_hours=48, max_updates=14000,
             allow_repetition=True, watchdog_grace_minutes=60, validation_max_dialogues=1000,
             estimated_iteration_seconds=11.510017357512954, assumed_overhead_fraction=.1)
    c.update(kwargs)
    return c


def manifest(n=640000):
    c = config()
    return dict(status='passed', schema='nemochat-nemotron-sft-v1',
                format='official-super35-selected-assistant-v1', sequence_length=4096,
                train_dialogues=n, heldout_dialogues=3200, train_heldout_overlap=0,
                kmmlu_exact_question_matches=0, kmmlu_reserved_questions=34871,
                training_repetition=False, prior_heldout_reservation_file=c['prior_heldout_reservation'],
                prior_reserved_prompt_groups=14)


def test_longrun_explicit_epochs_and_control_cannot_bypass_main_gate():
    plan = NATIVE['stage_plan'](config(), manifest())
    assert plan['updates'] == 14000
    assert plan['available_updates'] == 15000
    assert plan['unique_dialogues'] == 640000
    assert plan['planned_exposures'] == 1792000
    with pytest.raises(ValueError, match='50000'):
        NATIVE['stage_plan'](config(), manifest(82))
    with pytest.raises(ValueError, match='repetition'):
        NATIVE['stage_plan'](config(allow_repetition=False), manifest())
    with pytest.raises(ValueError, match='duration-control'):
        NATIVE['stage_plan'](config(execution_mode='duration_control', max_hours=1/60), manifest(82), pilot=True)
    control = NATIVE['stage_plan'](config(execution_mode='duration_control', epochs=100,
        max_hours=1/60, max_updates=1000), manifest(82), pilot=True, duration_control=True)
    assert control['global_batch_size'] == 8 and control['updates'] == 1000
    with pytest.raises(ValueError, match='5 minutes'):
        NATIVE['stage_plan'](config(execution_mode='duration_control'), manifest(82), pilot=True, duration_control=True)


def test_seeded_epochs_keep_zero_loss_tail_and_exact_prefix_coverage(tmp_path):
    tokens = np.tile(np.arange(4097), (10, 1)).astype(np.uint32)
    masks = np.zeros((10, 4096), dtype=np.uint8)
    for i in range(10):
        tokens[i, 0] = i + 100
        masks[i, :i+1] = 1
    np.save(tmp_path/'train-tokens.npy', tokens)
    np.save(tmp_path/'train-masks.npy', masks)
    rows = NATIVE['PreparedRows'](tmp_path, 'train', 48, 8, epochs=3, seed=42)
    ids = [[int(rows[e*16+i]['tokens'][0]) for i in range(10)] for e in range(3)]
    assert all(sorted(x) == list(range(100,110)) for x in ids)
    assert ids[0] != ids[1] != ids[2]
    assert all(int(rows[e*16+i]['loss_mask'].sum()) == 0 for e in range(3) for i in range(10,16))
    expected_loss = 55 + sum(i-99 for i in ids[1][:8])
    assert rows.coverage(3) == dict(loss_tokens=expected_loss, exposures=18, unique_dialogues=10,
                                    tail_rows=6, processed_positions=24*4096)
    with pytest.raises(ValueError, match='epoch'):
        NATIVE['PreparedRows'](tmp_path, 'train', 49, 8, epochs=3)


def test_config_is_forwarded_and_duration_receipt_accepts_only_saved_prefix(tmp_path):
    payload = LAUNCH['make_payload']('duration-test', Path('datasets/pilot/manifest.json'),
        pilot=True, config=Path('sft/configs/nemochat_baby_nemotron_sft_48h.yaml'), duration_control=True)
    assert '--config' in payload['Cmd'] and '--duration-control' in payload['Cmd']
    assert payload['Cmd'][payload['Cmd'].index('--config')+1].endswith('_48h.yaml')
    # Handwritten lifecycle fixture, not GPU evidence.
    plan = NATIVE['stage_plan'](config(execution_mode='duration_control', epochs=100,
        max_hours=1/60, max_updates=1000), manifest(82), pilot=True, duration_control=True)
    (tmp_path/'container-state.json').write_text(json.dumps(dict(state=dict(Running=False, ExitCode=0, Pid=0, OOMKilled=False))))
    for rank in range(8):
        for kind, value in dict(load=dict(source=plan['source_checkpoint'], source_sft_update=391,
            stage_step=0, model_only=True), entrypoint=dict(native_exit_code=0, updates=1,
            termination_reason='native_duration_limit', elapsed_seconds=70),
            save=dict(save_steps=[1], save_optim=True, save_rng=True),
            coverage=dict(loss_tokens=5, processed_positions=8*4096, exposures=8,
                          unique_dialogues=8, tail_rows=0)).items():
            (tmp_path/f'rank-{rank}-{kind}.json').write_text(json.dumps(value))
        (tmp_path/f'rank-{rank}-updates.jsonl').write_text(json.dumps(dict(update=1, loss=1., grad_norm=1.,
            lr=1e-5, finite_positive_gradient=True, global_loss_tokens=5, global_processed_positions=8*4096))+'\n')
    result = LAUNCH['verify_run'](tmp_path, plan, manifest(82))
    assert result['updates'] == 1 and result['termination_reason'] == 'native_duration_limit'
    assert result['target_duration_reached'] is True
    with pytest.raises(ValueError, match='final heldout'):
        LAUNCH['verify_run'](tmp_path, dict(plan,require_final_validation=True), manifest(82))
    for rank in range(8):
        (tmp_path/f'rank-{rank}-heldout.jsonl').write_text(json.dumps(dict(update=1,heldout_ce=1.,heldout_ppl=2.718))+'\n')
    assert LAUNCH['verify_run'](tmp_path, dict(plan,require_final_validation=True), manifest(82))['updates'] == 1
    path = tmp_path/'rank-7-entrypoint.json'
    path.write_text(json.dumps(dict(native_exit_code=0, updates=1, termination_reason='unexpected_exit', elapsed_seconds=1)))
    with pytest.raises(ValueError, match='premature'):
        LAUNCH['verify_run'](tmp_path, plan, manifest(82))


def test_watchdog_grace_and_durable_supervisor_contract():
    assert LAUNCH['watchdog_seconds'](config()) == 49*3600
    control = config(max_hours=1/60, watchdog_grace_minutes=10)
    assert LAUNCH['watchdog_seconds'](control) == 660
    cmd = LAUNCH['supervisor_command']('longrun-01', ['--name','longrun-01','--execute','--supervise'], config())
    assert '--user' in cmd and '--service-type=exec' in cmd
    assert '--property=RuntimeMaxSec=176640' in cmd
    assert '--supervise' not in cmd
    assert '--property=KillMode=control-group' in cmd
    assert '--setenv=NEMOTRON_SFT_SUPERVISED=longrun-01' in cmd
    assert any(arg.startswith('--property=ExecStopPost=') and '--cleanup-owned' in arg for arg in cmd)


def test_longrun_blocks_short_corpus_budget_and_requires_native_calibration():
    with pytest.raises(ValueError, match='48h|duration budget'):
        NATIVE['stage_plan'](config(), manifest(64000))
    plan = NATIVE['stage_plan'](config(), manifest())
    with pytest.raises(ValueError, match='calibration'):
        LAUNCH['validate_calibration'](plan, Path('missing-manifest.json'))
    calibration = NATIVE['stage_plan'](config(execution_mode='calibration', epochs=1,
        max_updates=20, max_hours=.25, allow_repetition=False), manifest(64000))
    assert calibration['updates'] == 20 and calibration['global_batch_size'] == 128


def test_fixed_stratified_heldout_window_repeats_exactly_with_masked_tail(tmp_path):
    tokens = np.tile(np.arange(4097), (10, 1)).astype(np.uint32)
    tokens[:,0] = np.arange(100,110)
    np.save(tmp_path/'heldout-tokens.npy', tokens)
    np.save(tmp_path/'heldout-masks.npy', np.ones((10,4096),dtype=np.uint8))
    (tmp_path/'heldout-dialogues.jsonl').write_text(''.join(json.dumps(dict(source_key='a' if i<8 else 'b'))+'\n' for i in range(10)))
    rows = NATIVE['PreparedRows'](tmp_path, 'heldout', 32, 8, window=5)
    first = [int(rows[i]['tokens'][0]) for i in range(5)]
    assert any(x>=108 for x in first)
    assert [int(rows[8+i]['tokens'][0]) for i in range(5)] == first
    assert all(int(rows[i]['loss_mask'].sum()) == 0 for i in (5,6,7,13,14,15))
    assert NATIVE['final_evaluation_due'](1,61,dict(updates=1000,max_hours=1/60))
    assert NATIVE['final_evaluation_due'](1000,1,dict(updates=1000,max_hours=48))
    assert not NATIVE['final_evaluation_due'](1,1,dict(updates=1000,max_hours=48))


def test_final_evaluation_can_complete_after_deadline_and_restores_native_limit():
    from types import SimpleNamespace
    train = SimpleNamespace(exit_duration_in_mins=1.)
    limit = NATIVE['suspend_evaluation_deadline'](train)
    assert limit == 1. and train.exit_duration_in_mins is None
    NATIVE['restore_evaluation_deadline'](train, limit)
    assert train.exit_duration_in_mins == 1.


def test_calibration_must_reference_accepted_native_run(tmp_path):
    import hashlib
    path = tmp_path/'manifest.json'
    path.write_text('{}')
    receipt = tmp_path/'calibration.json'
    receipt.write_text(json.dumps(dict(status='passed', global_batch_size=128,
        manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), warm_updates=15,
        source_checkpoint=config()['source_checkpoint'], ranks=list(range(8)),
        mean_iteration_seconds=12., verification=str(tmp_path/'missing-verification.json'))))
    with pytest.raises(ValueError, match='accepted native'):
        LAUNCH['validate_calibration'](NATIVE['stage_plan'](config(calibration_receipt=str(receipt)),manifest()), path)


def test_preview_cannot_be_combined_with_execute():
    import subprocess
    import sys
    result = subprocess.run([sys.executable,'sft/training/launch_nemotron_sft.py',
        '--name','never-launch','--manifest','missing-manifest.json','--execute','--preview'],
        capture_output=True,text=True)
    assert result.returncode == 2
    assert 'not allowed with argument' in result.stderr


def test_runner_provenance_hashes_only_source_configuration_and_manifest(tmp_path):
    c = tmp_path/'config.yaml'
    m = tmp_path/'manifest.json'
    c.write_text('execution_mode: control\n')
    m.write_text('{}')
    result = LAUNCH['runner_provenance'](c,m)
    import hashlib
    assert result['config_sha256'] == hashlib.sha256(c.read_bytes()).hexdigest()
    assert result['manifest_sha256'] == hashlib.sha256(m.read_bytes()).hexdigest()
    assert len(result['scripts_sha256']) == 2
    assert result['checkpoint_bytes_read'] is False


def test_additional_epochs_require_separately_frozen_repetition_authorization():
    with pytest.raises(ValueError, match='epoch'):
        NATIVE['stage_plan'](config(epochs=32),manifest(64000))
    plan = NATIVE['stage_plan'](config(epochs=32,authorized_max_epochs=32),manifest(64000))
    assert plan['updates'] == 14000 and plan['unique_dialogues'] == 64000
    assert plan['planned_exposures'] == 1792000
