"""16K runtime CPU contracts; native evidence is separate."""
import json
from pathlib import Path
import runpy
import numpy as np
import pytest
import yaml

N = runpy.run_path('scripts/posttraining/native_nemotron_sft.py')
L = runpy.run_path('scripts/posttraining/launch_nemotron_sft.py')
LONG = 'nvidia/Nemotron-SFT-Long-Context-Transduction-v1'


def test_16k_rows_shift_tail_and_coverage(tmp_path):
    tokens = np.tile(np.arange(16385, dtype=np.uint32), (9, 1))
    masks = np.zeros((9, 16384), dtype=np.uint8)
    masks[:, 5000:6000] = 1
    np.save(tmp_path/'train-tokens.npy', tokens)
    np.save(tmp_path/'train-masks.npy', masks)
    rows = N['PreparedRows'](tmp_path, 'train', 16, 8, sequence_length=16384)
    item = rows[0]
    assert item['tokens'].shape == item['labels'].shape == (16384,)
    assert int(item['labels'][-1]) == 16384
    assert int(item['position_ids'][-1]) == 16383
    assert int(item['loss_mask'][4096:].sum()) == 1000
    assert rows[15]['tokens'].shape == (16384,)
    assert int(rows[15]['loss_mask'].sum()) == 0
    assert rows.coverage(2)['processed_positions'] == 16*16384
    assert rows.coverage(2)['loss_tokens'] == 9000
    with pytest.raises(ValueError, match='aligned'):
        N['PreparedRows'](tmp_path, 'train', 16, 8)


def test_16k_plan_requires_whole_dataset_exclusion_and_cp1_ep8():
    c = yaml.safe_load(Path('configs/posttraining/nemochat_baby_nemotron_sft.yaml').read_text())
    c.update(sequence_length=16384, excluded_datasets=[LONG], context_parallel_size=1,
             expert_model_parallel_size=8)
    m = dict(status='passed', schema='nemochat-nemotron-sft-v1',
             format='official-super35-selected-assistant-v1', sequence_length=16384,
             train_dialogues=32, heldout_dialogues=3, train_heldout_overlap=0,
             kmmlu_exact_question_matches=0, kmmlu_reserved_questions=34871,
             training_repetition=False, excluded_datasets=[LONG])
    plan = N['stage_plan'](c, m, pilot=True)
    assert plan['updates'] == 4 and plan['sequence_length'] == 16384
    with pytest.raises(ValueError, match='exclusion'):
        N['stage_plan'](c, dict(m, excluded_datasets=[]), pilot=True)
    with pytest.raises(ValueError, match='CP1'):
        N['stage_plan'](dict(c, context_parallel_size=2), m, pilot=True)


def test_provider_runtime_width_preserves_all_parameter_dimensions():
    from types import SimpleNamespace
    provider = SimpleNamespace(seq_length=4096, max_sequence_length=4096,
        hidden_size=1024, vocab_size=131072, context_parallel_size=1, expert_model_parallel_size=8)
    N['configure_provider_runtime'](provider, dict(sequence_length=16384))
    assert provider.seq_length == provider.max_sequence_length == 16384
    assert provider.hidden_size == 1024 and provider.vocab_size == 131072
    assert provider.context_parallel_size == 1 and provider.expert_model_parallel_size == 8


def test_input_policy_accepts_three_active_sources_but_rejects_excluded_heldout(tmp_path):
    m = json.loads(Path('datasets/nemotron_sft/nemochat_baby/prepared-pilot-01/data/manifest.json').read_text())
    m.update(excluded_datasets=[LONG], files={})
    key = next(s['key'] for s in m['sources'] if s['dataset'] == LONG)
    m['per_source'][key] = dict(excluded_dataset_rows=405)
    p = tmp_path/'manifest.json'
    p.write_text(json.dumps(m))
    migration = 'results/pretraining/nemochat_baby/tokenizer_migration/manifest.json'
    assert N['verify_inputs'](p, migration)['data_files_verified'] == 0
    m['per_source'][key]['heldout'] = 1
    p.write_text(json.dumps(m))
    with pytest.raises(ValueError, match='excluded'):
        N['verify_inputs'](p, migration)


def test_launcher_accepts_actual_16k_all8_positions(tmp_path):
    c = yaml.safe_load(Path('configs/posttraining/nemochat_baby_nemotron_sft.yaml').read_text())
    c.update(sequence_length=16384, excluded_datasets=[LONG])
    m = dict(status='passed', schema='nemochat-nemotron-sft-v1',
        format='official-super35-selected-assistant-v1', sequence_length=16384,
        train_dialogues=32, heldout_dialogues=3, train_heldout_overlap=0,
        kmmlu_exact_question_matches=0, kmmlu_reserved_questions=34871,
        training_repetition=False, excluded_datasets=[LONG], train_loss_tokens=400)
    plan = N['stage_plan'](c,m,pilot=True)
    (tmp_path/'container-state.json').write_text(json.dumps(dict(state=dict(Running=False,ExitCode=0,Pid=0,OOMKilled=False))))
    for rank in range(8):
        for kind,value in dict(load=dict(source=plan['source_checkpoint'],source_sft_update=391,stage_step=0,model_only=True),
            entrypoint=dict(native_exit_code=0,updates=4),save=dict(save_steps=[4],save_optim=True,save_rng=True)).items():
            (tmp_path/f'rank-{rank}-{kind}.json').write_text(json.dumps(value))
        updates = [dict(update=i,loss=1.,grad_norm=1.,lr=1e-5,finite_positive_gradient=True,
            global_loss_tokens=100,global_processed_positions=8*16384) for i in range(1,5)]
        (tmp_path/f'rank-{rank}-updates.jsonl').write_text(''.join(json.dumps(u)+'\n' for u in updates))
    assert L['verify_run'](tmp_path,plan,m)['updates'] == 4


def test_16k_protocol_configs_freeze_no_repetition_and_fresh_calibration():
    c = yaml.safe_load(Path('configs/posttraining/nemochat_baby_nemotron_sft_16k_48h.yaml').read_text())
    assert c['sequence_length'] == 16384 and c['context_parallel_size'] == 1 and c['expert_model_parallel_size'] == 8
    assert c['epochs'] == 1 and not c['allow_repetition']
    assert c['calibration_receipt'] is None and c['estimated_iteration_seconds'] is None
    assert c['excluded_datasets'] == [LONG]
    cal = yaml.safe_load(Path('results/pretraining/nemochat_baby/nemotron_sft/seq16k_protocol/calibration.yaml').read_text())
    assert cal['max_updates'] >= 15 and cal['max_hours'] == .75
    assert cal['validation_max_dialogues'] == 128 and cal['require_final_validation']
    assert cal['sequence_length'] == 16384 and cal['execution_mode'] == 'calibration'


def test_16k_pilot_is_native_wall_bounded_and_main_needs_measured_rate():
    c = yaml.safe_load(Path('results/pretraining/nemochat_baby/nemotron_sft/seq16k_protocol/pilot.yaml').read_text())
    c['max_hours'] = .5
    m = dict(status='passed', schema='nemochat-nemotron-sft-v1',
        format='official-super35-selected-assistant-v1', sequence_length=16384,
        train_dialogues=32, heldout_dialogues=3, train_heldout_overlap=0,
        kmmlu_exact_question_matches=0, kmmlu_reserved_questions=34871,
        training_repetition=False, excluded_datasets=[LONG])
    assert N['stage_plan'](c,m,pilot=True)['max_hours'] == .5
    c = yaml.safe_load(Path('configs/posttraining/nemochat_baby_nemotron_sft_16k_48h.yaml').read_text())
    m.update(train_dialogues=1000000, heldout_dialogues=5000,
        prior_heldout_reservation_file=c['prior_heldout_reservation'], prior_reserved_prompt_groups=14)
    with pytest.raises(ValueError, match='calibration'):
        N['stage_plan'](c,m)


def test_cuda_memory_receipt_uses_actual_allocator_values():
    from types import SimpleNamespace
    cuda = SimpleNamespace(memory_allocated=lambda:11, memory_reserved=lambda:22,
        max_memory_allocated=lambda:33, max_memory_reserved=lambda:44)
    assert N['cuda_memory_receipt'](cuda) == dict(cuda_allocated_bytes=11,
        cuda_reserved_bytes=22,cuda_peak_allocated_bytes=33,cuda_peak_reserved_bytes=44)


def test_16k_prepared_shape_and_physical_exclusion_gate(tmp_path):
    m = dict(sequence_length=16384,train_dialogues=2,heldout_dialogues=1,
        excluded_datasets=[LONG])
    for split,n in [('train',2),('heldout',1)]:
        np.save(tmp_path/f'{split}-tokens.npy', np.ones((n,16385),dtype=np.uint32))
        np.save(tmp_path/f'{split}-masks.npy', np.ones((n,16384),dtype=np.uint8))
        (tmp_path/f'{split}-dialogues.jsonl').write_text(''.join(json.dumps(dict(source_key='Nemotron-SFT-BBQ-v1--train'))+'\n' for _ in range(n)))
    receipt = N['verify_prepared_shapes'](tmp_path,m)
    assert receipt['train_loss_tokens_beyond_4096'] == 2*(16384-4096)
    (tmp_path/'heldout-dialogues.jsonl').write_text(json.dumps(dict(source_key='Nemotron-SFT-Long-Context-Transduction-v1--train'))+'\n')
    with pytest.raises(ValueError, match='excluded'):
        N['verify_prepared_shapes'](tmp_path,m)


def test_16k_calibration_cannot_accept_old_runtime_receipt(tmp_path):
    import hashlib
    p = tmp_path/'manifest.json'
    p.write_text('{}')
    v = tmp_path/'verification.json'
    v.write_text(json.dumps(dict(status='passed',updates=15,ranks=[dict(rank=r) for r in range(8)],final_validation_accepted=True)))
    r = tmp_path/'calibration.json'
    receipt = dict(status='passed',global_batch_size=128,manifest_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
        warm_updates=10,source_checkpoint=N['SOURCE'],ranks=list(range(8)),mean_iteration_seconds=100,
        verification=str(v),sequence_length=4096,context_parallel_size=1,expert_model_parallel_size=8)
    r.write_text(json.dumps(receipt))
    plan = dict(sequence_length=16384,context_parallel_size=1,expert_model_parallel_size=8,
        source_checkpoint=N['SOURCE'],calibration_receipt=str(r),updates=100,max_hours=1,assumed_overhead_fraction=.1)
    with pytest.raises(ValueError, match='sequence|topology'):
        L['validate_calibration'](plan,p)
    receipt['sequence_length'] = 16384
    r.write_text(json.dumps(receipt))
    assert L['validate_calibration'](plan,p)['warm_updates'] == 10
