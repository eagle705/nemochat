"""CPU contract controls only; no native/GPU execution evidence."""
import importlib.util
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]

def module(name='longrun_native'):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'midtraining' / (name + '.py'))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_horizon_uses_eight_warm_unprofiled_full_batch_updates():
    m = module()
    rows = [dict(update=i, seconds=100 if i <= 8 else 20, peak_memory_bytes=123,
                 global_batch_size=128, sequence_length=4096, profiled=False,
                 lr_used=0 if i == 1 else 1e-5, positive_gradient=True) for i in range(1, 17)]
    receipt = m.calibration_summary(rows)
    assert receipt['warm_updates'] == list(range(9, 17))
    assert receipt['mean_step_seconds'] == 20
    assert receipt['zero_lr_optimizer_calls'] == 1
    plan = m.derive_horizon(receipt)
    assert plan['train_iters'] == 3888
    assert plan['lr_warmup_iters'] == 39
    assert plan['lr_wsd_decay_iters'] == 389
    with pytest.raises(ValueError, match='16'):
        m.calibration_summary(rows[:4])
    rows[10]['profiled'] = True
    with pytest.raises(ValueError, match='unprofiled'):
        m.calibration_summary(rows)


def prepared(root, tokens=10_000_000):
    sources = {}
    for key, weight in zip(module().KEYS, module().WEIGHTS):
        sources[key] = {}
        for split in ('train', 'validation'):
            prefix = root / key / split
            prefix.parent.mkdir(parents=True, exist_ok=True)
            Path(str(prefix)+'.bin').write_bytes(b'fixture')
            Path(str(prefix)+'.idx').write_bytes(b'fixture')
            sources[key][split] = dict(prefix=str(prefix), tokens=int(tokens*weight) if split == 'train' else 50000,
                                       documents=4, reader_verified=True)
    return dict(phase='prepared_verified', result=dict(sources=sources, cycling=False,
                preparation_only=True, production_training_budget=None,
                full_bounded_shards_scanned=True, audit=dict(train_heldout_token_document_overlap=0,
                training_duplicate_token_documents=0)))


def test_actual_preparation_schema_capacity_is_per_source_and_shifted(tmp_path):
    m = module()
    manifest = m.adapt_prepared_manifest(prepared(tmp_path))
    receipt = m.capacity_gate(manifest, 16, 128, 4096)
    assert receipt['no_training_repeat'] is True
    assert set(receipt['sources']) == set(m.KEYS)
    assert receipt['sources']['code_concepts']['required_tokens'] > 16*128*4096*.6
    with pytest.raises(ValueError, match='capacity'):
        m.capacity_gate(manifest, 1000, 128, 4096)
    manifest['sources']['formal_logic']['train']['tokens'] = 1
    with pytest.raises(ValueError, match='formal_logic'):
        m.capacity_gate(manifest, 16, 128, 4096)


def test_prepared_manifest_rejects_wrong_sources_leakage_and_live_outputs(tmp_path):
    m = module()
    data = prepared(tmp_path)
    data['phase'] = 'tokenizing'
    with pytest.raises(ValueError, match='prepared_verified'):
        m.adapt_prepared_manifest(data)
    data['phase'] = 'prepared_verified'
    data['result']['audit']['train_heldout_token_document_overlap'] = 1
    with pytest.raises(ValueError, match='overlap'):
        m.adapt_prepared_manifest(data)


def test_no_day_completion_claim_from_iteration_budget_or_missing_reload():
    m = module()
    assert m.completion_status('production', 100, 100, 5000, True, True) == 'iteration_budget_complete_short_of_wall_target'
    assert m.completion_status('production', 99, 100, 86401, True, True) == 'duration_complete_reload_pending'
    with pytest.raises(ValueError, match='final'):
        m.completion_status('production', 99, 100, 86401, False, True)
    assert m.completion_status('duration_control', 1, 100, 62, True, True) == 'duration_control_complete_reload_pending'


def test_native_index_guard_rejects_repeated_epochs():
    from types import SimpleNamespace
    m = module()
    one = SimpleNamespace(indices=[0, 1], document_index=[1, 0])
    assert m.no_repeat_gate(one)['native_train_epochs'] == 1
    with pytest.raises(ValueError, match='repeat'):
        m.no_repeat_gate(SimpleNamespace(indices=[0, 1], document_index=[1, 0, 1, 0]))


def test_frozen_configs_original_base_and_native_schedule(tmp_path):
    m = module()
    c = m.load_config(ROOT/'midtraining/calibration.yaml')
    assert c['train']['train_iters'] == 16
    assert c['train']['global_batch_size'] == 128
    assert c['student_hf'] == 'checkpoints/exports/nemochat_baby_base_hf_bf16_02'
    assert c['optimizer']['clip_grad'] == 1
    p = m.load_config(ROOT/'midtraining/longrun24h.yaml')
    assert p['train']['train_iters'] is None
    with pytest.raises(ValueError, match='calibration'):
        m.freeze_config(p, None)
    stats = dict(warm_updates=list(range(9,17)), profiled=False, global_batch_size=128,
                 sequence_length=4096, mean_step_seconds=20)
    frozen = m.freeze_config(p, stats)
    assert frozen['scheduler']['lr_decay_iters'] == frozen['train']['train_iters'] == 3888
    assert frozen['train']['exit_duration_in_mins'] == 1440
    frozen['tracking']['run_id']='test-native-run'
    m.validate_config(frozen)
    bad = dict(c, student_hf='checkpoints/midtraining/super35-kd-pilot-01')
    with pytest.raises(ValueError, match='original BASE'):
        m.validate_config(bad)
    frozen['scheduler']['lr_decay_iters'] -= 1
    with pytest.raises(ValueError, match='horizon'):
        m.validate_config(frozen)


def test_duration_force_only_on_early_exit_not_normal_final_budget():
    m = module()
    assert m.force_duration_evaluation(60.1, 1, 1, 16)
    assert not m.force_duration_evaluation(60.1, 1, 16, 16)
    assert not m.force_duration_evaluation(1, 60, 1, 16)


def test_native_tokens_require_fresh_zero_alias_audit_not_old_climbmix_proof():
    m = module()
    with pytest.raises(ValueError, match='tokenizer'):
        m.corpus_tokenizer_gate({})
    receipt = dict(repository=m.TEACHER_REPO, revision=m.REVISION, vocab_size=131072,
                   eos_id=11, asset_sha256={'tokenizer.json':m.TOKENIZER_SHA256})
    with pytest.raises(ValueError, match='fresh'):
        m.corpus_tokenizer_gate({'tokenizer':receipt})
    audit = dict(counts={str(i):0 for i in range(18,30)}, actual_prepared_bin_scan=True)
    result = m.corpus_tokenizer_gate({'tokenizer':receipt}, audit)
    assert result['old_climbmix_absence_proof_reused'] is False
    audit['counts']['18'] = 1
    with pytest.raises(ValueError, match='reserved'):
        m.corpus_tokenizer_gate({'tokenizer':receipt}, audit)


def test_real_manifest_companion_digest_and_int32_scan(tmp_path):
    import hashlib
    import json
    import struct
    m = module()
    original = prepared(tmp_path)
    data = dict(schema_version=1, status='prepared_verified', sequence_length=4096,
                source_order=list(m.KEYS), weights=dict(zip(m.KEYS,m.WEIGHTS)),
                sources=original['result']['sources'], audit=original['result']['audit'], cycling=False,
                tokenizer=dict(repository=m.TEACHER_REPO, revision=m.REVISION, vocab_size=131072,
                    eos_id=11, asset_sha256={'tokenizer.json':m.TOKENIZER_SHA256}))
    for source in data['sources'].values():
        for item in source.values():
            raw = struct.pack('<4i',11,25,999,11)
            Path(item['prefix']+'.bin').write_bytes(raw)
            item.update(dtype='int32', tokens=4, bin_bytes=len(raw), idx_bytes=7,
                        bin_sha256=hashlib.sha256(raw).hexdigest(),
                        idx_sha256=hashlib.sha256(b'fixture').hexdigest(), truncated_documents=0)
    path=tmp_path/'manifest.json'
    path.write_text(json.dumps(data))
    (tmp_path/'success.json').write_text(json.dumps(dict(phase='prepared_verified',
                result=dict(manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))))
    manifest, digest = m.load_prepared_manifest(path)
    audit=m.audit_reserved_ids(manifest)
    assert audit['counts']['25'] == 8
    assert audit['actual_prepared_bin_scan'] is True
    with pytest.raises(ValueError, match='reserved'):
        m.corpus_tokenizer_gate(manifest,audit)
    path.write_text(path.read_text()+' ')
    with pytest.raises(ValueError, match='SHA256'):
        m.load_prepared_manifest(path)


def test_startup_heldout_reserves_extra_native_sampler_window():
    m=module()
    assert m.evaluation_budget_with_startup([2048,16,8],1,8)==[2048,24,8]
    assert m.evaluation_interval_for_step(16,1,16,0,180)==1
    assert m.evaluation_interval_for_step(16,16,16,99999,180)==16
    assert m.evaluation_interval_for_step(250,99,3888,86401,1440)==99
    assert m.evaluation_interval_for_step(250,99,3888,5000,1440)==250


def test_run_and_launcher_help_are_inert():
    import subprocess
    import sys
    for name in ('longrun_native.py', 'launch_longrun.py'):
        result = subprocess.run([sys.executable, str(ROOT/'midtraining'/name), '--help'],
                                capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        assert '--config' in result.stdout
    result = subprocess.run([sys.executable, str(ROOT/'midtraining/launch_longrun.py'), '--preview'],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert 'inert_preview' in result.stdout


def test_native_runtime_and_config_api_exist_without_importing_cuda():
    m = module()
    assert callable(m.make_native_config)
    assert callable(m.run_native)
    assert callable(m.install_native_dataset_audit)


def test_launcher_payload_bounded_secretless_explicit_owned_gpus(tmp_path):
    m = module('launch_longrun')
    native = module()
    c = native.load_config(ROOT/'midtraining/calibration.yaml')
    c['checkpoint_save'] = 'checkpoints/midtraining/test-fresh-01'
    payload = m.container_payload(c, 'test-fresh-01', 'train', 'results/test/config.yaml')
    assert payload['Image'].startswith('nvcr.io/nvidia/nemo@sha256:')
    assert payload['HostConfig']['Privileged'] is False
    assert payload['HostConfig']['RestartPolicy']['Name'] == 'no'
    assert payload['HostConfig']['DeviceRequests'][0]['DeviceIDs'] == [str(i) for i in range(8)]
    assert payload['HostConfig']['NetworkMode'] == 'none'
    assert not any('API_KEY' in item for item in payload['Env'])
    assert 'results/test/config.yaml' in payload['Cmd']
    assert '--execute' in payload['Cmd']
    c['tracking'] = dict(mode='online', entity='nvidia', project='nemochat', run_id='test-fresh-01')
    online = m.container_payload(c, 'test-fresh-01', 'train', 'results/test/config.yaml')
    assert online['HostConfig']['NetworkMode'] == 'bridge'
    assert not any('API_KEY' in item for item in online['Env'])
    assert any('/run/nemochat-wandb.env:ro' in x for x in online['HostConfig']['Binds'])


def test_launcher_rejects_running_engine_and_unexecuted_wandb(tmp_path):
    m = module('launch_longrun')
    with pytest.raises(ValueError, match='Engine'):
        m.accept_native(tmp_path, dict(Running=True, ExitCode=0, OOMKilled=False), 'production')
    with pytest.raises(ValueError, match='all-rank'):
        m.accept_native(tmp_path, dict(Running=False, ExitCode=0, OOMKilled=False), 'production')


def test_exact_cleanup_refuses_unowned_container():
    m = module('launch_longrun')
    with pytest.raises(ValueError, match='identity'):
        m.check_owned(dict(Id='wrong', Name='/foreign', Config=dict(Labels={})), 'expected', 'test-01', 'train')


def test_native_config_consumes_native_blend_and_full_state_save(monkeypatch,tmp_path):
    import sys
    from types import ModuleType,SimpleNamespace
    m=module()
    c=m.load_config(ROOT/'midtraining/calibration.yaml')
    c['checkpoint_save']='checkpoints/midtraining/fresh-test'
    class Config:
        def __init__(self,**kw): self.__dict__.update(kw)
    config=ModuleType('megatron.bridge.training.config')
    config.GPTDatasetConfig=config.CheckpointConfig=Config
    config.runtime_config_update=lambda cfg: None
    monkeypatch.setitem(sys.modules,'megatron.bridge.training.config',config)
    calls=[]
    def make(c,Dataset,out,**kw):
        calls.append(kw)
        return SimpleNamespace(dataset=Dataset(),checkpoint=None)
    monkeypatch.setattr(m,'helper',lambda *a:SimpleNamespace(make_config=make))
    manifest=prepared(tmp_path)['result']
    cfg,_=m.make_native_config(c,manifest,tmp_path,config_only=True)
    assert cfg.dataset.blend_per_split[0][1]==list(m.WEIGHTS)
    assert len(cfg.dataset.blend_per_split[0][0])==4
    assert cfg.dataset.seq_length==4096
    assert cfg.checkpoint.save_optim and cfg.checkpoint.save_rng
    assert not cfg.checkpoint.load_optim and not cfg.checkpoint.load_rng
    assert cfg.checkpoint.most_recent_k==3
    reload,_=m.make_native_config(c,manifest,tmp_path,reload_checkpoint=tmp_path/'ckpt')
    assert reload.dataset.blend_per_split[0] is None
    assert reload.checkpoint.save is None
    assert reload.checkpoint.finetune
    assert calls[-1]['reload_checkpoint']==tmp_path/'ckpt'


def test_native_registry_audit_delegates_and_restores(monkeypatch):
    import sys
    import numpy as np
    from types import ModuleType,SimpleNamespace
    m=module()
    class GPT: pass
    config=ModuleType('megatron.bridge.training.config')
    config.GPTDatasetConfig=GPT
    data=ModuleType('megatron.bridge.data')
    train=SimpleNamespace(datasets=[SimpleNamespace(indices=[0,1],document_index=[1,0]) for _ in range(4)],
                          dataset_index=np.array([0,1,2,3,0,0]))
    original=lambda samples,cfg:(train,None,None)
    data.utils=SimpleNamespace(_REGISTRY={GPT:original})
    monkeypatch.setitem(sys.modules,'megatron.bridge.training.config',config)
    monkeypatch.setitem(sys.modules,'megatron.bridge.data',data)
    receipts=[]
    restore=m.install_native_dataset_audit({},lambda *a:receipts.append(a))
    assert data.utils._REGISTRY[GPT]([6,0,0],None)[0] is train
    assert receipts[-1][1]['source_samples']['code_concepts']==3
    train.datasets[0].document_index=[0,1,0,1]
    with pytest.raises(ValueError,match='repeat'):
        data.utils._REGISTRY[GPT]([6,0,0],None)
    restore()
    assert data.utils._REGISTRY[GPT] is original


def test_all_rank_acceptance_requires_real_update_save_and_heldout(tmp_path):
    import json
    m=module('launch_longrun')
    def out(rank,name,data):
        (tmp_path/f'rank-{rank}-{name}.json').write_text(json.dumps(data))
    for rank in range(8):
        out(rank,'complete',dict(status='duration_control_complete_reload_pending',completed_updates=1,
            save_steps=[1],native_elapsed_seconds=70))
        out(rank,'heldout',dict(evaluations=[dict(update=1,lm_loss=1e20)]))
        out(rank,'preflight',dict(manifest_sha256='fixture'))
        out(rank,'save',dict(save_optim=True,save_rng=True,native_synchronous_save_completed=True))
        out(rank,'start',dict(teacher_gradients_absent=True))
        out(rank,'dataset-built',dict(native_train_epochs=1,native_blend=True,
             source_samples=dict(zip(module().KEYS,[10,2,2,4]))))
        row=dict(update=1,lr_used=0,metrics={'LM Loss':1e20,'KD Loss':1e30,'Total Loss':1.5e20,'Grad Norm':1e10},
                 native_optimizer=True,teacher_frozen_no_grads=True)
        (tmp_path/f'rank-{rank}-updates.jsonl').write_text(json.dumps(row)+'\n')
    result=m.accept_native(tmp_path,dict(Running=False,ExitCode=0,OOMKilled=False),'duration_control')
    assert result['completed_updates']==1
    assert result['training_24h_accepted'] is False
    for rank in range(8):
        out(rank,'complete',dict(status='duration_complete_reload_pending',completed_updates=1,
            save_steps=[1],native_elapsed_seconds=70))
    with pytest.raises(ValueError,match='24h'):
        m.accept_native(tmp_path,dict(Running=False,ExitCode=0,OOMKilled=False),'production')
    for rank in range(8):
        out(rank,'complete',dict(status='duration_control_complete_reload_pending',completed_updates=1,
            save_steps=[1],native_elapsed_seconds=70))
    out(7,'save',dict(save_optim=False,save_rng=False))
    with pytest.raises(ValueError,match='full-state'):
        m.accept_native(tmp_path,dict(Running=False,ExitCode=0,OOMKilled=False),'duration_control')
