"""CPU-only continuation contracts; no accelerator initialization."""
import importlib
import json
import hashlib
from pathlib import Path
import pytest

MODULE='midtraining.training.offline_kd.continue2000.contract'

def contract():
    return importlib.import_module(MODULE)

def test_budget_preserves_original_sampler_and_refuses_capacity():
    c=contract()
    p=c.plan()
    assert (p['target_calls'],p['restored_calls'],p['teacher_start'],p['teacher_stop'])==(2000,16,1143,2064)
    assert p['consumed_teacher_start']==146304
    assert p['sampler_budget']==[468736,120,8]
    assert p['schedule']=={'horizon':3662,'warmup':37,'decay':366}
    assert p['new_source_samples']==921*128
    assert p['new_source_positions']==921*128*4096
    assert c.remaining_capacity(16,2064,range(16),{20,86,98,121,122,126})==2042
    with pytest.raises(ValueError,match='capacity'):
        c.require_capacity(2000,16,1143,range(16),set())
    with pytest.raises(ValueError,match='original source'):
        c.plan(teacher_stop=3663)

@pytest.fixture
def continuation_inputs(tmp_path, monkeypatch):
    """Handwritten CPU contract input, NOT an accepted native checkpoint receipt.

    Historical production fixtures used to leak in through a hardcoded checkout
    root. Build the same binding schema through the real producer on tiny metadata.
    No cache, model state, runtime acceptance or training success is fabricated.
    """
    from types import SimpleNamespace
    from midtraining.patches.offline_kd_pr.bridge_cached_kd.integration import (
        CachedKDConfig, CachedKDForwardStep)
    c = contract()
    old = tmp_path / 'handwritten-contract-input'
    old.mkdir()
    manifest = {'cache_id': c.CACHE_ID, 'fixture_scope': 'CPU unit input only'}
    (old / 'prefix-manifest.json').write_text(json.dumps(manifest))
    state = dict(optimizer_calls=16, committed_mapping=list(range(16)),
                 next_unconsumed_physical=16, prefetched_unconsumed=[16, 17],
                 scanned_cursor=18, quarantined_physical=[], exclusions=[],
                 physical_limit=1143)
    state['mapping_sha256'] = c.digest(state)
    runtime = SimpleNamespace(seed=43, seq_length=4096, train_samples=468736,
                              global_batch_size=128, micro_batch_size=1,
                              data_parallel_size=8, expert_parallel_size=8,
                              dataset_identity='handwritten-CPU-only',
                              blend=None, blend_per_split=None)
    adapter = CachedKDForwardStep(CachedKDConfig('read', '/unused', c.CACHE_ID),
                                  lambda unused: runtime)
    binding = adapter._binding(runtime, 16)
    binding.update(dynamic_state=state,
                   dynamic_manifest_sha256=c.sha(old / 'prefix-manifest.json'))
    (old / 'checkpoint-step16-transaction.json').write_text(json.dumps({'binding': binding}))
    monkeypatch.setattr(c, 'OLD', old)
    return c


def test_extension_checks_committed_prefix_and_ancestry(continuation_inputs):
    c=continuation_inputs
    binding=json.loads((c.OLD/'checkpoint-step16-transaction.json').read_text())['binding']
    manifest=json.loads((c.OLD/'prefix-manifest.json').read_text())
    state=c.extend_state(binding,manifest,c.sha(c.OLD/'prefix-manifest.json'),2064)
    assert binding['dynamic_state']['physical_limit']==1143
    assert state['physical_limit']==2064
    assert state['committed_mapping']==list(range(16))
    assert state['next_unconsumed_physical']==16
    assert state['prefetched_unconsumed']==[16,17]
    for field,value in [('cache_id','bad'),('consumed_samples',2176),('next_iteration',17),('train_samples',256000)]:
        bad=dict(binding);bad[field]=value
        with pytest.raises(ValueError):c.extend_state(bad,manifest,c.sha(c.OLD/'prefix-manifest.json'),2064)
    with pytest.raises(ValueError):c.extend_state(binding,manifest,'0'*64,2064)

def test_extended_broker_rereads_prefetch_and_never_cycles(tmp_path, continuation_inputs):
    c=continuation_inputs;bmod=importlib.import_module('midtraining.training.offline_kd.continue2000.broker')
    binding=json.loads((c.OLD/'checkpoint-step16-transaction.json').read_text())['binding']
    manifest=json.loads((c.OLD/'prefix-manifest.json').read_text())
    state=c.extend_state(binding,manifest,c.sha(c.OLD/'prefix-manifest.json'),2064)
    reads=[]
    def reader(p):
        reads.append(p)
        if p==18:raise bmod.CacheRejected({'error':'test bad data'})
        return p
    def gather(row):return [dict(row,rank=r) for r in range(8)]
    b=bmod.Broker(0,reader,gather,tmp_path,limit=2064)
    b.restore(state)
    assert reads==[16,17]
    assert b.begin(16)==16 and b.take()==16
    b.complete()
    assert reads==[16,17,18,19]
    assert b.calls==17 and b.snapshot()['committed_mapping']==list(range(17))
    assert b.snapshot()['prefetched_unconsumed']==[17,19]
    assert b.exclusions[-1]['no_optimizer_or_lr_advance']
    assert b.begin(17)==17 and b.take()==17
    b.complete()
    assert b.begin(18)==19
    assert len(reads)==len(set(reads))

def test_new_readiness_requires_allrank_exact_window_and_native_identity():
    c=contract();p=c.plan(teacher_stop=1145)
    rows=[{'rank':r,'iteration':i,'path':f'/new/cp0_dp{r}__{i}.tar'} for r in range(8) for i in (1143,1144)]
    ranks=[dict(rank=r,scored=[1143,1144],quarantined=[],optimizer_calls=0,frozen=True,cache_id=c.CACHE_ID) for r in range(8)]
    assert c.check_new_coverage(p,rows,ranks)==[1143,1144]
    with pytest.raises(ValueError):c.check_new_coverage(p,rows[:-1],ranks)
    with pytest.raises(ValueError):c.check_new_coverage(p,rows+[rows[0]],ranks)
    ranks[7]=dict(ranks[7],cache_id='bad')
    with pytest.raises(ValueError):c.check_new_coverage(p,rows,ranks)

def test_release_requires_parent_diag_and_exact_clean_engine():
    c=contract()
    with pytest.raises(ValueError,match='diagnosis'):c.validate_diag(None,'pins')
    diag=dict(status='parent_reviewed',source_pins_sha256='pins',input18_disposition='quarantine',
              known_global_input_exclusions=[18],diagnosis_evidence_sha256='a'*64,
              unexpected_input_digest_mismatch='fatal')
    assert c.validate_diag(diag,'pins')==[18]
    for field,value in [('status','pending'),('source_pins_sha256','stale'),('unexpected_input_digest_mismatch','ignore')]:
        bad=dict(diag);bad[field]=value
        with pytest.raises(ValueError):c.validate_diag(bad,'pins')
    with pytest.raises(ValueError):c.clean_engine({'id':'a','state':{'Running':True,'ExitCode':0,'OOMKilled':False}},'a')
    with pytest.raises(ValueError):c.clean_engine({'id':'b','state':{'Running':False,'ExitCode':0,'OOMKilled':False}},'a')
    assert c.clean_engine({'id':'a','state':{'Running':False,'ExitCode':0,'OOMKilled':False}},'a')

def test_inputgate_schema_fatal_digest_quarantinable():
    m=importlib.import_module('midtraining.training.offline_kd.continue2000.input_gate')
    import torch
    sample=dict(tokens=torch.arange(4096,dtype=torch.int64),labels=torch.arange(1,4097,dtype=torch.int64),
                loss_mask=torch.ones(4096),position_ids=torch.arange(4096,dtype=torch.int64))
    m.validate_schema(sample)
    with pytest.raises(m.SourceDigestMismatch) as e:m.compare_digest(sample,'0'*64,2332)
    assert e.value.observation['observed_sha256']==m.sample_digest(sample)
    assert e.value.observation['source_index']==2332
    bad=dict(sample,tokens=torch.zeros(4095,dtype=torch.int64))
    with pytest.raises(ValueError):m.validate_schema(bad)
    assert not isinstance(ValueError('schema'),m.SourceDigestMismatch)

def test_adapter_full_resume_native_binding_is_not_refreshed(tmp_path, continuation_inputs):
    from types import SimpleNamespace
    c=continuation_inputs;m=importlib.import_module('midtraining.training.offline_kd.continue2000.binding')
    from importlib import util
    spec=util.spec_from_file_location('accepted_integration',c.REPO/'midtraining/patches/offline_kd_pr/bridge_cached_kd/integration.py')
    sdk=util.module_from_spec(spec);__import__('sys').modules[spec.name]=sdk;spec.loader.exec_module(sdk)
    binding=json.loads((c.OLD/'checkpoint-step16-transaction.json').read_text())['binding']
    runtime=SimpleNamespace(**{k:binding[k] for k in ('seed','seq_length','train_samples','global_batch_size','micro_batch_size','data_parallel_size','expert_parallel_size','dataset_identity','blend','blend_per_split')},iteration=lambda:16)
    broker=SimpleNamespace(snapshot=lambda:{'new':'state'})
    cls=m.adapter_class(sdk.CachedKDForwardStep,binding,broker,c.sha(c.OLD/'prefix-manifest.json'),'newmanifest')
    adapter=cls(sdk.CachedKDConfig('read','/cache',c.CACHE_ID),lambda s:runtime,resume_binding=binding)
    adapter.validate_resume(runtime)
    assert adapter.resume_binding==binding
    future=adapter._binding(runtime,17)
    assert future['dynamic_manifest_sha256']=='newmanifest'
    assert future['ancestry_dynamic_manifest_sha256']==binding['dynamic_manifest_sha256']
    runtime.train_samples=256000
    with pytest.raises(ValueError):adapter.validate_resume(runtime)

def test_real_preview_is_inert_and_execute_refuses_without_review(tmp_path):
    import subprocess,sys
    c=contract();script=c.REPO/'midtraining/training/offline_kd/continue2000/start.py'
    result=subprocess.run([sys.executable,str(script)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    p=json.loads(result.stdout)
    assert p['mode']=='INERT_PREVIEW' and p['gpu_launched'] is False
    assert '--user' in p['systemd_argv'] and '--property=RuntimeMaxSec=86400' in p['systemd_argv']
    denied=subprocess.run([sys.executable,str(script),'--execute'],capture_output=True,text=True)
    assert denied.returncode!=0 and 'parent source review' in denied.stderr

def test_native_resume_source_keeps_optimizer_rng_full_sampler_and_target():
    import ast
    c=contract();source=(c.REPO/'midtraining/training/offline_kd/continue2000/native.py').read_text()
    ast.parse(source)
    assert 'STOP=2000' in source
    assert 'load_optim=resume,load_rng=resume,finetune=False' in source
    assert "lr_decay_iters=C['train_iters']" in source
    assert "lr_warmup_iters=C['warmup_iters']" in source
    assert 'BROKER.restore(extended_state)' in source
    assert "start=16" in source and "consumed_train_samples)==start*128" in source
    assert 'load=str(OLD/\'checkpoint\')' in source
    assert 'most_recent_k=3' in source
    assert 'VerifiedSource(train,expected)' in source
    assert 'load_weights=not resume' in source
    assert 'full_teacher_ready=True' not in source
    assert "math.isclose(float(ctx.optimizer.param_groups[0]['lr']),1e-5*16/37" in source

def test_teacher_uses_real_consumed_cursor_and_preforward_allrank_gate():
    import ast
    c=contract();source=(c.REPO/'midtraining/training/offline_kd/continue2000/teacher.py').read_text();ast.parse(source)
    assert "consumed_samples=p['consumed_teacher_start']" in source
    assert "range(p['teacher_start'],p['teacher_stop'])" in source
    assert "original.dataset_budget()==[468736,120,8]" in source
    assert source.index('ranks=gather(local)')<source.index('with torch.no_grad()')
    assert 'verified_sample(train,index,given=given)' in source
    assert 'LogitsSaverHooks' in source and 'attach_hooks(unwrap_model(model))' in source
    assert 'optimizer.step' not in source and '.backward(' not in source

def test_new_cache_auditor_requires_emitted_payload_and_native_metadata():
    m=importlib.import_module('midtraining.training.offline_kd.continue2000.new_cache')
    pytest.importorskip('zstandard',reason='real codec runs in pinned CPU image; host has no zstandard')
    c=contract();full=json.loads((c.OLD/'prefix-manifest.json').read_text());full=json.loads(Path(full['full_expectations_path']).read_text())
    from midtraining.cache.directio.cpu_gate import native_module
    from midtraining.cache.directio.binding import install
    from midtraining.training.offline_kd.prefix.dynamic.restart01.reader import ImmutableOpener
    native=native_module(full);opener=ImmutableOpener();install(native,full['native_contract'],opener)
    row=next(x for x in full['shards'] if x['rank']==0 and x['iteration']==0)
    with pytest.raises(ValueError,match='emitted payload'):
        m.audit_closed(Path(row['path']),0,0,full,native,opener,'0'*64)

def test_sample_log_merge_preserves_old_prefix_and_ignores_partial_tail():
    m=importlib.import_module('midtraining.training.offline_kd.continue2000.merge')
    def row(p,micro):return dict(micro=p*16+micro,iteration=p,rank=0,sha256='a'*64,masked_tokens=4096)
    old=[row(i,j) for i in range(1143) for j in range(16)]+[row(1143,0)]
    new=[row(1143,j) for j in range(16)]
    merged=m.sample_rows(old,new,0,[1143])
    assert len(merged)==1144*16 and merged[:16]==old[:16]
    with pytest.raises(ValueError):m.sample_rows(old,new[:-1],0,[1143])
    with pytest.raises(ValueError):m.sample_rows(old+[old[0]],new,0,[1143])

def test_parent_review_rejects_incomplete_source_allowlist(tmp_path,monkeypatch):
    start=importlib.import_module('midtraining.training.offline_kd.continue2000.start');c=contract()
    monkeypatch.setattr(start,'OUT',tmp_path)
    (tmp_path/'preparation').mkdir()
    pins=tmp_path/'preparation/source-pins.json';pins.write_text(json.dumps(dict(source_sha256={},snapshot_sha256={},ancestry_sha256={})))
    review=tmp_path/'review.json';review.write_text(json.dumps(dict(status='parent_approved',source_pins_sha256=c.sha(pins))))
    with pytest.raises(ValueError,match='allowlist'):start.verify_review(review,c.sha(review))

def test_runtime_policy_uses_bound_receipt_and_evidence(tmp_path):
    """Handwritten guard fixture, never a parent approval or production receipt."""
    m=importlib.import_module('midtraining.training.offline_kd.continue2000.merge');c=contract()
    evidence=tmp_path/'input-digest-investigation-01';evidence.mkdir()
    data=evidence/'handwritten-cpu-input.txt';data.write_text('CPU fixture only; not diagnosis evidence')
    p=tmp_path/'handwritten-policy.json'
    p.write_text(json.dumps(dict(known_global_quarantine_addition=[18],confirmed_source_index=2332,
        source_index_formula='physical*128+micro*8+rank',post_acceptance_input_mutation_still_fatal=True,
        model_optimizer_collective_logic_nonfinite_faults_still_fatal=True,startup_source_review_still_required=True,
        evidence_sha256={data.name:c.sha(data)})))
    assert m.accepted_policy(dict(diagnosis_policy_path=str(p),diagnosis_policy_sha256=c.sha(p)))['confirmed_source_index']==2332
    with pytest.raises(ValueError):m.accepted_policy(dict(diagnosis_policy_path=str(p),diagnosis_policy_sha256='0'*64))
    data.write_text('tampered fixture')
    with pytest.raises(ValueError,match='evidence changed'):
        m.accepted_policy(dict(diagnosis_policy_path=str(p),diagnosis_policy_sha256=c.sha(p)))
