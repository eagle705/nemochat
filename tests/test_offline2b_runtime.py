"""CPU controller contracts, not GPU acceptance."""
import importlib.util
from pathlib import Path
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]
PATCH=ROOT/'midtraining/patches/offline_kd_pr'
sys.path.insert(0,str(PATCH))


def module(name):
    spec=importlib.util.spec_from_file_location('offline2b_'+name,ROOT/'midtraining/offline2b'/f'{name}.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def test_plan_fixed_gate_budget_and_bounded_production():
    m=module('launch')
    gate=m.plan('gate')
    assert gate['train_iters']==4 and gate['resume_step']==2
    assert gate['train_samples']==512 and gate['num_microbatches']==16
    assert gate['compile_threads']==1
    prod=m.plan('production')
    assert prod['train_iters']==3815 and prod['positions']==2000158720
    assert prod['warmup_iters']==38 and prod['decay_iters']==382
    with pytest.raises(ValueError):m.plan('online')


def test_dp8_resume_binding_survives_json_native_blend_tuples():
    import json
    from bridge_cached_kd.runtime import CoreRuntime
    from bridge_cached_kd.integration import CachedKDConfig,CachedKDForwardStep
    r=CoreRuntime(43,4096,512,128,1,None,[(['train'],[1.]),(['valid'],[1.]),None],lambda:2,lambda:16,
                  data_parallel_size=8,expert_parallel_size=8,dataset_identity='corpus-sha')
    a=CachedKDForwardStep(CachedKDConfig('read','/cache','id'),lambda s:r)
    binding=json.loads(json.dumps(a._binding(r,2)))
    CachedKDForwardStep(a.config,lambda s:r,resume_binding=binding).validate_resume(r)


def test_production_does_not_replay_completed_budget_for_reload():
    m=module('supervisor')
    assert m.stages('gate')==('teacher','student','resume')
    assert m.stages('production')==('teacher','student')


def test_supervisor_refuses_partial_allrank_or_failed_engine(tmp_path):
    m=module('supervisor')
    with pytest.raises(ValueError,match='eight'):m.accept_rank_rows([{'rank':0}],4)
    rows=[dict(rank=i,completed=4,positive_updates=4,skipped=0,nonfinite=0) for i in range(8)]
    assert m.accept_rank_rows(rows,4)['all_rank_completed']==4
    rows[7]['positive_updates']=0
    with pytest.raises(ValueError,match='positive'):m.accept_rank_rows(rows,4)
    state=tmp_path/'state.json'
    m.claim(state,'unit-1')
    with pytest.raises(FileExistsError):m.claim(state,'unit-1')


def test_source_freeze_detects_mutation_without_model_payloads(tmp_path):
    m=module('launch');m.REPO=tmp_path
    source=tmp_path/'native.py';source.write_text('original source\n')
    m.source_files=lambda:[source]
    frozen=m.freeze();m.verify_freeze(frozen)
    source.write_text('modified source\n')
    with pytest.raises(ValueError,match='freeze'):m.verify_freeze(frozen)


def test_foreign_cleanup_is_refused():
    m=module('supervisor')
    class Engine:
        def request(self,method,path):
            return {'Id':'exact-id','Config':{'Labels':{m.LABEL:'foreign-owner'}}}
    with pytest.raises(ValueError,match='foreign'):m.owned(Engine(),'exact-id','owned-task')


def test_production_start_requires_parent_review_before_any_side_effect(tmp_path):
    import subprocess
    output=ROOT/'results/midtraining/offline2b-never-created-test'
    assert not output.exists()
    result=subprocess.run([sys.executable,str(ROOT/'midtraining/offline2b/launch.py'),
        '--mode','production','--output',str(output),'--start'],capture_output=True,text=True)
    assert result.returncode!=0 and 'parent production review' in result.stderr
    assert not output.exists()


def test_dp8_cursor_binding_freezes_full_sampler_identity():
    from bridge_cached_kd.runtime import CoreRuntime
    from bridge_cached_kd.integration import CachedKDConfig,CachedKDForwardStep
    r=CoreRuntime(43,4096,512,128,1,None,None,lambda:2,lambda:16,
                  data_parallel_size=8,expert_parallel_size=8,dataset_identity='manifest-sha')
    a=CachedKDForwardStep(CachedKDConfig('read','/cache','cache-sha'),lambda s:r)
    a._runtime=r;a._next_iteration=2
    a._check_cursor(type('State',(),{'train_state':type('Train',(),{'step':2})()})())
    binding=a.checkpoint_binding()
    assert binding['data_parallel_size']==8 and binding['dataset_identity']=='manifest-sha'
    assert binding['consumed_samples']==256 and binding['train_samples']==512
    b=CachedKDForwardStep(a.config,lambda s:r,resume_binding=binding)
    b.validate_resume(r)


def test_dp8_requires_exact_topology_and_native_microbatches():
    from bridge_cached_kd.runtime import validate_topology
    assert validate_topology(tp=1,cp=1,dp=8,pp=1,ep=8,gbs=128,mbs=1,microbatches=16)==8
    assert validate_topology(tp=1,cp=1,dp=1,pp=1,ep=1,gbs=2,mbs=1,microbatches=2)==1
    for changed in [dict(tp=2),dict(cp=2),dict(dp=4),dict(pp=2),dict(ep=1),dict(microbatches=128)]:
        with pytest.raises((ValueError,NotImplementedError)):
            validate_topology(**(dict(tp=1,cp=1,dp=8,pp=1,ep=8,gbs=128,mbs=1,microbatches=16)|changed))
