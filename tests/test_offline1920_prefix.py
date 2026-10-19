"""Prefix execution must never shorten the full dataset or scheduler identity."""
import importlib.util
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]

def load():
    path=ROOT/'midtraining/offline2b/prefix/contract.py'
    assert path.exists(), 'prefix contract implementation missing'
    spec=importlib.util.spec_from_file_location('prefix_contract',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def test_segment_keeps_full_budget_and_cache_identity():
    import json,hashlib
    c=json.loads((ROOT/'results/midtraining/offline2b-production1920-01/run-config.json').read_text())
    before=json.dumps(c,sort_keys=True)
    m=load();p=m.segment_plan(c,1143)
    assert p['sampler_budget']==[468736,120,8]
    assert p['schedule_horizon']==3662 and p['stop_step']==1143
    assert p['cache_id']==hashlib.sha256(before.encode()).hexdigest()
    assert json.dumps(c,sort_keys=True)==before
    class Context:tokenizer='native'
    class Dataset:
        def finalize(self):self.finalized=True
    ds=Dataset();calls=[]
    result=m.build_full_datasets(c,Context(),lambda budget,dataset: calls.append(budget) or ('train','valid',None),ds)
    assert calls==[[468736,120,8]] and ds.tokenizer=='native' and ds.finalized
    assert result==('train','valid',None)

def test_closed_tar_ownership_rejects_wrong_iteration(tmp_path):
    import io,tarfile,json
    c={'seed':43,'train_samples':468736}
    path=tmp_path/'cp0_dp2__3.tar'
    with tarfile.open(path,'w') as tar:
        data=json.dumps({'identifiers':{'seed':43,'sequence_length':4096,'train_samples':468736},'saver':{'k':32,'p':None,'min_k':1,'save_dtype':'fp16'}}).encode()
        info=tarfile.TarInfo('_meta.json');info.size=len(data);tar.addfile(info,io.BytesIO(data))
        data=b'x';info=tarfile.TarInfo('4.pt.zst');info.size=1;tar.addfile(info,io.BytesIO(data))
    with pytest.raises(ValueError,match='ownership'):load().authenticate_shard(path,c,3)

def test_chart_requires_all_eight_rank_updates(tmp_path):
    import json
    path=ROOT/'midtraining/offline2b/prefix/charts/render.py'
    assert path.exists(), 'real-metric chart renderer missing'
    spec=importlib.util.spec_from_file_location('prefix_chart',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    row={'update':1,'metrics':{'lm loss':2.,'logits distillation loss':1.,'total loss':1.5},'grad_norm':1.,'lr_used':0.,'skipped':False,'allrank_masked_tokens':524288,'seconds':1.}
    for rank in range(7):(tmp_path/f'student-rank{rank}-updates.jsonl').write_text(json.dumps(dict(row,rank=rank))+'\n')
    assert module.collect(tmp_path)==[]
    (tmp_path/'student-rank7-updates.jsonl').write_text(json.dumps(dict(row,rank=7))+'\n')
    result=module.collect(tmp_path)
    assert len(result)==1 and result[0]['update']==1 and result[0]['lr_used']==0
    assert result[0]['total_loss']==1.5 and result[0]['positions']==524288
    summary=module.summarize_ranks(tmp_path,1)
    assert len(summary)==8
    assert all(r['calls']==1 and r['zero_lr_calls']==1 and r['positive_lr_updates']==0 and r['skipped']==0 for r in summary)


def test_continuation_preserves_cursor_and_validation_capacity():
    import json
    path=ROOT/'midtraining/offline2b/prefix/continuation/contract.py'
    assert path.exists(), 'continuation contract missing'
    spec=importlib.util.spec_from_file_location('continuation_contract',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    c=json.loads((ROOT/'results/midtraining/offline2b-production1920-01/run-config.json').read_text())
    p=m.continuation_plan(c,64,1143)
    assert p['resume_step']==64 and p['stop_step']==1143
    assert p['validation_interval']==250
    assert p['remaining_validation_samples']<=120
    assert p['sampler_budget']==[468736,120,8]
    with pytest.raises(ValueError):m.continuation_plan(c,1143,1143)


def test_resume_compiler_cache_never_copies_absolute_old_paths(tmp_path):
    path=ROOT/'midtraining/offline2b/prefix/continuation02/contract.py'
    assert path.exists(), 'fresh-cache recovery contract missing'
    spec=importlib.util.spec_from_file_location('recovery_contract',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    old=tmp_path/'old';old.mkdir();(old/'kernel.py').write_text('cache_file="/old/absolute/kernel.py"')
    target=tmp_path/'new'/'inductor'
    assert m.create_local_compiler_cache(target)==target
    assert target.is_dir() and list(target.iterdir())==[]
    assert (old/'kernel.py').read_text()=='cache_file="/old/absolute/kernel.py"'


def test_invalid_stop():
    m=load();c={'train_samples':468736,'train_iters':3662,'warmup_iters':37,'decay_iters':366}
    for invalid in (0,1144,3662):
        with pytest.raises(ValueError):m.segment_plan(c,invalid)
