"""Authenticate completed C call mapping and input oracles; no tensor payload audit."""
import hashlib
import importlib.util
import json
from pathlib import Path

REPO=Path(__file__).resolve().parents[5]
HERE=Path(__file__).resolve().parent
C=REPO/'results/midtraining/offline2b-prefixtrain-1000-01'
OUT=REPO/'results/midtraining/offline2b-lmonly-1000-02'
TEACHER=REPO/'results/midtraining/offline2b-production1920-01'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def put(name,value):
    with (OUT/name).open('x') as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\n')
def main():
    OUT.mkdir(parents=True,exist_ok=False)
    config=json.loads((C/'run-config.json').read_text())
    assert (config['train_iters'],config['train_samples'],config['seed'])==(3662,468736,43)
    mapping=None;expected_sha={};provenance={}
    for rank in range(8):
        up=C/f'student-rank{rank}-updates.jsonl'
        rows=list(map(json.loads,up.read_text().splitlines()))
        assert [r['update'] for r in rows]==list(range(1,1001))
        assert all(not r['skipped'] and r['grad_norm']>0 for r in rows)
        current=[r['physical_iteration'] for r in rows]
        assert current==sorted(set(current)) and current[0]==0 and current[-1]==1090
        if mapping is None:mapping=current
        assert current==mapping
        provenance[str(up)]=sha(up)
        actual=C/f'student-rank{rank}-samples.jsonl'
        samples=list(map(json.loads,actual.read_text().splitlines()))
        teacher=TEACHER/f'teacher-rank{rank}-samples.jsonl'
        oracle={r['micro']:r for r in map(json.loads,teacher.read_text().splitlines())}
        assert len(samples)==16000
        expected=[]
        for logical,physical in enumerate(mapping):
            for micro in range(16):
                row=samples[logical*16+micro]
                assert row['logical_iteration']==logical and row['physical_iteration']==physical
                selected={k:row[k] for k in ('micro','iteration','rank','sha256','masked_tokens')}
                assert selected==oracle[physical*16+micro]
                assert selected['masked_tokens']==4096
                expected.append(selected)
        path=OUT/f'expected-rank{rank}.jsonl'
        with path.open('x') as f:
            for row in expected:f.write(json.dumps(row,sort_keys=True)+'\n')
        expected_sha[str(rank)]=sha(path)
        provenance.update({str(actual):sha(actual),str(teacher):sha(teacher)})
    assert len(set(range(1091))-set(mapping))==91
    # Retain the exact accepted C config (not its learned state), including corpus digest.
    config.update(lm_coefficient=1.0,kd_disabled=True,comparison_kind='practical_deployed_recipe')
    put('run-config.json',config)
    put('replay-manifest.json',dict(accepted_physical_mapping=mapping,expected_sha256=expected_sha,
        approved_scratch_source=json.loads((C/'scratch.json').read_text())['path'],
        source_c=str(C),source_base=config['student_hf'],provenance=provenance,
        lm_coefficient=1.0,kd_disabled=True,coefficient_matched_ablation=False,comparison_kind='practical_deployed_recipe',
        sampler_budget=[468736,120,8],schedule_horizon=3662,stop_calls=1000,
        input_policy='new input read/digest fault is all-rank pre-forward fatal; no data reselection/retry/rebase',
        checkpoint_integrity='not_checked_explicit_user_waiver'))
    spec=importlib.util.spec_from_file_location('lmonly_supervisor',HERE/'supervisor.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    put('source-freeze.json',m.sources())
    print(json.dumps(dict(out=str(OUT),calls=1000,samples=128000,positions=524288000,
                         mapping_sha256=hashlib.sha256(json.dumps(mapping).encode()).hexdigest())))
if __name__=='__main__':main()
