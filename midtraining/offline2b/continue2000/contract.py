"""Finite unique-source window, separate from immutable full native C."""
import hashlib
import json
from pathlib import Path
REPO=Path('/raid/yoon/nemochat')
OUT=REPO/'results/midtraining/offline2b-2000-01'
OLD=REPO/'results/midtraining/offline2b-prefixtrain-dynamic-restart-01'
SOURCE=REPO/'results/midtraining/offline2b-production1920-01'
CACHE_ID='8e2ba57bb9e10d4b6b5ca2fe82a7e1969c285acb392afca9121cbda29e36bd4d'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
def plan(teacher_stop=2064):
    if not 1143<teacher_stop<=3662:raise ValueError('outside original source capacity')
    return dict(target_calls=2000,restored_calls=16,teacher_start=1143,teacher_stop=teacher_stop,
        consumed_teacher_start=1143*128,sampler_budget=[468736,120,8],
        schedule=dict(horizon=3662,warmup=37,decay=366),physical_limit=teacher_stop,
        new_source_samples=(teacher_stop-1143)*128,new_source_positions=(teacher_stop-1143)*128*4096,
        teacher_requested_iterations=teacher_stop-1143,teacher_requested_rank_shards=(teacher_stop-1143)*8,
        total_source_samples=teacher_stop*128,cache_id=CACHE_ID,repeat=False,
        controller_seconds=86400,stage_seconds=43200,checkpoint_integrity='not_checked_explicit_user_waiver')
def remaining_capacity(calls,limit,mapping,excluded):
    mapping=list(mapping)
    if len(mapping)!=calls or mapping!=sorted(set(mapping)) or set(mapping)&set(excluded):
        raise ValueError('committed mapping invalid')
    return len(set(range(mapping[-1]+1 if mapping else 0,limit))-set(excluded))
def require_capacity(target,calls,limit,mapping,excluded):
    n=remaining_capacity(calls,limit,mapping,excluded)
    if calls+n<target:raise ValueError(f'unique source capacity {calls+n} < target {target}; no cycling')
    return n

def extend_state(binding,old_manifest,old_sha,limit):
    if (binding['cache_id']!=CACHE_ID or old_manifest['cache_id']!=CACHE_ID
        or binding['dynamic_manifest_sha256']!=old_sha
        or binding['next_iteration']!=16 or binding['consumed_samples']!=2048
        or binding['train_samples']!=468736 or binding['seed']!=43
        or binding['seq_length']!=4096 or binding['global_batch_size']!=128
        or binding['data_parallel_size']!=8 or binding['expert_parallel_size']!=8
        or not 1143<limit<=3662):raise ValueError('resume ancestry/full sampler identity mismatch')
    state=json.loads(json.dumps(binding['dynamic_state']))
    check=dict(state);saved=check.pop('mapping_sha256')
    if (digest(check)!=saved or state['committed_mapping']!=list(range(16))
        or state['optimizer_calls']!=16 or state['physical_limit']!=1143
        or state['next_unconsumed_physical']!=16
        or state['prefetched_unconsumed']!=[16,17]):raise ValueError('committed checkpoint16 cursor mismatch')
    state['physical_limit']=limit
    state.pop('mapping_sha256');state['mapping_sha256']=digest(state)
    return state

def check_new_coverage(p,rows,ranks):
    wanted=set(range(p['teacher_start'],p['teacher_stop']))
    if len(ranks)!=8 or [r['rank'] for r in ranks]!=list(range(8)):raise ValueError('all8 completion required')
    first=ranks[0];scored=set(first['scored']);bad=set(first['quarantined'])
    if scored&bad or scored|bad!=wanted:raise ValueError('new teacher window gap/overlap')
    for r in ranks:
        if (r['scored']!=sorted(scored) or r['quarantined']!=sorted(bad)
            or r['optimizer_calls']!=0 or r['frozen'] is not True or r['cache_id']!=CACHE_ID):
            raise ValueError('teacher rank identity/coverage mismatch')
    pairs=[(r['rank'],r['iteration']) for r in rows]
    if len(pairs)!=len(set(pairs)) or set(pairs)!={(r,i) for r in range(8) for i in scored}:
        raise ValueError('new native closed-shard coverage gap/duplicate')
    return sorted(scored)

def validate_diag(diag,pins_sha):
    if not diag:raise ValueError('parent diagnosis required before learner release')
    import re
    excluded=diag.get('known_global_input_exclusions',[])
    if (diag.get('status')!='parent_reviewed' or diag.get('source_pins_sha256')!=pins_sha
        or diag.get('unexpected_input_digest_mismatch')!='fatal'
        or not re.fullmatch('[0-9a-f]{64}',diag.get('diagnosis_evidence_sha256',''))
        or diag.get('input18_disposition') not in ('quarantine','verified_safe')
        or (diag['input18_disposition']=='quarantine' and 18 not in excluded)
        or excluded!=sorted(set(excluded)) or any(type(i)!=int or not 16<=i<2064 for i in excluded)):
        raise ValueError('invalid parent diagnosis policy')
    return excluded

def clean_engine(receipt,cid):
    s=receipt['state']
    if receipt['id']!=cid or s['Running'] or s['ExitCode']!=0 or s.get('OOMKilled',False):
        raise ValueError('exact clean Engine exit required')
    return True
