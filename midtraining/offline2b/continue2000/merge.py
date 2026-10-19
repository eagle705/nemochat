"""Host-only clean-exit acceptance and immutable old/new identity routing."""
import json
import re
from pathlib import Path
from .contract import OLD,SOURCE,OUT,plan,sha,check_new_coverage,clean_engine,extend_state
from .broker import durable

def sample_rows(old,new,rank,scored):
    rows=[r for r in old if r['iteration']<1143]+list(new)
    expected={i*16+j for i in list(range(1143))+list(scored) for j in range(16)}
    keys=[r['micro'] for r in rows]
    if len(keys)!=len(set(keys)) or set(keys)!=expected:raise ValueError('actual teacher sample coverage gap/duplicate')
    for r in rows:
        if (set(r)!={'micro','iteration','rank','sha256','masked_tokens'} or r['rank']!=rank
            or r['micro']//16!=r['iteration'] or r['masked_tokens']!=4096
            or not re.fullmatch('[0-9a-f]{64}',r['sha256'])):raise ValueError('teacher actual sample identity invalid')
    return sorted(rows,key=lambda r:r['micro'])
def jsonl(path):return list(map(json.loads,path.read_text().splitlines()))

def accepted_policy(review):
    path=Path(review['diagnosis_policy_path']);expected=review['diagnosis_policy_sha256']
    if sha(path)!=expected:raise ValueError('parent diagnosis receipt changed')
    policy=json.loads(path.read_text())
    if (policy.get('known_global_quarantine_addition')!=[18] or policy.get('confirmed_source_index')!=2332
        or policy.get('source_index_formula')!='physical*128+micro*8+rank'
        or policy.get('post_acceptance_input_mutation_still_fatal') is not True
        or policy.get('model_optimizer_collective_logic_nonfinite_faults_still_fatal') is not True
        or policy.get('startup_source_review_still_required') is not True):raise ValueError('parent input policy not approved')
    evidence=path.parent/'input-digest-investigation-01'
    for name,digest in policy['evidence_sha256'].items():
        if Path(name).name!=name or sha(evidence/name)!=digest:raise ValueError('parent diagnosis evidence changed')
    return policy

def merge(review,engine_receipt):
    cid=json.loads((OUT/'teacher-engine-created.json').read_text())['id'];clean_engine(engine_receipt,cid)
    policy=accepted_policy(review);previous=json.loads((OLD/'prefix-manifest.json').read_text())
    prepared=json.loads((OUT/'preparation/prepared.json').read_text())
    if sha(OLD/'prefix-manifest.json')!=prepared['old_manifest_sha256']:raise ValueError('original ancestor changed')
    full=json.loads((OUT/'preparation/old-expectations.json').read_text())
    if sha(OUT/'preparation/old-expectations.json')!=prepared['old_expectations_sha256']:raise ValueError('original expectations changed')
    oldrows=list(full['shards']);oldpairs=[(r['rank'],r['iteration']) for r in oldrows]
    if len(oldpairs)!=9144 or set(oldpairs)!={(r,i) for r in range(8) for i in range(1143)}:raise ValueError('original exact closed prefix changed')
    ranks=[json.loads((OUT/f'teacher-rank{r}-complete.json').read_text()) for r in range(8)]
    rows=[row for r in range(8) for row in jsonl(OUT/f'teacher-rank{r}-shards.jsonl')]
    scored=check_new_coverage(plan(),rows,ranks);bad=ranks[0]['quarantined']
    for row in rows:
        from midtraining.offline2b.prefix.filtered.transport import identity
        expected=OUT/'cache'/f"cp0_dp{row['rank']}__{row['iteration']}.tar"
        if (row['path']!=str(expected) or row['stat']!=list(identity(expected.stat()))
            or not all(row[k] is True for k in ('native_codec','direct_sha','finite'))):raise ValueError('new closed cache identity/codec audit not accepted')
    if bad:
        ledger=json.loads((OUT/'teacher-input-quarantine.json').read_text())
        if [r['iteration'] for r in ledger]!=bad or any(not r['no_forward'] or r['optimizer_calls'] for r in ledger):
            raise ValueError('teacher pre-forward quarantine evidence incomplete')
    for rank in range(8):
        actual=sample_rows(jsonl(SOURCE/f'teacher-rank{rank}-samples.jsonl'),
            jsonl(OUT/f'teacher-rank{rank}-samples.jsonl'),rank,scored)
        with (OUT/f'merged-samples-rank{rank}.jsonl').open('x') as f:
            for row in actual:f.write(json.dumps(row)+'\n')
    full['shards']=oldrows+rows
    # No old data payload is rehashed or audited, and every old row is identical.
    if full['shards'][:9144]!=oldrows:raise ValueError('committed cache provenance modified')
    durable(OUT/'merged-expectations.json',full)
    manifest=dict(previous)
    observations=list(previous['historical_observations'])
    observations.append(dict(rank=4,source_iteration=18,observation=dict(
        expected_sha256=policy['expected_sha256'],observed_sha256=policy['current_buffered_observed_sha256'],
        source_index=2332,error='parent-confirmed input SourceDigestMismatch',
        historical=True,historical_observation_source=review['diagnosis_policy_path'],
        note='later CPU observation; original fatal observation SHA was not captured')))
    manifest['historical_observations']=observations
    manifest.update(physical_limit=2064,max_accepted=2064-len(set(previous['historical_global_exclusions']+bad+[18])),
        full_expectations_path=str(OUT/'merged-expectations.json'),full_expectations_sha256=sha(OUT/'merged-expectations.json'),
        historical_global_exclusions=sorted(set(previous['historical_global_exclusions']+bad+[18])),
        merged_shard_count=len(full['shards']),old_closed_iterations=1143,new_scored=scored,new_quarantined=bad,
        full_teacher_ready=False,coverage_end_exclusive=2064,model_start='FULL_restart01_checkpoint16',
        ancestry_dynamic_manifest_sha256=prepared['old_manifest_sha256'],diagnosis_policy_sha256=review['diagnosis_policy_sha256'],
        merged_sample_sha256={str(r):sha(OUT/f'merged-samples-rank{r}.jsonl') for r in range(8)},
        source_pins_sha256=sha(OUT/'preparation/source-pins.json'),teacher_clean_engine_id=cid)
    state=extend_state(json.loads((OUT/'resume-ancestry.json').read_text())['binding'],previous,prepared['old_manifest_sha256'],2064)
    durable(OUT/'extension-compatibility.json',dict(unchanged_committed_mapping=state['committed_mapping'],
        old_manifest_sha256=prepared['old_manifest_sha256'],original_cache_id=previous['cache_id'],
        original_sampler_budget=[468736,120,8],prefetch_reread=[16,17],expected_resume_step=16,consumed_samples=2048,
        optimizer_rng_full_restore_required=True,checkpoint_payload_integrity='not_checked_explicit_user_waiver'))
    durable(OUT/'prefix-manifest.json',manifest)
    ready=dict(status='accepted_window',full_teacher_ready=False,source_pins_sha256=sha(OUT/'preparation/source-pins.json'),
        manifest_sha256=sha(OUT/'prefix-manifest.json'),engine_receipt_sha256=sha(OUT/'teacher-engine-final.json'),
        teacher_engine_id=cid,coverage_end_exclusive=2064,new_rank_shards=len(rows),new_scored=len(scored),new_quarantined=len(bad),
        optimizer_calls=0,diagnosis_policy_sha256=review['diagnosis_policy_sha256'])
    durable(OUT/'window-ready.json',ready);return ready

def validate_ready(review):
    accepted_policy(review)
    ready=json.loads((OUT/'window-ready.json').read_text());manifest=json.loads((OUT/'prefix-manifest.json').read_text())
    if (ready['status']!='accepted_window' or ready['manifest_sha256']!=sha(OUT/'prefix-manifest.json')
        or ready['source_pins_sha256']!=sha(OUT/'preparation/source-pins.json')
        or ready['diagnosis_policy_sha256']!=review['diagnosis_policy_sha256']
        or ready['engine_receipt_sha256']!=sha(OUT/'teacher-engine-final.json')
        or manifest['full_expectations_sha256']!=sha(OUT/'merged-expectations.json')
        or manifest['cache_id']!=plan()['cache_id'] or manifest['sampler_budget']!=[468736,120,8]
        or manifest['full_teacher_ready'] is not False):raise ValueError('accepted window readiness/binding changed')
    clean_engine(json.loads((OUT/'teacher-engine-final.json').read_text()),ready['teacher_engine_id'])
    for rank,digest in manifest['merged_sample_sha256'].items():
        if sha(OUT/f'merged-samples-rank{rank}.jsonl')!=digest:raise ValueError('accepted actual sample log changed')
    return ready
