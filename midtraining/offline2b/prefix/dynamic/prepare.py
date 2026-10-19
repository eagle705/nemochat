"""Freeze original full expectations and explicit latest dynamic policy."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
REPO=Path('/raid/yoon/nemochat');sys.path.insert(0,str(REPO))
OUT=REPO/'results/midtraining/offline2b-prefixtrain-dynamic-01'
HERE=Path(__file__).resolve().parent

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def put(name,value):
    with (OUT/name).open('x') as stream:json.dump(value,stream,sort_keys=True,indent=2);stream.write('\n')
def main():
    OUT.mkdir(exist_ok=False,parents=True)
    inherited=REPO/'results/midtraining/offline2b-prefixtrain-filtered-01/prefix-manifest.json'
    previous=json.loads(inherited.read_text())
    fullpath=Path(previous['full_expectations_path']);assert sha(fullpath)==previous['full_expectations_sha256']
    full=json.loads(fullpath.read_text());config=full['budget']
    assert config['train_iters']==3662 and config['train_samples']==468736
    assert {(x['rank'],x['iteration']) for x in full['shards']}=={(r,p) for r in range(8) for p in range(1143)}
    assert len(full['shards'])==9144
    rows={(x['rank'],x['iteration']):x for x in full['shards']}
    gate=fullpath.parent/'engine-02/cpu-02/shards.jsonl'
    observations=[]
    for bad in map(json.loads,gate.read_text().splitlines()):
        if bad['ok']:continue
        row=rows[bad['rank'],bad['iteration']]
        observations.append(dict(rank=bad['rank'],source_iteration=bad['iteration'],observation=dict(
            path=bad['path'],expected_sha256=row['sha256'],expected_bytes=row['stat'][2],
            observed_sha256=None,observed_bytes=None,error=bad['error'],historical=True,
            historical_observation_source=str(gate),note='old gate did not retain observed bytes/digest; no reconstruction')))
    rejected=REPO/'results/midtraining/offline2b-prefixtrain-filtered-01/cache-read-rejected-rank7.jsonl'
    import re
    bad=json.loads(rejected.read_text().splitlines()[0]);text=bad['error']
    fields={k:re.search(k+r'=([^ ]+)',text)[1] for k in ('expected_sha256','observed_sha256','expected_bytes','observed_bytes')}
    fields['expected_bytes']=int(fields['expected_bytes']);fields['observed_bytes']=int(fields['observed_bytes'])
    observations.append(dict(rank=7,source_iteration=20,observation=dict(fields,path=bad['path'],error=text,historical=True,historical_observation_source=str(rejected))))
    manifest=dict(cache_id=previous['cache_id'],original_full_cache_id=previous['cache_id'],sampler_budget=[468736,120,8],
        physical_limit=1143,max_accepted=1137,historical_global_exclusions=[20,86,98,121,122,126],historical_observations=observations,
        full_expectations_path=str(fullpath),full_expectations_sha256=sha(fullpath),native_reader_sha256=previous['native_reader_sha256'],
        approved_scratch_source=previous['approved_scratch_source'],
        policy='dynamically quarantine whole physical global batch collectively before any forward; mapping/model/optimizer/collective failures fatal',
        provenance={str(p):sha(p) for p in (inherited,fullpath,gate,rejected)},model_start='original_BASE19074_fresh_zero',
        checkpoint_integrity=config['checkpoint_integrity'])
    put('run-config.json',config);put('prefix-manifest.json',manifest)
    spec=importlib.util.spec_from_file_location('dynamic_supervisor',HERE/'supervisor.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    put('source-freeze.json',m.sources())
    print(json.dumps(dict(manifest_sha256=sha(OUT/'prefix-manifest.json'),out=str(OUT),historical=[20,86,98,121,122,126])))
if __name__=='__main__':main()
