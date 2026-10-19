"""Authenticate inherited complete CPU gate, filter inventory, freeze fresh run."""
import hashlib,importlib.util,json,sys
from pathlib import Path
REPO=Path('/raid/yoon/nemochat');sys.path.insert(0,str(REPO))
from midtraining.offline2b.prefix.filtered.mapping import plan,identity,PHYSICAL
OUT=REPO/'results/midtraining/offline2b-prefixtrain-filtered-01'
PARENT=REPO/'results/midtraining/offline1920-parent-01'
GATE=PARENT/'directio-acceptance-01'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def put(name,data):
    p=OUT/name
    with p.open('x') as f:json.dump(data,f,indent=2);f.write('\n')
def main():
    request=PARENT/'filtered-exclusion-request.json';policy=json.loads(request.read_text())
    assert policy['physical_source_iterations']==list(PHYSICAL)
    acceptance=json.loads((GATE/'acceptance.json').read_text())
    expectation=GATE/'expectations.json';full=json.loads(expectation.read_text())
    assert digest(expectation)==acceptance['manifest_sha256']
    rows=[json.loads(x) for x in (GATE/'engine-02/cpu-02/shards.jsonl').read_text().splitlines()]
    pairs=[(x['rank'],x['iteration']) for x in rows]
    assert len(pairs)==len(set(pairs))==9144
    assert set(pairs)=={(r,i) for r in range(8) for i in range(1143)}
    bad={(x['rank'],x['iteration']) for x in rows if not x['ok']}
    assert bad=={(6,86),(7,98),(7,121),(3,122),(6,122),(5,126)}
    passed={(x['rank'],x['iteration']):x for x in rows if x['ok']}
    kept=[x for x in full['shards'] if x['iteration'] in PHYSICAL]
    expected={(r,i) for r in range(8) for i in PHYSICAL}
    assert len(kept)==9104 and {(x['rank'],x['iteration']) for x in kept}==expected
    for x in kept:
        prior=passed[(x['rank'],x['iteration'])]
        assert prior['sha256']==x['sha256'] and prior['direct_sha_match'] and prior['native'] and prior['metadata'] and prior['zip_crc'] and prior['zstd']
    native=Path(full['native_source']);scratch=native.parents[6]
    assert (scratch/'sdk').is_dir() and (scratch/'staged').is_dir(),scratch
    reader=native.parent/'cached_logits_loss.py'
    config=full['budget'];cache_id=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    put('run-config.json',config)
    put('prefix-manifest.json',dict(cache_id=cache_id,original_full_cache_id=cache_id,sampler_budget=[468736,120,8],
        mapping=plan(),filtered_mapping_sha256=identity(),shards=kept,native_contract=full['native_contract'],native_reader_sha256=digest(reader),
        full_expectations_path=str(expectation),full_expectations_sha256=digest(expectation),approved_scratch_source=str(scratch),
        inherited_gate_engine_id=acceptance['engine_id'],inherited_gate_good=9138,inherited_gate_bad=6,
        full_gate_preserved_blocked=True,user_authorized_global_exclusion=True,metadata_hash=full['metadata_hash'],
        provenance={str(p):digest(p) for p in [request,GATE/'acceptance.json',GATE/'engine-02/cpu-02/shards.jsonl',GATE/'blocking-files.json']},
        kept_shards=9104,excluded_shards=40,additional_good_shards_excluded=34))
    put('release.json',dict(status='authorized_filtered_cpu_gate_reuse',kept=9104,logical_updates=1138,optimizer_calls_before_start=0,positive_lr_updates_before_start=0,request_sha256=digest(request)))
    spec=importlib.util.spec_from_file_location('filtered_supervisor',Path(__file__).parent/'supervisor.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    put('source-freeze.json',m.sources())
    print(json.dumps(dict(kept=9104,logical_updates=1138,source_scratch=str(scratch),manifest_sha256=digest(OUT/'prefix-manifest.json'))))
if __name__=='__main__':main()
