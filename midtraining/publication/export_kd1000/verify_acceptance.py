"""Read back exact accepted files and owned stopped Engine containers."""
import collections,json,os,sys
from pathlib import Path
ROOT=Path('/raid/yoon/nemochat');R=ROOT/'results/midtraining/hf-kd1000-conversion-01'
sys.path.insert(0,str(ROOT/'results/pretrain_followup'))
from engine_helper import request
e=json.loads((R/'export_result.json').read_text());v=json.loads((R/'validation_result.json').read_text());p=Path(e['output'])
files={x.name:x.stat().st_size for x in p.iterdir() if x.is_file()}
assert files==e['files']
assert all(os.access(x,os.R_OK) and x.stat().st_uid==1001 for x in p.iterdir())
index=json.loads((p/'model.safetensors.index.json').read_text())
# Official Transformers reverses its checkpoint conversion on save:
# model -> backbone and stacked experts -> individual expert records.
import re
provenance=json.loads((R/'mapping_provenance.json').read_text())
serialized_to_runtime={}
for key in index['weight_map']:
    runtime=key.replace('backbone.','model.',1).replace('model.embedding.weight','model.embeddings.weight')
    runtime=re.sub(r'\.experts\.\d+\.(up_proj|down_proj)\.weight$',r'.experts.\1',runtime)
    assert runtime in provenance,(key,runtime)
    serialized_to_runtime[key]=runtime
assert set(serialized_to_runtime.values())==set(provenance)
assert len(index['weight_map'])==len(provenance)-23*2+23*2*128
(R/'serialized_to_runtime_keys.json').write_text(json.dumps(serialized_to_runtime,indent=2))
assert v['source']==e['source'] and v['status']=='passed'
assert not any(v['loading_info'].values())
assert v['full_model_reload'] and not v['trust_remote_code']
assert len(v['cases'])==2 and all(c['finite'] and c['generation_scores_finite'] and c['generation_cache'] and c['new_tokens']==4 for c in v['cases'])
states={};specs={}
for create in sorted(R.glob('nemochat-*.create.json')):
    receipt=json.loads(create.read_text());info=request('GET',f"/containers/{receipt['id']}/json")
    state=info['State'];assert not state['Running'] and state['Pid']==0 and state['ExitCode']==0 and not state['OOMKilled']
    host=info['HostConfig'];assert not host.get('DeviceRequests') and not host.get('Devices') and host['NetworkMode']=='none'
    assert host['Memory']==137438953472 and host['NanoCpus']==4000000000
    states[create.stem]={'id':receipt['id'],'state':state,'device_requests':host.get('DeviceRequests'),'devices':host.get('Devices'),'network':host['NetworkMode'],'memory_limit':host['Memory'],'nano_cpus':host['NanoCpus'],'image':info['Image']}
exclusions=collections.Counter(json.loads((R/'excluded_native_records.json').read_text()).values())
report={'status':'accepted_CPU_model_executed','source':e['source'],'output':str(p),'validation_receipt':str(R/'validation_result.json'),'file_count':len(files),'safetensors_shard_count':len(list(p.glob('*.safetensors'))),'total_file_bytes':sum(files.values()),'files':files,'host_readable':True,'export_owner_uid':1001,'source_model_records':619,'hf_runtime_records':len(provenance),'serialized_safetensors_records':len(index['weight_map']),'serialized_to_runtime_provenance':str(R/'serialized_to_runtime_keys.json'),'excluded_record_reasons':dict(exclusions),'source_byte_audit':'explicitly_waived_not_run','native_hf_forward_parity':'not_run','published_to_hub':False,'containers':states}
(R/'final_acceptance.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:val for k,val in report.items() if k not in ['files','containers']},indent=2));print('verified stopped containers',len(states))
