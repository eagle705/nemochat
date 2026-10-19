"""Read-only producer receipt and independent bounded read evidence."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import verify

out=verify.OUT
spec=importlib.util.spec_from_file_location('engine','results/pretrain_followup/engine_helper.py')
e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)
p=Path('results/pretraining/nemochat_baby/nemotron_sft/cpu-full-03/full-preparation.final-state.json')
historical=json.loads(p.read_text())
try:
 live=e.request('GET','/containers/'+historical['Id']+'/json')
 verify.save(out/'producer-current-engine.json',{'id':live['Id'],'State':live['State'],'Path':live['Path'],'Args':live['Args'],'Image':live['Config']['Image'],'Mounts':live['Mounts'],'HostConfig':{k:live['HostConfig'].get(k) for k in ['DeviceRequests','Devices','NetworkMode','Memory','NanoCpus']}})
except Exception as error:verify.save(out/'producer-current-engine.json',{'error':repr(error),'historical_container_id':historical['Id']})
records=[]
for p in out.glob('sft-*.json'):
 r=json.loads(p.read_text());w=r.get('first_difference')
 if not w:continue
 source=Path(r['path']);offset=w['page_offset'];length=Path(w['direct_snapshot']).stat().st_size
 rec={'source':str(source),'offset':offset,'length':length,'initial_evidence':w,'repeats':[]}
 for i in range(2):
  for mode in ('buffered','direct'):
   cmd=['dd','if='+str(source),'bs=4096','skip='+str(offset),'count='+str(length),'iflag=skip_bytes,count_bytes'+(',direct' if mode=='direct' else ''),'status=none']
   z=subprocess.run(cmd,capture_output=True,check=True);b=z.stdout;dest=out/(p.stem+'-range-'+mode+'-'+str(i)+'.bin');dest.write_bytes(b)
   rec['repeats'].append({'mode':mode,'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b),'snapshot':str(dest),'dd_exit':z.returncode,'same_as_initial_buffered':b==Path(w['buffered_snapshot']).read_bytes(),'same_as_initial_direct':b==Path(w['direct_snapshot']).read_bytes()})
 records.append(rec)
verify.save(out/'small-range-independent-evidence.json',records)
print(json.dumps({'ranges':len(records),'observations':[(r['source'],[(v['mode'],v['sha256'],v['same_as_initial_buffered'],v['same_as_initial_direct']) for v in r['repeats']]) for r in records]},indent=2))
