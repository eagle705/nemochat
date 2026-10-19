"""Durable wall deadline: exact owned Engine IDs only, never foreign jobs."""
import importlib.util
import json
from pathlib import Path
import time

ROOT=Path('/raid/yoon/nemochat')
OUT=ROOT/'results/midtraining/offline-kd-smoke-01'

def verify_owned(cid,actual):
    if (actual['Id']!=cid or not actual['Name'].startswith('/nemochat-offline-kd-smoke-01-')
        or actual['Config']['Labels'].get('nemochat.offline-kd-smoke.owner')!='offline-kd-smoke-01'
        or actual['HostConfig']['DeviceRequests'][0]['DeviceIDs']!=['0']):
        raise ValueError('refuse to stop a foreign or scope-mismatched container')

def main():
    spec=importlib.util.spec_from_file_location('engine',ROOT/'results/pretrain_followup/engine_helper.py')
    engine=importlib.util.module_from_spec(spec);spec.loader.exec_module(engine)
    rows=[]
    for stage in ('teacher','student','reload'):
        path=OUT/(stage+'-engine-created.json')
        if not path.exists():continue
        cid=json.loads(path.read_text())['id']
        actual=engine.request('GET',f'/containers/{cid}/json');verify_owned(cid,actual)
        stopped=bool(actual['State']['Running'])
        if stopped:
            engine.request('POST',f'/containers/{cid}/stop?t=10')
            actual=engine.request('GET',f'/containers/{cid}/json')
            assert not actual['State']['Running']
        rows.append(dict(id=cid,stage=stage,stopped_at_deadline=stopped,state=actual['State']))
    (OUT/'deadline.json').write_text(json.dumps(dict(time=time.time(),containers=rows),indent=2)+'\n')

if __name__=='__main__':main()
