"""Exact completed GPU producer contract; call only after prerequisite service exit.

The producer writes complete.json while still holding flock. A marker alone
never permits SFT; caller must obtain the same cooperative lock and check idle.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess


def accept_actual(root):
    root=Path(root)
    out=root/'results/midtraining/offline2b-ablation-gpu-01'
    marker=out/'complete.json'
    if not marker.is_file():
        raise ValueError('exact GPU completion marker missing')
    proof=json.loads(marker.read_text())
    if (proof.get('status')!='verified_all_three_complete' or proof.get('all_three_complete') is not True or
            proof.get('unit')!='nemochat-offline2b-ablation-gpu-01.service'):
        raise ValueError('exact GPU completion contract mismatch')
    state=subprocess.run(['systemctl','--user','show',proof['unit'],'-p','MainPID','-p','ActiveState','-p','ExecMainStatus'],capture_output=True,text=True,check=True)
    props=dict(line.split('=',1) for line in state.stdout.splitlines() if '=' in line)
    if props.get('MainPID')!='0' or props.get('ActiveState')!='inactive' or props.get('ExecMainStatus')!='0':
        raise ValueError('exact prerequisite service must exit cleanly before marker acceptance')
    p=root/'midtraining/evaluation/ablation_gpu/controller.py'
    spec=importlib.util.spec_from_file_location('accepted_gpu_producer',p)
    producer=importlib.util.module_from_spec(spec);spec.loader.exec_module(producer)
    producer.verify_freeze()
    suite=producer.load_suite()
    if proof['suite_sha256']!=producer.SUITE_SHA:
        raise ValueError('GPU suite substitution')
    if (out/'failure.json').exists():
        raise ValueError('GPU producer failure retained; cannot release SFT')
    reports={};engines={};rows={}
    for arm in ('A','C','B'):
        expected=out/(arm+'-full01/report.json')
        if proof['verified_reports'].get(arm)!=str(expected):
            raise ValueError('exact GPU full report path required')
        r=json.loads(expected.read_text())
        producer.validate_report(r,suite,producer.MODELS[arm])
        owned=json.loads((out/(arm+'-full.owned.json')).read_text())
        x=producer.engine().request('GET','/containers/'+owned['id']+'/json')
        producer.clean_engine(x,owned['id'])
        if x['Config']['Labels'].get(producer.OWNER)!=producer.UNIT:
            raise ValueError('GPU Engine owner changed')
        producer.validate_devices(x['HostConfig']['DeviceRequests'],producer.spec(arm,'full')['HostConfig']['DeviceRequests'])
        for q in r['questions']:
            if json.loads((expected.parent/(q['id'].replace('/','__')+'.json')).read_text())!=q:
                raise ValueError('actual GPU per-question receipt mismatch')
        reports[arm]=dict(path=str(expected),sha256=hashlib.sha256(expected.read_bytes()).hexdigest())
        engines[arm]=dict(id=x['Id'],state=x['State'])
        rows[arm]=dict(question_ids=[q['id'] for q in r['questions']],heldout_window_ids=[w['id'] for w in r['heldout']],heldout_targets=r['heldout_target_count'],clean_exit=True,score=sum(q['acc'] for q in r['questions'])/400,heldout_nll=r['heldout_full_vocab_nll'])
    from PIL import Image
    if len(proof['images'])!=2:
        raise ValueError('two actual comparison PNGs required')
    for image in proof['images']:
        image=Path(image)
        if not image.is_absolute() or image.parent!=out:
            raise ValueError('exact comparison image scope required')
        with Image.open(image) as im:
            im.verify()
    return dict(status='accepted_exact_finished_GPU_benchmark',marker=str(marker),marker_sha256=hashlib.sha256(marker.read_bytes()).hexdigest(),producer_source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),service=props,reports=reports,engines=engines,images=proof['images'],normalized=dict(suite_sha256=producer.SUITE_SHA,backend='GPU BF16 Transformers 5.12.1 eager',clean_exit=True,gpu_lock_released=True,models=rows),lock_reacquisition_and_live_idle_still_required=True)
