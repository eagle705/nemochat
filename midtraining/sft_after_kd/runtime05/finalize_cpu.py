#!/usr/bin/env python3
"""Freeze runtime05 small source and aggregate genuine CPU gates; no GPU start."""
import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import controller as c
from readiness import NORMALIZED


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--freeze-only',action='store_true');args=parser.parse_args()
    c.OUT.mkdir(parents=True,exist_ok=True)
    frozen={path:c.sha(c.ROOT/path) for path in c.source_paths()}
    freeze=c.OUT/'source-freeze.json'
    if freeze.exists():c.require(c.read(freeze)==frozen,'existing runtime05 source freeze drift; preserve failed evidence before explicit CPU correction')
    else:c.write(freeze,frozen)
    freeze_sha=c.sha(freeze)
    if args.freeze_only:
        print(json.dumps(dict(status='SOURCE_ONLY_FROZEN_NO_GPU_RELEASE',sha256=freeze_sha,files=len(frozen))));return
    suites={}
    for name,suite,pattern in [('runtime05',HERE,'test_*.py'),('frozen_v1',HERE.parent,'test_controls.py')]:
        run=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(suite),'-p',pattern,'-v'],capture_output=True,text=True)
        (c.OUT/(name+'-tests.log')).write_text(run.stdout+run.stderr)
        count=re.search(r'Ran (\d+) tests?',run.stderr)
        c.require(run.returncode==0 and count is not None and int(count.group(1))>0,name+' CPU tests failed or empty')
        suites[name]=dict(exit_code=run.returncode,test_count=int(count.group(1)))
    old=c.ROOT/'results/midtraining/sft-after-kd-runtime04/source-freeze.json'
    oldmap=c.read(old)
    c.require(c.sha(old)=='4182e1ca4454d56d1f2abe85ec19ab94f940f76e5316c6254f7dccf70cda62d7' and all(c.sha(c.ROOT/path)==digest for path,digest in oldmap.items()),'runtime04 frozen source changed')
    c.write(c.OUT/'prior-runtime04-preservation.json',dict(status='EXACT_PRIOR_RUNTIME04_SOURCE_PRESERVED',source_freeze_sha256=c.sha(old),verified_files=len(oldmap),payloads_read=False))
    paths={
        'storage':c.OUT/'storage-cpu-proof.json',
        'collective':c.OUT/'cpu-collective-proof.json',
        'sdk':c.OUT/'config-sdk03/config-only.json',
        'guard':c.OUT/'config-sdk03/rank-0-source-guard.json',
        'regression':c.OUT/'config-sdk03/source-guard-regression.json',
        'actual_fd':c.OUT/'actual-sealed-cpu-proof.json'}
    proofs={k:c.read(p) for k,p in paths.items()}
    storage=proofs['storage'];actual=proofs['actual_fd']
    storage_blocked=storage['status']=='PINNED_CPU_MANAGER_PASS_NATIVE_STORAGE_BLOCKED_CUDA'
    c.require(storage['source_freeze_sha256']==freeze_sha,'storage final source lineage mismatch')
    if storage_blocked:
        c.require(storage['manager']['status']=='PINNED_TORCH_MANAGER_QUEUE_PASS' and storage['native_save_attempted'] is True and storage['native_save_completed'] is False and storage['no_cuda_mock_or_sdk_patch'] is True,'genuine native CUDA blocker and actual Manager proof required')
    else:
        c.require(storage['status']=='PINNED_CPU_MANAGER_AND_NATIVE_TORCH_DIST_STORAGE_PASS' and storage['native_save_manager_count']>0 and storage['loaded_exact_tensor_equal'] is True,'real native CPU storage gate missing')
    c.require(all(x['bytes']==len(x['path'].encode()) and x['bytes']<108 and x['path'].startswith('/nct/') for x in storage['all_manager_paths']),'actual manager path budget failed')
    c.require(len(proofs['collective']['ranks'])==8 and proofs['sdk']['native_full_epoch_updates']==34135 and proofs['sdk']['execution_update_cap']==16,'eight-rank fixture/full native horizon required')
    c.require(proofs['guard']['unchanged_sdk_guard_invoked'] is True and proofs['guard']['local_wrapper_identity_verified'] is True and proofs['guard']['runtime_source_freeze_sha256']==freeze_sha,'actual post-install guard not bound to final source')
    c.require(len(proofs['regression']['negative_controls'])==5 and set(proofs['regression']['negative_controls'].values())=={'DENIED'},'five denied sourceguard controls required')
    ready=c.ROOT/'results/midtraining/sft-after-kd-sealed03/READY.sealed.json'
    c.require(c.sha(ready)=='24498630aee5a3b1ad3f6aa3a2e8dbfcaec5fca98e3acb899d7fbc55678da5fb','original READY changed')
    c.require(actual['status']=='ACTUAL_SEALED_CLIENT_CPU_ADAPTER_PASS' and actual['runtime_source_freeze_sha256']==freeze_sha and actual['ready_sha256']==c.sha(ready) and len(actual['ranks'])==8,'real FD CPU integration final source binding missing')
    e=c.engine();engines={}
    for kind,p in [('storage',c.OUT/('storage-cpu-'+storage['attempt'])/'container.json'),('collective',c.OUT/'cpu-proof-container.json'),('sdk',c.OUT/'config-sdk03-container.json'),('actual_fd',c.OUT/'actual-cpu-container.json')]:
        cid=c.read(p)['id'];a=e.request('GET','/containers/'+cid+'/json')
        c.require(a['Id']==cid and a['Name'].startswith('/nemochat-sft-sealed03-r05-') and not a['HostConfig'].get('DeviceRequests') and a['State']['Running'] is False and a['State']['ExitCode']==(1 if kind=='storage' and storage_blocked else 0) and a['State']['OOMKilled'] is False,'exact expected-exit no-device CPU Engine required: '+kind)
        c.require(dict(x.split('=',1) for x in a['Config']['Env'])['TMPDIR']=='/nct' and '/nct' in a['HostConfig']['Tmpfs'],'CPU Engine did not use short initial environment')
        engines[kind]=dict(id=cid,name=a['Name'],state=a['State'])
    digests={k:c.sha(p) for k,p in paths.items()}
    template=dict(schema='nemochat-sealed-runtime05-parent-release-v1',status='PARENT_REVIEW_REQUIRED_NOT_RELEASED',input_mode='sealed_scm_rights',source_hf=c.SOURCE,
        source_freeze_sha256=freeze_sha,ready_path=str(ready),ready_sha256=c.sha(ready),snapshot_id=c.read(ready)['snapshot_id'],holder_identity=c.read(ready)['holder_identity'],
        producer_source_freeze_path='PARENT_FILL_ABSOLUTE_PRODUCER_FREEZE_FILE',producer_source_freeze_sha256=c.read(ready)['producer_source_freeze_sha256'],
        actual_cpu_adapter_receipt_sha256=digests['actual_fd'],actual_source_guard_receipt_sha256=digests['guard'],source_guard_regression_sha256=digests['regression'],storage_cpu_receipt_sha256=digests['storage'],
        benchmark_acceptance_sha256=c.sha(c.ROOT/'results/midtraining/offline2b-ablation-gpu-01/parent-final-acceptance.json'),checkpoint_integrity='not_checked_explicit_user_waiver',automatic_main_after_calibration=False,cpu_native_storage_blocker_acknowledged=False,
        normalization=dict(ready_sha256=c.sha(ready),**{key:False for key in NORMALIZED}))
    c.write(c.OUT/'parent-release.template.json',template)
    for stage in ('calibration','main'):c.write(c.OUT/(stage+'-inert-preview.json'),c.preview(stage))
    c.require(not (c.OUT/'parent-release.json').exists() and not list(c.OUT.glob('*-execution-claim.json')) and not list(c.OUT.glob('*-service-started.json')),'CPU preparation must not release or execute learner')
    result=dict(status='RUNTIME05_MANAGER_SOURCE_FD_PASS_NATIVE_CPU_STORAGE_BLOCKED_NOT_RELEASED' if storage_blocked else 'RUNTIME05_CPU_PREFLIGHTS_PASS_NOT_RELEASED',native_cpu_storage_blocked=storage_blocked,native_cpu_storage_blocker=storage.get('blocker'),source_freeze_sha256=freeze_sha,source_files=len(frozen),suites=suites,total_unit_tests=sum(x['test_count'] for x in suites.values()),
        proof_paths={k:str(v) for k,v in paths.items()},proof_sha256=digests,engines=engines,storage_actual_manager_paths=storage['all_manager_paths'],
        renewed_user_permission_required=True,production_service_started=False,gpu_launches=0,training_updates=0,automatic_main_after_calibration=False,
        full_model_optimizer_save_reload_proven=False,checkpoint_integrity='not_checked_explicit_user_waiver',source_hf=c.SOURCE,native_full_epoch_updates=34135,
        prior_attempt04=dict(executed_positive_lr_updates=16,committed_resumable_checkpoints=0,updates_not_recoverable_or_spliced=True),
        production_entrypoint='bash midtraining/sft_after_kd/runtime05/start.sh --calibration --release /raid/yoon/nemochat/results/midtraining/sft-after-kd-runtime05/parent-release.json',
        remaining=['Renewed user authorization and explicit parent release; template is NOT_RELEASED','Native CPU storage attempted but blocked by unconditional CUDA synchronize; parent must explicitly acknowledge before future GPU calibration' if storage_blocked else 'Native tiny CPU storage save/load passed', 'Genuine native GPU calibration16 finite/save16/full-state operational reload','Only after accepted calibration: fresh original-C main with fresh optimizer/RNG/cursor, never prior/calibration curve splice'],checked_at=time.time())
    c.write(c.OUT/'cpu-verification.json',result);c.write(c.OUT/'HANDOFF.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
