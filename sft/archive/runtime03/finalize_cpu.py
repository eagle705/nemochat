#!/usr/bin/env python3
"""Freeze only runtime03 allowlisted source, CPU results and inert templates."""
import json,os,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import controller as c
from readiness import NORMALIZED,C_MANIFEST_SHA

def main():
    c.OUT.mkdir(parents=True,exist_ok=True)
    suites={}
    for name,args in [('runtime03',['discover','-s',str(HERE),'-p','test_*.py','-v']),('frozen_v1',['discover','-s',str(HERE.parent),'-p','test_controls.py','-v'])]:
        run=subprocess.run([sys.executable,'-m','unittest',*args],capture_output=True,text=True)
        (c.OUT/(name+'-tests.log')).write_text(run.stdout+run.stderr)
        import re
        matched=re.search(r'Ran (\d+) tests?',run.stderr)
        c.require(run.returncode==0 and matched is not None and int(matched.group(1))>0,name+' CPU tests failed or empty')
        suites[name]={'exit_code':run.returncode,'test_count':int(matched.group(1)),'log':str(c.OUT/(name+'-tests.log'))}
    # Weight payloads are never opened. Source metadata contains full HF count.
    p=c.ROOT/c.SOURCE;index=c.read(p/'model.safetensors.index.json');config=c.read(p/'config.json')
    c.write(c.OUT/'source-metadata.json',dict(status='metadata_only_no_weights_loaded',source_hf=c.SOURCE,
        config_sha256=c.sha(p/'config.json'),index_sha256=c.sha(p/'model.safetensors.index.json'),
        full_source_parameters=index['metadata']['total_parameters'],source_tensor_keys=len(index['weight_map']),
        embedding_expected_shape=[131072,1024],embedding_import='existing_native_transform_and_full_destination_coverage_guard',
        source_shard_bytes={n:(p/n).stat().st_size for n in sorted(set(index['weight_map'].values()))},
        checkpoint_payload_sha_crc='not_checked_explicit_user_waiver'))
    frozen={path:c.sha(c.ROOT/path) for path in c.source_paths()}
    c.write(c.OUT/'source-freeze.json',frozen);freeze_sha=c.sha(c.OUT/'source-freeze.json')
    # Preserve and verify previous immutable source generations.
    old=c.ROOT/'results/midtraining/sft-after-kd-runtime02/source-freeze.json'
    oldmap=c.read(old)
    c.require(c.sha(old)=='b0791752795a7340112503b7421e7fa840e99c40b44575d1373f1e7bbfc3c65b' and all(c.sha(c.ROOT/path)==digest for path,digest in oldmap.items()),'frozen runtime02/v1 source drift')
    c.write(c.OUT/'v1-runtime02-preservation.json',dict(status='EXACT_PRIOR_SOURCE_FREEZE_PRESERVED',runtime02_freeze_sha256=c.sha(old),verified_source_files=len(oldmap)))
    template=dict(schema='nemochat-sealed-runtime03-parent-release-v1',status='PARENT_REVIEW_REQUIRED_NOT_RELEASED',input_mode='sealed_scm_rights',source_hf=c.SOURCE,
        source_freeze_sha256=freeze_sha,ready_path=str(c.ROOT/'results/midtraining/sft-after-kd-sealed03/READY.sealed.json'),ready_sha256='PARENT_FILL_RAW_READY_SHA256',snapshot_id='PARENT_FILL_SNAPSHOT_ID',holder_identity={},
        producer_source_freeze_path='PARENT_FILL_ABSOLUTE_PRODUCER_FREEZE_FILE',producer_source_freeze_sha256='PARENT_FILL_PRODUCER_FREEZE_SHA256',
        actual_cpu_adapter_receipt_sha256='PARENT_FILL_ACTUAL_SEALED_CPU_PROOF_SHA256',benchmark_acceptance_sha256=c.sha(c.ROOT/'results/midtraining/offline2b-ablation-gpu-01/parent-final-acceptance.json'),
        checkpoint_integrity='not_checked_explicit_user_waiver',automatic_main_after_calibration=False,
        normalization=dict(ready_sha256='PARENT_FILL_RAW_READY_SHA256',**{key:False for key in NORMALIZED}))
    c.write(c.OUT/'parent-release.template.json',template)
    for stage in ('calibration','main'):c.write(c.OUT/(stage+'-inert-preview.json'),c.preview(stage))
    c.require(c.sha(c.MANIFEST)==c.MANIFEST_SHA and c.sha(c.ROOT/'datasets/midtraining/super35-kd-01/offline1920-v1/data/manifest.json')==C_MANIFEST_SHA,'immutable manifest literal mismatch')
    collective=c.read(c.OUT/'cpu-collective-proof.json');sdk=c.read(c.OUT/'config-sdk02/config-only.json')
    c.require(len(collective['ranks'])==8 and sdk['native_full_epoch_updates']==34135 and sdk['execution_update_cap']==16,'CPU proof scope mismatch')
    c.write(c.OUT/'cpu-verification.json',dict(status='CPU_ONLY_TESTS_AND_PINNED_FIXTURES_PASS',suites=suites,pinned_collective=str(c.OUT/'cpu-collective-proof.json'),pinned_sdk=str(c.OUT/'config-sdk02/config-only.json'),source_freeze_sha256=freeze_sha,actual_snapshot_integration=False,gpu_launches=0))
    result=dict(status='NEW_SEALED_RAM_RUNTIME_CPU_TESTED_PARENT_REVIEW_REQUIRED',source_freeze_sha256=freeze_sha,source_files=len(frozen),input_mode='sealed_scm_rights',
        production_entrypoint='bash sft/archive/runtime03/start.sh --calibration --release /raid/yoon/nemochat/results/midtraining/sft-after-kd-runtime03/parent-release.json',
        parent_actual_cpu_integration='.venv/bin/python sft/archive/runtime03/integrate_snapshot.py --ready /raid/yoon/nemochat/results/midtraining/sft-after-kd-sealed03/READY.sealed.json',
        cpu_verification=str(c.OUT/'cpu-verification.json'),source_hf=c.SOURCE,full_source_parameters=index['metadata']['total_parameters'],native_full_epoch_updates=34135,execution_calibration_updates=16,
        production_service_started=False,gpu_launches=0,training_updates=0,actual_snapshot_ready_polled=False,actual_producer_client_integrated=False,
        remaining=['Parent review/recompute producer full all7 SFT/all16 C and tokenizer/provenance audit proofs','Parent run actual producer-client CPU integration using supplied no-device executable','Parent bind raw READY/snapshot/live holder/runtime freeze/producer freeze/actual CPU proof and start already-authorized calibration','Actual native finite-gradient learning16/full-state save/operational reload then fresh-C main and final saved on-time owned cleanup'],
        limitations=['Pinned eight-process fixture uses a TEST_LOCAL public API stub, real memfd/SCM_RIGHTS/Gloo but NOT real full03 producer','No GPU model/optimizer run by this worker','Initial pinned SDK CPU attempt failed missing libcuda; preserved config-sdk*, corrected no-device readonly-library config-sdk02 exited0','Native microbatch MIN guard prevents bad microbatch forward/failed update, not rollback of earlier good microbatch forwards'],checked_at=time.time())
    c.write(c.OUT/'HANDOFF.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
