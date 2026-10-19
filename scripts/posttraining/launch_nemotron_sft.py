#!/usr/bin/env python3
"""Parent-only all8 native launcher. Default --preview is inert; --execute is explicit."""
import argparse
from contextlib import ExitStack
import fcntl
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import socket
import struct
import subprocess
import sys
import hashlib
import shlex

ROOT = Path(__file__).resolve().parents[2]
RESULTS = Path('results/pretraining/nemochat_baby/nemotron_sft')
IMAGE = 'nvcr.io/nvidia/nemo@sha256:fdd6e9c7929b76c8624ddfea939dda345beaf67695d9c3994994c18f9e79b9bf'


def load_module(path,name):
    spec = importlib.util.spec_from_file_location(name,path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def watchdog_seconds(plan):
    return int(plan['max_hours']*3600 + plan.get('watchdog_grace_minutes',30)*60)


def runner_provenance(config_path, manifest_path):
    scripts = [ROOT/'scripts/posttraining'/name for name in ('launch_nemotron_sft.py','native_nemotron_sft.py')]
    return dict(config_sha256=hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
        manifest_sha256=hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest(),
        scripts_sha256={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in scripts},
        checkpoint_integrity='not_checked_explicit_user_waiver',checkpoint_bytes_read=False)


def supervisor_command(name, argv, plan):
    cleanup = [sys.executable,str(ROOT/'scripts/posttraining/launch_nemotron_sft.py'),'--name',name,'--cleanup-owned']
    return ['systemd-run','--user','--unit=nemotron-sft-'+name,'--service-type=exec',
        '--working-directory='+str(ROOT), '--property=RuntimeMaxSec='+str(watchdog_seconds(plan)+240),
        '--property=TimeoutStopSec=120','--property=KillMode=control-group','--property=NoNewPrivileges=yes',
        '--property=ExecStopPost='+shlex.join(cleanup),
        '--setenv=NEMOTRON_SFT_SUPERVISED='+name,
        '--property=StandardOutput=append:'+str(ROOT/RESULTS/(name+'-supervisor.log')),
        '--property=StandardError=append:'+str(ROOT/RESULTS/(name+'-supervisor.log')),
        sys.executable,str(ROOT/'scripts/posttraining/launch_nemotron_sft.py')]+[a for a in argv if a != '--supervise']


def cleanup_owned(name):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}',name):
        raise ValueError('safe owned run name required')
    output = ROOT/RESULTS/'runs'/name
    if not (output/'container-id.json').exists():
        return dict(status='not_created')
    engine = load_module(ROOT/'results/pretrain_followup/engine_helper.py','nemotron_cleanup_engine')
    cid = json.loads((output/'container-id.json').read_text())['id']
    state = engine.request('GET','/containers/'+cid+'/json')
    if (state['Config']['Labels'].get('nemochat.validation.name') != name or
            state['Config']['Labels'].get('nemochat.validation.session') != 'nemochat-baby-nemotron-sft'):
        raise ValueError('owned Engine identity mismatch; refusing termination')
    was_running = state['State']['Running']
    if was_running:
        engine.request('POST','/containers/'+cid+'/kill')
        engine.request('POST','/containers/'+cid+'/wait?condition=not-running',timeout=120)
    state = engine.request('GET','/containers/'+cid+'/json')['State']
    if state['Running'] or state['Pid']:
        raise RuntimeError('owned Engine cleanup not verified')
    receipt = dict(id=cid,status='watchdog_or_supervisor_failure' if was_running else 'already_exited',
                   state=state,was_running=was_running)
    (output/'owned-cleanup.json').write_text(json.dumps(receipt,indent=2))
    return receipt


def validate_calibration(plan, manifest_path):
    path = plan.get('calibration_receipt')
    if not path or not Path(path).is_file():
        raise ValueError('accepted new-data native GB128 calibration receipt required')
    receipt = json.loads(Path(path).read_text())
    if plan.get('sequence_length') == 16384 and any(receipt.get(k) != plan.get(k,default)
            for k,default in (('sequence_length',16384),('context_parallel_size',1),('expert_model_parallel_size',8))):
        raise ValueError('calibration sequence/topology mismatch; old 4K receipts cannot authorize 16K')
    expected = hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest()
    if (receipt.get('status') != 'passed' or receipt.get('global_batch_size') != 128 or
            receipt.get('manifest_sha256') != expected or receipt.get('warm_updates',0) < 10 or
            receipt.get('source_checkpoint') != plan['source_checkpoint'] or receipt.get('ranks') != list(range(8))):
        raise ValueError('calibration lineage, warm window or all8 acceptance mismatch')
    verification_path = Path(receipt.get('verification',''))
    if not verification_path.is_file():
        raise ValueError('calibration requires accepted native producer verification')
    verification = json.loads(verification_path.read_text())
    if (verification.get('status') != 'passed' or verification.get('updates',0) < 15 or
            {row['rank'] for row in verification.get('ranks',[])} != set(range(8)) or
            not verification.get('final_validation_accepted')):
        raise ValueError('calibration requires accepted native all8/final-heldout verification')
    seconds = receipt['mean_iteration_seconds']
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError('calibration timing invalid')
    if plan['updates']*seconds/(1-plan['assumed_overhead_fraction']) < plan['max_hours']*3600:
        raise ValueError('calibrated duration budget too short; refreeze explicit corpus/epochs/horizon')
    return receipt


def make_payload(name,manifest,*,pilot=False,online=False,run_id=None,config_only=False,config=None,duration_control=False):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}',name):
        raise ValueError('safe fresh run name required')
    manifest = Path(manifest)
    if manifest.is_absolute():
        manifest = manifest.resolve().relative_to(ROOT)
    if '..' in manifest.parts:
        raise ValueError('repository-local manifest required')
    if online and (not run_id or not re.fullmatch(r'nemochat-baby-nemotron-sft-[A-Za-z0-9_-]+',run_id)):
        raise ValueError('distinct stable Nemotron SFT run ID required')
    output = RESULTS/'runs'/name
    argv = ['torchrun','--nnodes=1','--nproc_per_node=8','--master_addr=127.0.0.1','--master_port=29783',
            'scripts/posttraining/native_nemotron_sft.py','--manifest',str(manifest),'--output',str(output)]
    if pilot:
        argv.append('--pilot')
    if duration_control:
        argv.append('--duration-control')
    if config is not None:
        config = Path(config)
        if config.is_absolute():
            config = config.resolve().relative_to(ROOT)
        if '..' in config.parts:
            raise ValueError('repository-local config required')
        argv += ['--config',str(config)]
    if online:
        argv.append('--online')
    if config_only:
        argv.append('--config-only')
    env = ['PYTHONPATH=/workspace/src','PYTHONDONTWRITEBYTECODE=1','HF_HUB_OFFLINE=1','TRANSFORMERS_OFFLINE=1',
           'OMP_NUM_THREADS=2','NCCL_SOCKET_IFNAME=lo','GLOO_SOCKET_IFNAME=lo','HOME=/workspace/'+str(output/'home'),
           'WANDB_MODE='+('online' if online else 'disabled'),'WANDB_DISABLE_CODE=true','WANDB_DISABLE_GIT=true','WANDB_SILENT=true']
    if online:
        env += ['WANDB_RUN_ID='+run_id,'WANDB_RESUME=never','WANDB_RUN_GROUP=nemochat-baby-nemotron-sft','WANDB_JOB_TYPE=nemotron-sft']
    return dict(Image=IMAGE,Entrypoint=argv[:1],Cmd=argv[1:],User='0:0',WorkingDir='/workspace',Tty=False,Env=env,
        Labels={'nemochat.validation.session':'nemochat-baby-nemotron-sft','nemochat.validation.name':name},
        HostConfig=dict(NetworkMode='bridge' if online else 'none',Privileged=False,CapDrop=['ALL'],CapAdd=['DAC_OVERRIDE'],
            SecurityOpt=['no-new-privileges:true'],Binds=[str(ROOT)+':/workspace:ro',str(ROOT/output)+':/workspace/'+str(output)+':rw'],
            Memory=412316860416,ShmSize=17179869184,RestartPolicy={'Name':'no'},
            DeviceRequests=[dict(Driver='nvidia',DeviceIDs=list(map(str,range(8))),Capabilities=[['gpu']])]))


def verify_run(output,plan,manifest):
    state = json.loads((output/'container-state.json').read_text())['state']
    if state['Running'] or state['ExitCode'] != 0 or state['Pid'] != 0 or state.get('OOMKilled'):
        raise ValueError('native Engine exit not accepted')
    ranks = []
    for rank in range(8):
        load = json.loads((output/f'rank-{rank}-load.json').read_text())
        entry = json.loads((output/f'rank-{rank}-entrypoint.json').read_text())
        save = json.loads((output/f'rank-{rank}-save.json').read_text())
        updates = [json.loads(line) for line in (output/f'rank-{rank}-updates.jsonl').open()]
        if load['source'] != plan['source_checkpoint'] or load['source_sft_update'] != 391 or load['stage_step'] != 0 or not load['model_only']:
            raise ValueError('SFT391/reset lineage mismatch')
        performed = entry['updates']
        extended = plan.get('execution_mode','one_epoch') != 'one_epoch'
        reason = entry.get('termination_reason', 'completed_budget')
        if entry['native_exit_code'] != 0 or not 0 < performed <= plan['updates'] or len(updates) != performed:
            raise ValueError('native optimizer updates not accepted')
        if performed < plan['updates'] and not (extended and reason == 'native_duration_limit' and entry['elapsed_seconds'] >= plan['max_hours']*3600):
            raise ValueError('premature unexpected native exit')
        if [u['update'] for u in updates] != list(range(1,performed+1)):
            raise ValueError('update sequence mismatch')
        if performed not in save['save_steps'] or not save['save_optim'] or not save['save_rng']:
            raise ValueError('full-state final native save missing')
        for update in updates:
            if not all(math.isfinite(update[k]) for k in ('loss','grad_norm','lr')) or update['grad_norm'] <= 0 or not update['finite_positive_gradient']:
                raise ValueError('finite-positive native update gate failed')
        coverage = json.loads((output/f'rank-{rank}-coverage.json').read_text()) if extended else dict(loss_tokens=manifest['train_loss_tokens'])
        if sum(u['global_loss_tokens'] for u in updates) != coverage['loss_tokens']:
            raise ValueError('native assistant-target coverage not exactly one unique epoch')
        if sum(u['global_processed_positions'] for u in updates) != performed*plan['global_batch_size']*plan['sequence_length']:
            raise ValueError('processed padded-position coverage mismatch')
        if plan.get('require_final_validation'):
            path = output/f'rank-{rank}-heldout.jsonl'
            heldout = [json.loads(line) for line in path.open()] if path.exists() else []
            if not any(row['update'] == performed and all(math.isfinite(row[key]) for key in ('heldout_ce','heldout_ppl')) for row in heldout):
                raise ValueError('final heldout native evaluation missing or nonfinite')
        ranks.append(dict(rank=rank,updates=len(updates),save_steps=save['save_steps'],termination_reason=reason,
                          elapsed_seconds=entry.get('elapsed_seconds'),coverage=coverage))
    if len({(r['updates'],r['termination_reason']) for r in ranks}) != 1:
        raise ValueError('all8 completed-prefix mismatch')
    return dict(status='passed',stage='nemotron-sft',pilot=plan['pilot'],ranks=ranks,updates=performed,
                final_validation_accepted=bool(plan.get('require_final_validation')),
                planned_updates=plan['updates'],termination_reason=reason,target_duration_reached=reason=='native_duration_limit',
                source_checkpoint=plan['source_checkpoint'],checkpoint_integrity=plan['checkpoint_integrity'],checkpoint_bytes_read=False)


def main():
    import yaml
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name',required=True)
    parser.add_argument('--manifest',type=Path)
    parser.add_argument('--cleanup-owned',action='store_true',help='supervisor exact-owned-Engine cleanup only')
    parser.add_argument('--config',type=Path,default=Path('configs/posttraining/nemochat_baby_nemotron_sft.yaml'))
    parser.add_argument('--pilot',action='store_true')
    parser.add_argument('--duration-control',action='store_true')
    parser.add_argument('--online',action='store_true')
    parser.add_argument('--run-id')
    execution = parser.add_mutually_exclusive_group()
    execution.add_argument('--execute',action='store_true')
    parser.add_argument('--supervise',action='store_true',help='durable parent-owned user-systemd launcher; execute only')
    execution.add_argument('--preview',action='store_true')
    parser.add_argument('--config-only',action='store_true',help='still requires parent GPU authorization; no weight loading')
    parser.add_argument('--parent-lock-fd',type=int,action='append',default=[],help='both inherited global and all8 lock FDs, in order')
    args = parser.parse_args()
    if args.cleanup_owned:
        print(json.dumps(cleanup_owned(args.name)))
        return
    if not args.manifest:
        parser.error('--manifest required')
    native = load_module(ROOT/'scripts/posttraining/native_nemotron_sft.py','nemotron_native_launcher')
    config = yaml.safe_load(args.config.read_text())
    manifest = json.loads(args.manifest.read_text())
    plan = native.stage_plan(config,manifest,pilot=args.pilot,duration_control=args.duration_control)
    verification = native.verify_inputs(args.manifest,config['tokenizer_manifest'])
    payload = make_payload(args.name,args.manifest,pilot=args.pilot,online=args.online,run_id=args.run_id,
                           config_only=args.config_only,config=args.config,duration_control=args.duration_control)
    if not args.execute:
        print(json.dumps(dict(status='inert preview; no GPU launch',plan=plan,data_verification=verification,payload=payload),indent=2))
        return
    if args.supervise:
        if args.parent_lock_fd:
            raise ValueError('supervisor acquires its own cooperative locks; inherited FDs cannot cross systemd')
        subprocess.run(supervisor_command(args.name,sys.argv[1:],plan),check=True)
        print(subprocess.check_output(['systemctl','--user','show','nemotron-sft-'+args.name+'.service',
              '-p','MainPID','-p','ActiveState','-p','SubState','-p','Result','-p','ExecMainStatus'],text=True))
        return
    if plan.get('execution_mode') == 'longrun' and not args.config_only and os.environ.get('NEMOTRON_SFT_SUPERVISED') != args.name:
        raise ValueError('longrun execution requires --supervise durable parent ownership')
    if plan.get('execution_mode') == 'longrun' and not args.config_only:
        validate_calibration(plan,args.manifest)
    output = ROOT/RESULTS/'runs'/args.name
    engine = load_module(ROOT/'results/pretrain_followup/engine_helper.py','nemotron_engine')
    locks = [ROOT/'logs/gpu_execution.lock',ROOT/'results/pretraining/nemochat_baby/training/gpu-all8.lock']
    if args.parent_lock_fd and len(args.parent_lock_fd) != 2:
        raise ValueError('pass both existing cooperative lock descriptors, never just one')
    with ExitStack() as stack:
        for index,path in enumerate(locks):
            handle = stack.enter_context(os.fdopen(os.dup(args.parent_lock_fd[index]),'r') if args.parent_lock_fd else path.open('r'))
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
            raise RuntimeError('GPU occupied; no container created')
        output.mkdir(parents=True,exist_ok=False)
        (output/'home').mkdir()
        (output/'data-preflight.json').write_text(json.dumps(verification,indent=2))
        (output/'runner-provenance.json').write_text(json.dumps(runner_provenance(args.config,args.manifest),indent=2))
        (output/'stage-plan.json').write_text(json.dumps(plan,indent=2))
        (output/'source-lineage.json').write_text(json.dumps(dict(source=native.SOURCE,source_sft_update=391,stage_step=0,
            checkpoint_integrity=config['checkpoint_integrity'],model_only=True,optimizer_scheduler_rng_cursor_reset=True),indent=2))
        (output/'command.json').write_text(json.dumps(payload,indent=2))
        key = ''
        if args.online:
            secure = load_module(ROOT/'results/bridge_wandb/wandb_preflight.py','nemotron_secure_wandb')
            key = secure.read_wandb_env(ROOT/'.env').get('WANDB_API_KEY','')
            if not key:
                raise ValueError('allowlisted runtime-only W&B credential missing')
            payload['Env'].append('WANDB_API_KEY='+key)
        cid = engine.request('POST','/containers/create?name=nemotron-sft-'+args.name,payload)['Id']
        (output/'container-id.json').write_text(json.dumps({'id':cid}))
        print('CONTAINER_ID',cid,flush=True)
        watchdog = False
        try:
            engine.request('POST','/containers/'+cid+'/start')
            engine.request('POST','/containers/'+cid+'/wait?condition=not-running',timeout=watchdog_seconds(plan))
        except (TimeoutError,socket.timeout):
            watchdog = True
            engine.request('POST','/containers/'+cid+'/kill')
            engine.request('POST','/containers/'+cid+'/wait?condition=not-running',timeout=120)
        finally:
            state = engine.request('GET','/containers/'+cid+'/json')
            connection = engine.Connection('localhost',timeout=120)
            connection.request('GET','/v1.52/containers/'+cid+'/logs?stdout=true&stderr=true&timestamps=true')
            raw = connection.getresponse().read()
            connection.close()
            chunks,index = [],0
            while index+8 <= len(raw):
                count = struct.unpack('>BBBBI',raw[index:index+8])[-1]
                index += 8
                chunks.append(raw[index:index+count])
                index += count
            text = b''.join(chunks).decode(errors='replace')
            if key:
                text = text.replace(key,'[REDACTED]')
            (output/'container.log').write_text(text)
            (output/'container-state.json').write_text(json.dumps(dict(id=cid,image=state['Image'],state=state['State'],watchdog_exceeded=watchdog),indent=2))
        if watchdog:
            raise RuntimeError('wall-time watchdog exceeded')
        if args.config_only:
            if state['State']['ExitCode'] != 0 or 'NATIVE_NEMOTRON_SFT_CONFIG_CONSTRUCTED' not in text:
                raise ValueError('native configuration gate failed')
            print('Native configuration constructed; weights/training unrun')
            return
        receipt = verify_run(output,plan,manifest)
        (output/'verification.json').write_text(json.dumps(receipt,indent=2))
        if plan.get('execution_mode') == 'calibration':
            rows = [json.loads(line) for line in (output/'rank-0-updates.jsonl').open()][5:]
            if len(rows) < 10:
                raise ValueError('calibration requires >=10 warm updates after discarding first five')
            times = [row['iteration_seconds'] for row in rows]
            calibration = dict(status='passed',global_batch_size=128,ranks=list(range(8)),
                sequence_length=plan['sequence_length'],context_parallel_size=plan.get('context_parallel_size',1),
                expert_model_parallel_size=plan.get('expert_model_parallel_size',8),
                manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                source_checkpoint=plan['source_checkpoint'],warm_updates=len(times),
                mean_iteration_seconds=sum(times)/len(times),cold_updates_excluded=5,
                verification=str((output/'verification.json').relative_to(ROOT)))
            (output/'calibration.json').write_text(json.dumps(calibration,indent=2))
        print(json.dumps(receipt))


if __name__ == '__main__':
    main()
