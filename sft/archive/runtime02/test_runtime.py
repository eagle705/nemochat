"""CPU fixtures only. No real dataset or GPU work is accepted here."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))

class RuntimeTests(unittest.TestCase):
    def test_missing_readiness_refuses_before_launch(self):
        import controller as c
        with tempfile.TemporaryDirectory(dir='/home/yoon/.hermes/cache/scratch') as t:
            with self.assertRaises((FileNotFoundError,PermissionError)):
                c.validate_release(Path(t)/'missing.json')

    def test_budget_uses_allrank_warm9_16_and_full_sampler(self):
        import controller as c
        rows=[[dict(update=i,iteration_seconds=10+r,loss=1.,grad_norm=2.,lr=1e-5,native_optimizer=True,finite_positive_gradient=True) for i in range(1,17)] for r in range(8)]
        b=c.timing_budget(rows,c.TRAIN_STOP-3600)
        self.assertEqual(b['warm_updates'],list(range(9,17)))
        self.assertEqual(b['warm_mean_seconds'],17)
        self.assertEqual(b['native_full_epoch_updates'],34135)
        self.assertEqual(b['epochs'],1)
        self.assertLess(b['execution_update_cap'],34135)
        self.assertEqual(b['training_stop_timestamp'],c.TRAIN_STOP)
        with self.assertRaises(PermissionError):c.timing_budget(rows,c.TRAIN_STOP-1)
        rows[7][-1]['grad_norm']=0
        with self.assertRaises(PermissionError):c.timing_budget(rows,c.TRAIN_STOP-3600)

    def test_native_plan_keeps_full_capacity_for_calibration_and_main(self):
        import controller as c
        manifest={'train_dialogues':4369280,'heldout_dialogues':45090}
        for stage in ('calibration','main'):
            p=c.stage_config(stage,manifest,c.TRAIN_STOP-4000,budget={'execution_update_cap':100,'conservative_iteration_seconds':25})
            self.assertEqual(p['max_updates'],34135)
            self.assertEqual(p['epochs'],1)
            self.assertFalse(p['allow_repetition'])
            self.assertEqual(p['source_checkpoint'],c.SOURCE)
            self.assertEqual(p['execution_update_cap'],16 if stage=='calibration' else 100)
            self.assertEqual(p['validation_max_dialogues'],128)
        with self.assertRaises(PermissionError):c.stage_config('main',manifest,c.TRAIN_STOP-4000)

    def test_engine_owner_and_service_preview_are_exact(self):
        import controller as c
        v=c.preview('calibration')
        argv=v['supervisor_argv']
        self.assertTrue(any('ExecStopPost=' in x for x in argv))
        self.assertFalse(any('User=' in x or 'Group=' in x for x in argv))
        self.assertNotIn('--execute',argv[:2])
        a={'Id':'exact','Name':'/nemochat-sft-after-kd-runtime02-calibration','Config':{'Labels':{c.OWNER:'calibration','image.label':'retained'}}}
        c.check_owned(a,'exact','calibration')
        a['Id']='other'
        with self.assertRaises(PermissionError):c.check_owned(a,'exact','calibration')
        p=c.container_payload('calibration',Path('/output/c.yaml'),1000)
        self.assertEqual(p['HostConfig']['NetworkMode'],'none')
        self.assertEqual(p['HostConfig']['DeviceRequests'][0]['DeviceIDs'],list(map(str,range(8))))
        self.assertEqual(p['HostConfig']['RestartPolicy']['Name'],'no')
        self.assertTrue(p['HostConfig']['Binds'][0].endswith(':ro'))
        self.assertIn('WANDB_MODE=disabled',p['Env'])

    def test_native_wrapper_builds_full_sampler_and_early_save(self):
        import native as n
        from types import SimpleNamespace as S
        cfg=S(train=S(train_iters=34135),checkpoint=S(save_interval=250),validation=S(eval_iters=16))
        plan={'updates':34135,'execution_update_cap':16,'runtime_stage':'calibration'}
        n.runtime_config(cfg,plan)
        self.assertEqual(cfg.train.train_iters,34135)
        self.assertEqual(cfg.train.exit_interval,16)
        self.assertEqual(cfg.checkpoint.save_interval,16)
        self.assertEqual(cfg.validation.eval_iters,16)
        plan['updates']=16
        with self.assertRaises(PermissionError):n.runtime_config(cfg,plan)

    def test_native_acceptance_rejects_missing_finalsave_and_checks_all8(self):
        import controller as c
        with tempfile.TemporaryDirectory(dir='/home/yoon/.hermes/cache/scratch') as t:
            root=Path(t)
            state={'Running':False,'ExitCode':0,'OOMKilled':False,'Error':''}
            with self.assertRaises((PermissionError,FileNotFoundError)):
                c.accept_native(root,state,{'execution_update_cap':16},'calibration')
            for r in range(8):
                def put(name,value): (root/f'rank-{r}-{name}.json').write_text(json.dumps(value))
                rows=[dict(rank=r,update=i,loss=1.,grad_norm=1.,lr=1e-5,iteration_seconds=10.,native_optimizer=True,finite_positive_gradient=True) for i in range(1,17)]
                (root/f'rank-{r}-updates.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
                (root/f'rank-{r}-heldout.jsonl').write_text(json.dumps({'update':16,'heldout_ce':1.})+'\n')
                put('load',{'source':c.SOURCE,'stage_step':0,'model_only':True,'optimizer_scheduler_rng_cursor_reset':True})
                put('entrypoint',{'updates':16,'native_exit_code':0})
                put('hf-import',{'source_key_coverage':'all_index_keys_have_native_mappings','source_tensor_keys':6289,'native_embedding_transform_calls':1,'native_embedding_shape':[131072,1024]})
                put('destination-import',{'destination_mapping_complete':True,'destination_required_tensors':3,'destination_covered_tensors':3,'native_transform_outputs':3})
                put('save',{'save_steps':[16],'save_optim':True,'save_rng':True,'native_synchronous_save_completed':True})
            accepted=c.accept_native(root,state,{'execution_update_cap':16},'calibration')
            self.assertEqual(accepted['updates'],16)
            from unittest.mock import patch
            with patch.object(c.time,'time',return_value=c.DEADLINE+1):
                with self.assertRaises(PermissionError):c.accept_native(root,state,{'execution_update_cap':16},'main')
            dst=root/'rank-7-destination-import.json';saved=dst.read_text();dst.unlink()
            with self.assertRaises((PermissionError,FileNotFoundError)):c.accept_native(root,state,{'execution_update_cap':16},'calibration')
            dst.write_text(saved)
            save=root/'rank-7-save.json';v=json.loads(save.read_text());v['save_steps']=[];save.write_text(json.dumps(v))
            with self.assertRaises(PermissionError):c.accept_native(root,state,{'execution_update_cap':16},'calibration')

    def test_owned_cleanup_stops_only_recorded_runtime_container(self):
        import controller as c
        class Engine:
            def __init__(self): self.events=[];self.running=True
            def request(self,method,path,data=None,timeout=120):
                self.events.append((method,path))
                if method=='POST':self.running=False;return
                return {'Id':'exact','Name':'/nemochat-sft-after-kd-runtime02-calibration','Config':{'Labels':{c.OWNER:'calibration'}},'State':{'Running':self.running,'ExitCode':0}}
        old=c.OUT
        try:
            with tempfile.TemporaryDirectory(dir='/home/yoon/.hermes/cache/scratch') as t:
                c.OUT=Path(t);e=Engine()
                c.write(c.OUT/'calibration-container.json',{'id':'exact'})
                c.cleanup('calibration',e)
                self.assertEqual([p for m,p in e.events if m=='POST'],['/containers/exact/stop?t=60'])
                self.assertFalse(c.read(c.OUT/'calibration-cleanup.json')['calibration']['Running'])
        finally:c.OUT=old

    def test_real_cli_preview_and_start_missing_release_are_inert(self):
        preview=subprocess.run([sys.executable,str(HERE/'controller.py'),'--preview','calibration'],text=True,capture_output=True)
        self.assertEqual(preview.returncode,0,preview.stderr)
        self.assertEqual(json.loads(preview.stdout)['gpu_launches'],0)
        with tempfile.TemporaryDirectory(dir='/home/yoon/.hermes/cache/scratch') as t:
            refused=subprocess.run([sys.executable,str(HERE/'controller.py'),'--start','calibration','--release',str(Path(t)/'absent')],text=True,capture_output=True)
            self.assertEqual(refused.returncode,78,refused.stderr)
            self.assertIn('FileNotFoundError',refused.stderr)

    def test_real_pidfd_exit_timeout_and_stale_identity(self):
        import controller as c
        import select
        m=c.load_module('runtime02_test_pidfd',c.ROOT/'sft/training/controls.py')
        child=subprocess.Popen([sys.executable,'-u','-c','import sys;print("READY",flush=True);sys.stdin.readline()'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
        try:
            self.assertEqual(child.stdout.readline().strip(),'READY')
            ident=m.process_identity(child.pid)
            with self.assertRaises(TimeoutError):m.wait_exact(ident,.01)
            self.assertIsNone(child.poll())
            with self.assertRaises(ValueError):m.wait_exact(dict(ident,starttime=ident['starttime']+1),1)
            child.stdin.write('exit\n');child.stdin.flush()
            m.wait_exact(ident,5)
            self.assertEqual(child.wait(),0)
        finally:
            if child.poll() is None:child.kill();child.wait()
            child.stdin.close();child.stdout.close()

    def test_pinned_sdk_root_read_permissions_are_explicit(self):
        import controller as c
        p=c.container_payload('calibration',Path('/output/c.yaml'),1000)
        self.assertEqual(p['User'],'0:0')
        self.assertEqual(p['HostConfig']['CapAdd'],['DAC_OVERRIDE'])

    def test_chained_main_is_in_calibration_cleanup_scope(self):
        import controller as c
        class Engine:
            def __init__(self):self.running=True;self.stops=[]
            def request(self,method,path,data=None,timeout=120):
                if method=='POST':self.stops.append(path);self.running=False;return
                return {'Id':'main-id','Name':'/nemochat-sft-after-kd-runtime02-main','Config':{'Labels':{c.OWNER:'main'}},'State':{'Running':self.running,'ExitCode':0}}
        old=c.OUT
        try:
            with tempfile.TemporaryDirectory(dir='/home/yoon/.hermes/cache/scratch') as t:
                c.OUT=Path(t);e=Engine()
                c.write(c.OUT/'main-container.json',{'id':'main-id'})
                c.write(c.OUT/'calibration-owns-main.json',{'owner':c.unit('calibration')})
                c.cleanup('calibration',e)
                self.assertEqual(e.stops,['/containers/main-id/stop?t=60'])
        finally:c.OUT=old

    def test_fixture_release_binds_all7_audits_and_rejects_direct_only(self):
        import controller as c
        from unittest.mock import patch
        with tempfile.TemporaryDirectory(dir='/home/yoon/.hermes/cache/scratch') as t:
            root=Path(t);out=root/'out';out.mkdir();data=root/'data';data.mkdir()
            files={}
            for i in range(7):
                p=data/f'fixture-{i}.npy';p.write_bytes(b'CPU_FIXTURE_NOT_PRODUCTION')
                files[p.name]={'bytes':p.stat().st_size,'sha256':c.sha(p)}
            manifest=data/'manifest.json';c.write(manifest,{'files':files})
            ready=root/'READY.json';c.write(ready,{'status':'accepted_existing_full03_data'})
            receipt=root/'audit.json';c.write(receipt,{'CPU_fixture_only':True})
            freeze=out/'source-freeze.json';c.write(freeze,{})
            benchmark=root/'results/midtraining/offline2b-ablation-gpu-01/parent-final-acceptance.json';c.write(benchmark,{'CPU_fixture_only':True})
            d=dict(ready_sha256=c.sha(ready),manifest_sha256=c.sha(manifest),files={n:dict(expected_sha256=v['sha256'],observed_sha256=v['sha256'],bytes=v['bytes']) for n,v in files.items()})
            for k in ('buffered_mmap_consumer_accepted','new_sft_reserved_id_audit_passed','fresh_c_reserved_id_audit_passed','source_provenance_accepted','readonly_data_release'):d[k]=True
            for k in ('audit_receipt','provenance_receipt'):d[k]={'path':str(receipt),'sha256':c.sha(receipt)}
            r=dict(status='parent_accepted_runtime02',source_hf=c.SOURCE,source_freeze_sha256=c.sha(freeze),ready_path=str(ready),ready_sha256=c.sha(ready),data_contract=d,benchmark_acceptance_sha256=c.sha(benchmark),checkpoint_integrity='not_checked_explicit_user_waiver')
            release=root/'release.json';c.write(release,r)
            with patch.multiple(c,ROOT=root,OUT=out,MANIFEST=manifest,MANIFEST_SHA=c.sha(manifest)),patch.object(c,'source_paths',return_value=[]):
                self.assertEqual(c.validate_release(release)['status'],'parent_accepted_runtime02')
                d['buffered_mmap_consumer_accepted']=False;c.write(release,r)
                with self.assertRaises(PermissionError):c.validate_release(release)
                d['buffered_mmap_consumer_accepted']=True
                d['files']['fixture-6.npy']['observed_sha256']='0'*64;c.write(release,r)
                with self.assertRaises(PermissionError):c.validate_release(release)
                d['files']['fixture-6.npy']['observed_sha256']=files['fixture-6.npy']['sha256']
                r['source_hf']='BASE';c.write(release,r)
                with self.assertRaises(PermissionError):c.validate_release(release)

    def test_native_destination_coverage_refuses_random_missing_parameters(self):
        import import_guard as g
        from types import SimpleNamespace as S
        a,b=object(),object()
        class Model:
            def named_parameters(self):return [('a',a),('b',b)]
        tasks=[S(param_weight=a,megatron_module=object())]
        with self.assertRaises(PermissionError):g.destination_coverage([Model()],tasks,lambda m:[])
        tasks.append(S(param_weight=b,megatron_module=object()))
        r=g.destination_coverage([Model()],tasks,lambda m:[])
        self.assertEqual(r['destination_required_tensors'],2)
        self.assertEqual(r['destination_covered_tensors'],2)

    def test_start_shell_is_inert_by_default_and_forwards_calibration(self):
        p=subprocess.run(['bash',str(HERE/'start.sh')],text=True,capture_output=True)
        self.assertEqual(p.returncode,0,p.stderr)
        self.assertEqual(json.loads(p.stdout)['status'],'INERT_PARENT_REVIEW_REQUIRED')
        with tempfile.TemporaryDirectory(dir='/home/yoon/.hermes/cache/scratch') as t:
            p=subprocess.run(['bash',str(HERE/'start.sh'),'--calibration','--release',str(Path(t)/'absent')],text=True,capture_output=True)
            self.assertEqual(p.returncode,78,p.stderr)

    def test_metadata_gate_preserves_migrated_official_tokenizer(self):
        import native as n
        m={'tokenizer_revision':'revision','tokenizer_path':'path'}
        t={'status':'passed','vocab_size':131072,'allowed_changed_ids':list(range(18,30)),'new_revision':'revision','tokenizer_path':'path'}
        n.verify_runtime_metadata(m,t)
        t['vocab_size']=42
        with self.assertRaises(PermissionError):n.verify_runtime_metadata(m,t)

    def test_cleanup_reconciles_create_before_identity_receipt(self):
        import controller as c
        class Engine:
            def __init__(self):self.running=True
            def request(self,method,path,data=None,timeout=120):
                if method=='POST':self.running=False;return
                return {'Id':'recovered','Name':'/nemochat-sft-after-kd-runtime02-calibration','Config':{'Labels':{c.OWNER:'calibration'}},'State':{'Running':self.running,'ExitCode':0}}
        old=c.OUT
        try:
            with tempfile.TemporaryDirectory(dir='/home/yoon/.hermes/cache/scratch') as t:
                c.OUT=Path(t);c.write(c.OUT/'calibration-execution-claim.json',{'CPU_fixture_only':True})
                # Reload absent is represented with an exact Engine 404.
                e=Engine();original=e.request
                def request(method,path,data=None,timeout=120):
                    if 'runtime02-reload' in path:raise RuntimeError('Docker GET /containers/name/json: 404: no container')
                    return original(method,path,data,timeout)
                e.request=request
                c.cleanup('calibration',e)
                self.assertFalse(e.running)
                self.assertEqual(c.read(c.OUT/'calibration-container.json')['id'],'recovered')
        finally:c.OUT=old

    def test_service_runtime_reserves_bounded_engine_cleanup_before_noon(self):
        import controller as c
        import time
        a=c.supervisor_argv('main',Path('/parent/release.json'))
        cap=int(next(x.split('=',2)[2] for x in a if x.startswith('--property=RuntimeMaxSec=')))
        self.assertLessEqual(cap,int(c.DEADLINE-time.time()-110))

    def test_manifest_sha_literal_is_exact_64_hex_and_matches_metadata(self):
        import controller as c
        self.assertEqual(len(c.MANIFEST_SHA),64)
        self.assertEqual(c.MANIFEST_SHA,'78ab81fbbd9c0b13c55392b7d87c9e7035ad75778abad0dd3795d64121125c21')
        self.assertEqual(c.sha(c.MANIFEST),c.MANIFEST_SHA)

if __name__ == '__main__':
    unittest.main()
