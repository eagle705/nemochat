"""CPU-only controls; synthetic receipts never constitute benchmark acceptance."""
import copy
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import unittest

HERE=Path(__file__).resolve().parent

def controller():
    p=HERE/'controller.py'
    assert p.exists(), 'exact pidfd controller not implemented'
    s=importlib.util.spec_from_file_location('ablation_auto_controller',p)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

class PidfdTests(unittest.TestCase):
    def setUp(self):
        self.c=controller()
        self.p=subprocess.Popen([sys.executable,'-u','-c',"import sys; print('READY',flush=True); sys.stdin.readline()"],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
        self.assertEqual(self.p.stdout.readline().strip(),'READY')
        self.identity=self.c.process_identity(self.p.pid)
    def tearDown(self):
        if self.p.poll() is None:self.p.terminate()
        self.p.wait();self.p.stdin.close();self.p.stdout.close()
    def test_real_normal_exit(self):
        self.p.stdin.write('\n');self.p.stdin.flush()
        self.assertEqual(self.c.wait_exact(self.identity,2),'exited')
    def test_real_zombie(self):
        self.p.stdin.write('\n');self.p.stdin.flush()
        os.waitid(os.P_PID,self.p.pid,os.WEXITED|os.WNOWAIT)
        self.assertEqual(self.c.wait_exact(self.identity,2),'exited')
    def test_stale_starttime(self):
        i=copy.deepcopy(self.identity);i['starttime']+=1
        with self.assertRaises(ValueError):self.c.wait_exact(i,.1)
    def test_wrong_argv(self):
        i=copy.deepcopy(self.identity);i['argv'][-1]+='wrong'
        with self.assertRaises(ValueError):self.c.wait_exact(i,.1)
    def test_wrong_uid(self):
        i=copy.deepcopy(self.identity);i['uid']+=1
        with self.assertRaises(ValueError):self.c.wait_exact(i,.1)
    def test_timeout_leaves_prerequisite_alive(self):
        with self.assertRaises(TimeoutError):self.c.wait_exact(self.identity,.03)
        self.assertIsNone(self.p.poll())
    def test_already_missing(self):
        self.p.stdin.write('\n');self.p.stdin.flush();self.p.wait()
        self.assertEqual(self.c.wait_exact(self.identity,.1),'already_absent')

class NativeGatesTests(unittest.TestCase):
    def setUp(self):
        self.c=controller()
        self.mapping=list(range(1000))
        self.rows=[dict(rank=0,update=i+1,logical_iteration=i,physical_iteration=i,grad_norm=1.,lr_used=0. if i==0 else 1e-5,skipped=False,metrics={'lm loss':2.,'total loss':2.},local_masked_tokens=65536,allrank_masked_tokens=524288,global_batch_size=128,sequence_length=4096) for i in range(1000)]
        self.complete=dict(rank=0,completed=1000,optimizer_calls=1000,positive_updates=999,nonfinite=0,skipped=0,restored_step=0,operational_resume=False,save_steps=[1000])
    def test_full_rank(self):
        self.assertTrue(hasattr(self.c,'validate_rank'),'native all-rank gate missing')
        self.c.validate_rank(0,self.rows,self.complete,self.mapping,lmonly=True)
    def test_bad_rank_refusals(self):
        self.assertTrue(hasattr(self.c,'validate_rank'),'native gate missing')
        for field,value in [('grad_norm',0),('grad_norm',float('nan')),('lr_used',-1),('skipped',True),('physical_iteration',222),('update',999),('rank',1),('allrank_masked_tokens',5)]:
            rows=copy.deepcopy(self.rows);rows[4][field]=value
            with self.subTest(field=field,value=value),self.assertRaises(ValueError):self.c.validate_rank(0,rows,self.complete,self.mapping,lmonly=True)
        with self.assertRaises(ValueError):self.c.validate_rank(0,self.rows[:-1],self.complete,self.mapping,lmonly=True)
        rows=copy.deepcopy(self.rows);rows[3]['metrics']['total loss']=1.
        with self.assertRaises(ValueError):self.c.validate_rank(0,rows,self.complete,self.mapping,lmonly=True)
    def test_sample_order_actual_digest(self):
        self.assertTrue(hasattr(self.c,'validate_samples'),'actual input gate missing')
        rows=[dict(rank=0,logical_iteration=i//16,physical_iteration=i//16,logical_micro=i,micro=i,sha256=str(i),masked_tokens=4096) for i in range(16000)]
        self.c.validate_samples(0,rows,rows,self.mapping)
        wrong=copy.deepcopy(rows);wrong[-1]['sha256']='different'
        with self.assertRaises(ValueError):self.c.validate_samples(0,wrong,rows,self.mapping)
        with self.assertRaises(ValueError):self.c.validate_samples(0,rows[:-1],rows,self.mapping)
    def test_engine_failure(self):
        self.assertTrue(hasattr(self.c,'clean_engine'),'Engine exit gate missing')
        good={'Id':'exact','State':{'Running':False,'ExitCode':0,'OOMKilled':False,'Error':''}}
        self.c.clean_engine(good,'exact')
        for path,val in [('Running',True),('ExitCode',1),('OOMKilled',True),('Error','failed')]:
            d=copy.deepcopy(good);d['State'][path]=val
            with self.assertRaises(ValueError):self.c.clean_engine(d,'exact')
        with self.assertRaises(ValueError):self.c.clean_engine(good,'other')

class CheckpointTests(unittest.TestCase):
    def test_actual_completed_C_checkpoint_schema(self):
        c=controller();self.assertTrue(hasattr(c,'validate_checkpoint'),'checkpoint gate missing')
        mapping=c.read(c.C/'final-dynamic-state.json')['committed_mapping']
        proof=c.validate_checkpoint(c.C,mapping,replay=False)
        self.assertEqual(proof['rank_acknowledgements'],8)
        self.assertEqual(proof['rng_records'],8)
    def test_hf_real_C_and_mutations(self):
        c=controller();self.assertTrue(hasattr(c,'validate_hf'),'HF reload gate missing')
        v=c.read(c.ROOT/'results/midtraining/hf-kd1000-conversion-01/validation_result.json')
        e=c.read(c.ROOT/'results/midtraining/hf-kd1000-conversion-01/export_result.json')
        c.validate_hf(v,e,c.MODELS['C'],c.C/'checkpoint/iter_0001000')
        for field,val in [('trust_remote_code',True),('full_model_reload',False),('all_reloaded_tensors_finite',False),('model',str(c.MODELS['A'])),('source_step',999),('device','cuda')]:
            x=copy.deepcopy(v);x[field]=val
            with self.subTest(field=field),self.assertRaises(ValueError):c.validate_hf(x,e,c.MODELS['C'],c.C/'checkpoint/iter_0001000')
        x=copy.deepcopy(v);x['loading_info']['missing_keys']=['bad']
        with self.assertRaises(ValueError):c.validate_hf(x,e,c.MODELS['C'],c.C/'checkpoint/iter_0001000')
        x=copy.deepcopy(v);x['cases'][1]['generation_cache']=False
        with self.assertRaises(ValueError):c.validate_hf(x,e,c.MODELS['C'],c.C/'checkpoint/iter_0001000')

def synthetic_reports(c):
    """400 generated answer rows/64 windows, not real model scores."""
    suite=c.read(c.EOUT/'suite.json');template=c.read(c.EOUT/'base-smoke01/report.json')
    reports={}
    for arm,model in c.MODELS.items():
        r=copy.deepcopy(template);r.update(status='complete',model=str(model),question_count=400,heldout_window_count=64,heldout_target_count=65536)
        r['questions']=[dict(id=q['id'],task=q['task'],gold=q['gold'],choice_logp_full_vocab_sum=[-1.]*4,gold_logp=-1.,gold_choice_conditional_logp=-1.3862943611198906,prediction=0,prediction_char_norm=0,acc=int(q['gold']==0),acc_norm=int(q['gold']==0)) for q in suite['questions']]
        for row,q in zip(r['questions'],suite['questions']):
            pn=max(range(4),key=lambda i:-1./len(q['choices'][i]));row.update(prediction_char_norm=pn,acc_norm=int(pn==q['gold']))
        r['heldout']=[dict(id=w['id'],source=w['source'],target_count=1024,nll_sum=2048.,nll_mean=2.) for w in suite['heldout']]
        r['heldout_full_vocab_nll']=2.;r['heldout_ppl']=__import__('math').exp(2.)
        r['metrics']={t:dict(n=len(qs),correct=sum(q['acc'] for q in qs),acc=sum(q['acc'] for q in qs)/len(qs),acc_norm=sum(q['acc_norm'] for q in qs)/len(qs)) for t in suite['counts'] for qs in [[q for q in r['questions'] if q['task']==t]]}
        reports[arm]=r
    return suite,reports

class ReportTests(unittest.TestCase):
    def test_complete_coverage_and_partial_refusals(self):
        c=controller();self.assertTrue(hasattr(c,'validate_report'),'full score gate missing')
        s,rs=synthetic_reports(c);c.validate_report(rs['A'],s,c.MODELS['A'])
        for field,val in [('status','passed_smoke'),('device','cuda'),('transformers','wrong'),('dtype','float32'),('question_count',399),('heldout_target_count',1024),('model',str(c.MODELS['B'])),('parameters_unchanged',False)]:
            r=copy.deepcopy(rs['A']);r[field]=val
            with self.subTest(field=field),self.assertRaises(ValueError):c.validate_report(r,s,c.MODELS['A'])
        for mutate in [lambda r:r['questions'].pop(),lambda r:r['questions'].__setitem__(1,r['questions'][0]),lambda r:r['heldout'].pop(),lambda r:r['heldout'][0].update(nll_mean=float('nan')),lambda r:r['questions'][0]['choice_logp_full_vocab_sum'].pop(),lambda r:r['questions'][0].update(gold=99),lambda r:r['heldout'][0].update(nll_sum=3.)]:
            r=copy.deepcopy(rs['A']);mutate(r)
            with self.assertRaises(ValueError):c.validate_report(r,s,c.MODELS['A'])
    def test_real_figure_and_summary_from_declared_fixture(self):
        import tempfile
        c=controller();p=HERE/'report.py';self.assertTrue(p.exists(),'figure/report implementation missing')
        spec=importlib.util.spec_from_file_location('ablation_figures',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        s,rs=synthetic_reports(c)
        with tempfile.TemporaryDirectory(dir='/home/yoon/.hermes/cache/scratch') as tmp:
            summary=m.render(rs,s,Path(tmp),fixture=True)
            self.assertEqual(summary['display_order'],['A','C','B'])
            self.assertEqual(len(summary['images']),2)
            from PIL import Image
            for path in summary['images']:
                with Image.open(path) as im:self.assertGreater(im.width,1000);self.assertGreater(im.height,500);self.assertEqual(im.format,'PNG')
            self.assertIn('CPU TEST FIXTURE',Path(tmp,'FINAL_REPORT.md').read_text())
            self.assertTrue(Path(tmp,'FINAL_REPORT.ko.md').exists())
            self.assertFalse(Path(tmp,'complete.json').exists())

class WorkflowTests(unittest.TestCase):
    def test_preview_no_state_change_exact_start(self):
        c=controller();self.assertTrue(hasattr(c,'preview'),'executable preview missing')
        before={p.name for p in c.OUT.glob('*')}
        p=c.preview(verify=False)
        self.assertEqual(p['unit'],'nemochat-offline2b-ablation-auto-01.service')
        self.assertEqual(p['B_identity']['starttime'],47312487)
        self.assertFalse(p['production_started'])
        self.assertEqual(before,{p.name for p in c.OUT.glob('*')})
        self.assertNotIn(str(c.MODELS['A']),str(p['export_command']))
        self.assertIn('--source-step',p['export_command'])
    def test_native_gate_current_partial_refuses(self):
        c=controller();self.assertTrue(hasattr(c,'native_gate'),'native full pipeline gate missing')
        with self.assertRaises((ValueError,FileNotFoundError)):c.native_gate()
    def test_frozen_suite_real_and_tamper(self):
        import tempfile
        c=controller();self.assertTrue(hasattr(c,'load_suite'),'suite authentication missing')
        s=c.load_suite();self.assertEqual(s['suite_sha256'],c.SUITE_SHA)
        with tempfile.TemporaryDirectory(dir='/home/yoon/.hermes/cache/scratch') as t:
            p=Path(t,'suite.json');s['protocol']='changed';c.write(p,s)
            with self.assertRaises(ValueError):c.load_suite(p)
    def test_engine_cleanup_never_prerequisites(self):
        c=controller();self.assertTrue(hasattr(c,'cleanup'),'owned cleanup missing')
        called=[]
        def request(method,path,data=None,**kwargs):
            called.append((method,path))
            raise RuntimeError('Docker GET: 404: absent')
        c.cleanup(request=request)
        self.assertTrue(called)
        self.assertFalse(any(c.BENGINE in p or c.AENGINE in p for _,p in called))
        self.assertTrue(all(c.UNIT.removesuffix('.service') in p for _,p in called))
    def test_owned_spec_no_gpu_4cpu_128gb(self):
        c=controller();self.assertTrue(hasattr(c,'guard_spec'),'owned CPU spec guard missing')
        spec=dict(HostConfig=dict(NetworkMode='none',NanoCpus=4000000000,Memory=137438953472),Labels={})
        c.guard_spec(spec)
        self.assertEqual(spec['Labels'][c.OWNER],c.UNIT)
        for field,val in [('DeviceRequests',[{'Driver':'nvidia'}]),('Memory',1),('NetworkMode','bridge'),('NanoCpus',8000000000)]:
            s=copy.deepcopy(spec);s['HostConfig'][field]=val
            with self.subTest(field=field),self.assertRaises(ValueError):c.guard_spec(s)
    def test_stage_order_and_failure_durable(self):
        import tempfile
        c=controller();self.assertTrue(hasattr(c,'pipeline'),'tested pipeline missing')
        events=[]
        class Ops:
            def wait_b(self):events.append('wait_b')
            def native(self):events.append('native')
            def start_eval(self,arm):events.append('start_'+arm);return arm
            def export_b(self):events.append('export_b')
            def wait_a(self):events.append('wait_a')
            def finish_eval(self,arm,job):events.append('finish_'+arm)
            def final(self):events.append('final');raise ValueError('fixture deliberately cannot accept real benchmark')
            def cleanup(self):events.append('cleanup')
        with tempfile.TemporaryDirectory(dir='/home/yoon/.hermes/cache/scratch') as t:
            with self.assertRaises(ValueError):c.pipeline(Path(t),Ops())
            self.assertTrue(Path(t,'failure.json').exists());self.assertFalse(Path(t,'complete.json').exists())
        self.assertEqual(events,['wait_b','native','start_C','export_b','start_B','wait_a','finish_C','finish_B','final','cleanup'])

    def test_real_launcher_adapter_with_fixture_engine(self):
        import tempfile
        c=controller();self.assertTrue(hasattr(c,'cpu_job'),'foreground launcher adapter missing')
        class Client:
            spec=None
            def request(self,method,path,data=None,**kwargs):
                if path.startswith('/containers/create'):
                    self.spec=data;return {'Id':'fixture-owned-id'}
                if path.endswith('/start'):return None
                if path.endswith('/json'):return dict(Id='fixture-owned-id',Name='/'+c.owned_name('eval-C'),Config={'Labels':self.spec['Labels']},HostConfig=self.spec['HostConfig'],State={'Running':False,'ExitCode':0,'OOMKilled':False,'Error':''})
                raise AssertionError(path)
            class Connection:
                def __init__(self,*a,**kw):pass
                def request(self,*a,**kw):pass
                def getresponse(self):return self
                def read(self):return b''
                def close(self):pass
        with tempfile.TemporaryDirectory(dir='/home/yoon/.hermes/cache/scratch') as t:
            c.OUT=Path(t);c.verify_freeze=lambda:None;cl=Client()
            proof=c.cpu_job('eval-C',['python','fixture-not-executed'],client=cl)
            self.assertEqual(proof['State']['ExitCode'],0)
            self.assertEqual(cl.spec['Labels'][c.OWNER],c.UNIT)
            self.assertEqual(c.read(Path(t,'eval-C.owned.json'))['id'],'fixture-owned-id')
    def test_start_script_defaults_preview(self):
        c=controller();p=HERE/'start.sh';self.assertTrue(p.exists(),'safe start script missing')
        text=p.read_text();self.assertIn('RuntimeMaxSec=23h',text);self.assertIn('Restart=no',text)
        self.assertIn('ExecStopPost=',text);self.assertNotIn('User=',text)
    def test_finalizer_rejects_partial_and_fixture(self):
        import tempfile
        c=controller();self.assertTrue(hasattr(c,'final_reports'),'verified finalizer missing')
        s,rs=synthetic_reports(c)
        with tempfile.TemporaryDirectory(dir='/home/yoon/.hermes/cache/scratch') as t:
            paths={a:Path(t,a,'report.json') for a in rs}
            for a,p in paths.items():c.write(p,rs[a])
            # Declared fixture contains no real per-question execution receipts.
            with self.assertRaises((ValueError,FileNotFoundError)):c.final_reports(paths,s,Path(t))
            self.assertFalse(Path(t,'complete.json').exists())

    def test_terminal_marker_after_cleanup_and_no_false_success(self):
        import tempfile
        c=controller();events=[]
        class Ops:
            def wait_b(self):pass
            def native(self):pass
            def start_eval(self,a):return a
            def export_b(self):pass
            def wait_a(self):pass
            def finish_eval(self,*args):pass
            def final(self):return {'status':'verified_all_three_complete','all_three_complete':True,'images':[], 'CPU_control_fixture':True}
            def cleanup(self):events.append('cleanup')
        # CPU control in scratch: this is not a production completion receipt.
        with tempfile.TemporaryDirectory(dir='/home/yoon/.hermes/cache/scratch') as t:
            c.pipeline(Path(t),Ops())
            self.assertTrue(Path(t,'complete.json').exists(),'success marker never published')
            self.assertEqual(events,['cleanup'])
        class Bad(Ops):
            def final(self):return {'status':'partial'}
        with tempfile.TemporaryDirectory(dir='/home/yoon/.hermes/cache/scratch') as t:
            with self.assertRaises(ValueError):c.pipeline(Path(t),Bad())
            self.assertTrue(Path(t,'failure.json').exists());self.assertFalse(Path(t,'complete.json').exists())

    def test_authenticated_source_freeze_mutation(self):
        import tempfile
        c=controller()
        with tempfile.TemporaryDirectory(dir='/home/yoon/.hermes/cache/scratch') as t:
            c.OUT=Path(t);f=Path(t,'allowed.py');f.write_text('approved')
            c.allowlist=lambda:[f]
            c.write(c.OUT/'source-freeze.json',c.source_hashes());c.verify_freeze()
            f.write_text('modified')
            with self.assertRaises(ValueError):c.verify_freeze()
    def test_question_conditional_probability_not_just_finite(self):
        c=controller();s,rs=synthetic_reports(c);rs['A']['questions'][0]['gold_choice_conditional_logp']=-200
        with self.assertRaises(ValueError):c.validate_report(rs['A'],s,c.MODELS['A'])
    def test_exact_A_blocking_wait_not_restart(self):
        import tempfile
        c=controller();s,rs=synthetic_reports(c);calls=[]
        creation=c.read(c.EOUT/'ablation-eval-A-full01.create.json')
        class Client:
            first=True
            def request(self,method,path,data=None,**kwargs):
                calls.append((method,path,kwargs))
                if '/wait?' in path:return {'StatusCode':0}
                running=self.first;self.first=False
                return {'Id':c.AENGINE,'Config':{'Cmd':creation['spec']['Cmd']},'State':{'Running':running,'OOMKilled':False,'ExitCode':0,'Error':''}}
        client=Client();c.engine=lambda:client
        with tempfile.TemporaryDirectory(dir='/home/yoon/.hermes/cache/scratch') as t:
            c.OUT=Path(t);ops=c.ProductionOps();ops.paths['A']=Path(t,'A/report.json');c.write(ops.paths['A'],rs['A'])
            try:ops.wait_a()
            finally:ops.pool.shutdown()
        self.assertEqual([x[1] for x in calls if x[0]=='POST'],[f'/containers/{c.AENGINE}/wait?condition=not-running'])
        self.assertGreater(calls[1][2]['timeout'],20000)
    def test_partial_native_transaction_and_shards_rejected(self):
        import tempfile
        c=controller();mapping=c.read(c.C/'final-dynamic-state.json')['committed_mapping'];source=c.C/'checkpoint/iter_0001000'
        txn=c.read(c.C/'checkpoint-step1000-transaction.json')
        with tempfile.TemporaryDirectory(dir='/home/yoon/.hermes/cache/scratch') as t:
            run=Path(t);cp=run/'checkpoint/iter_0001000';cp.mkdir(parents=True)
            for f in source.iterdir():(cp/f.name).symlink_to(f)
            for mutate in [lambda x:x.update(save_rng=False),lambda x:x['ranks'].pop(),lambda x:x['ranks'][0].update(step=999),lambda x:x['binding'].update(consumed_samples=1)]:
                bad=copy.deepcopy(txn);mutate(bad)
                c.write(run/'checkpoint-step1000-transaction.json',bad);c.write(run/'student-checkpoint-step1000-transaction.json',bad)
                with self.assertRaises(ValueError):c.validate_checkpoint(run,mapping,False)
            c.write(run/'checkpoint-step1000-transaction.json',txn);c.write(run/'student-checkpoint-step1000-transaction.json',txn)
            (cp/'__7_0.distcp').unlink()
            with self.assertRaises(ValueError):c.validate_checkpoint(run,mapping,False)

    def test_A_failed_before_downstream_release(self):
        c=controller();self.assertTrue(hasattr(c,'a_health'),'pre-release A failure check missing')
        creation=c.read(c.EOUT/'ablation-eval-A-full01.create.json')
        class Client:
            def request(self,*args,**kwargs):return {'Id':c.AENGINE,'Config':{'Cmd':creation['spec']['Cmd']},'State':{'Running':False,'ExitCode':124,'OOMKilled':False,'Error':''}}
        c.engine=lambda:Client()
        with self.assertRaises(ValueError):c.a_health()
    def test_cleanup_failure_never_publishes_success(self):
        import tempfile
        c=controller()
        class Ops:
            def wait_b(self):pass
            def native(self):pass
            def start_eval(self,a):return a
            def export_b(self):pass
            def wait_a(self):pass
            def finish_eval(self,*args):pass
            def final(self):return {'status':'verified_all_three_complete','all_three_complete':True}
            def cleanup(self):raise RuntimeError('declared cleanup fixture failure')
        with tempfile.TemporaryDirectory(dir='/home/yoon/.hermes/cache/scratch') as t:
            with self.assertRaises(RuntimeError):c.pipeline(Path(t),Ops())
            self.assertTrue(Path(t,'failure.json').exists());self.assertFalse(Path(t,'complete.json').exists())

if __name__=='__main__':unittest.main()
