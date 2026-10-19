"""Fresh-only CPU ablation controller. Default command is inert preview."""
import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
HERE=ROOT/'midtraining/evaluation/ablation_auto'
OUT=ROOT/'results/midtraining/offline2b-ablation-auto-01'
B=ROOT/'results/midtraining/offline2b-lmonly-1000-02'
C=ROOT/'results/midtraining/offline2b-prefixtrain-1000-01'
EVAL=ROOT/'midtraining/evaluation/ablation_eval'
EOUT=ROOT/'results/midtraining/offline2b-ablation-eval-01'
EXPORT_CODE=ROOT/'midtraining/publication/export_kd1000'
BMODEL=ROOT/'checkpoints/exports/nemochat_mid_lmonly_hf_bf16_02'
MODELS={'A':ROOT/'checkpoints/exports/nemochat_baby_base_hf_bf16_02','C':ROOT/'checkpoints/exports/nemochat_mid_super35_distill_hf_bf16_01','B':BMODEL}
UNIT='nemochat-offline2b-ablation-auto-01.service'
BUNIT='nemochat-offline2b-lmonly-1000-02.service'
BENGINE='c3779c8241de4082eb37eb0c407f574b9505e6d2c366f55c72a7c4b5b5b1eeda'
AENGINE='e1404d98ba226a85000984e8a7dee2dc758a797c997fbb46ea9ec1aaa845bfa2'
BIDENTITY={'pid':4083833,'uid':1001,'starttime':47312487,'argv':['/raid/yoon/nemochat/.venv/bin/python','-u',str(ROOT/'midtraining/training/offline_kd/ablation_lm_only/ordinary02/supervisor.py')]}
SUITE_SHA='07609a0cf53e4a63d555a3e50938adfed1cb05471d02b37fb93461e7879af19e'

def require(condition,message):
    if not condition:raise ValueError(message)

def read(path):
    return json.loads(Path(path).read_text())

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.partial')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n');tmp.replace(path)

def process_identity(pid):
    p=Path(f'/proc/{pid}');stat=p.joinpath('stat').read_text().rsplit(')',1)[1].split()
    return {'pid':pid,'uid':p.stat().st_uid,'starttime':int(stat[19]),'argv':[s.decode() for s in p.joinpath('cmdline').read_bytes().split(b'\0') if s],'state':stat[0]}

def check_identity(actual,expected):
    require(all(actual[k]==expected[k] for k in ['pid','uid','starttime']),'process identity/starttime mismatch')
    if actual['state']!='Z':require(actual['argv']==expected['argv'],'process argv mismatch')

def wait_exact(expected,timeout):
    try:actual=process_identity(expected['pid'])
    except FileNotFoundError:return 'already_absent'
    check_identity(actual,expected)
    try:fd=os.pidfd_open(expected['pid'])
    except ProcessLookupError:return 'already_absent'
    try:
        try:check_identity(process_identity(expected['pid']),expected)
        except FileNotFoundError:return 'already_absent'
        poll=select.poll();poll.register(fd,select.POLLIN)
        if not poll.poll(max(1,math.ceil(timeout*1000))):raise TimeoutError('exact prerequisite pidfd deadline')
        return 'exited'
    finally:os.close(fd)

def clean_engine(inspect,cid):
    s=inspect['State']
    require(inspect['Id']==cid and not s['Running'] and s['ExitCode']==0 and not s['OOMKilled'] and not s.get('Error'),'unclean exact Engine completion')

def validate_rank(rank,rows,complete,mapping,lmonly=False):
    require(len(mapping)==len(rows)==1000,'exactly 1000 native calls required')
    require(all(complete[k]==v for k,v in dict(rank=rank,completed=1000,optimizer_calls=1000,positive_updates=999,skipped=0,nonfinite=0,restored_step=0,operational_resume=False).items()),'native rank complete mismatch')
    require(1000 in complete['save_steps'],'final save missing')
    for i,r in enumerate(rows):
        require(all(r[k]==v for k,v in dict(rank=rank,update=i+1,logical_iteration=i,physical_iteration=mapping[i],global_batch_size=128,sequence_length=4096,local_masked_tokens=65536,allrank_masked_tokens=524288).items()),'native update/input/shape mismatch')
        require(not r['skipped'] and r['grad_norm']>0 and r['lr_used']>=0,'skipped/nonpositive gradient/negative LR')
        require(all(math.isfinite(x) for x in [r['grad_norm'],r['lr_used'],*r['metrics'].values()]),'nonfinite native metric')
        if lmonly:
            require(abs(r['metrics']['total loss']-r['metrics']['lm loss'])<1e-6 and 'logits distillation loss' not in r['metrics'],'not ordinary LM-only objective')
    require(sum(r['lr_used']>0 for r in rows)==999 and sum(r['lr_used']==0 for r in rows)==1,'positive/zero LR count mismatch')

def validate_samples(rank,b,c,mapping):
    require(len(b)==len(c)==16000,'exact 1000 x 16 actual samples required per rank')
    for i,(x,y) in enumerate(zip(b,c)):
        logical=i//16;physical=mapping[logical]
        require(x['rank']==y['rank']==rank and x['logical_iteration']==y['logical_iteration']==logical and x['physical_iteration']==y['physical_iteration']==physical and x['logical_micro']==y['logical_micro']==i,'actual sample order mismatch')
        require(x['sha256']==y['sha256'] and x['masked_tokens']==y['masked_tokens']==4096,'actual input digest/token mismatch')

def rows(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]

def validate_checkpoint(run,mapping,replay):
    import pickle
    cp=run/'checkpoint/iter_0001000'
    for name in ['.metadata','train_state.pt','run_config.yaml','metadata.json',*[f'__{r}_0.distcp' for r in range(8)]]:
        require((cp/name).is_file() and (cp/name).stat().st_size>0,'incomplete native checkpoint inventory '+name)
    require(len(list(cp.glob('*.distcp')))==8,'not exactly 8 checkpoint shards')
    txn=read(run/'student-checkpoint-step1000-transaction.json')
    require(txn==read(run/'checkpoint-step1000-transaction.json'),'transaction copies disagree')
    require(all(txn[k] is True for k in ['native_synchronous_save_completed','save_optim','save_rng']),'not full-state synchronous save')
    require(txn['checkpoint_integrity']=='not_checked_explicit_user_waiver','checkpoint byte policy changed')
    binding=txn['binding'];ack=txn['ranks']
    require(len(ack)==8 and {r['rank'] for r in ack}==set(range(8)),'all-eight save acknowledgements missing')
    require(all(r['step']==1000 and r['binding']==binding for r in ack),'save binding mismatch')
    state=binding['replay_state' if replay else 'dynamic_state']
    require(state['committed_mapping']==mapping and binding['consumed_samples']==128000,'save committed input mapping mismatch')
    if replay:
        require(binding['next_iteration']==1000 and state['completed']==1000 and state['replay_only'] is True,'invalid final replay binding')
        require(binding['replay_manifest_sha256']==hashlib.sha256((run/'replay-manifest.json').read_bytes()).hexdigest(),'replay manifest binding changed')
        require(read(run/'final-replay-state.json')==state,'final replay/save disagreement')
    # Trusted project-produced metadata only: never read/hash tensor payloads.
    metadata=pickle.loads((cp/'.metadata').read_bytes());keys=set(metadata.state_dict_metadata)
    require('embedding.word_embeddings.weight' in keys,'native unwrapped model schema changed')
    require(any(k.startswith('optimizer.state.exp_avg.') for k in keys) and any(k.startswith('optimizer.state.exp_avg_sq.') for k in keys),'optimizer moments absent')
    rng=[k for k in keys if k.startswith('rng_state/')]
    require(len(rng)==8,'not eight RNG metadata records')
    return {'checkpoint':str(cp),'rank_acknowledgements':len(ack),'rng_records':len(rng),'metadata_records':len(keys),'checkpoint_integrity':'not_checked_explicit_user_waiver'}

def validate_hf(v,e,model,source):
    require(v['status']=='passed' and v['model']==str(model) and v['source']==str(source) and v['source_step']==1000 and v['model_prefix']=='','HF validation source/step mismatch')
    require(v['trust_remote_code'] is False and v['device']=='cpu' and v['cuda_visible'] is False and v['versions']['transformers']=='5.12.1','HF backend mismatch')
    require(all(v[k] is True for k in ['official_transformers','full_model_reload','all_reloaded_tensors_finite','tokenizer_bytes_exact_match','tokenizer_vocab_exact_match']),'incomplete official HF reload')
    require(all(not v['loading_info'].get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']),'HF reload key errors')
    require(v['class']=='transformers.models.nemotron_h.modeling_nemotron_h.NemotronHForCausalLM','wrong official class')
    require(len(v['cases'])==2 and [r['prompt'] for r in v['cases']]==['The capital of France is','대한민국의 수도는'],'bilingual prompts missing')
    require(all(r['finite'] and r['generation_scores_finite'] and r['generation_cache'] and r['new_tokens']>0 for r in v['cases']),'bilingual finite forward/cache/generation failure')
    require(e['output']==str(model) and e['source']==str(source) and e['source_step']==1000 and e['model_prefix']=='' and e['all_source_model_keys_mapped'] and e['all_source_tensors_finite'],'HF export provenance mismatch')
    require(e['source_weights']=='actual_selected_native_model_only_NOT_original_BASE','BASE substitution forbidden')

def validate_report(r,s,model):
    require(r['status']=='complete' and r['suite_sha256']==s['suite_sha256']==SUITE_SHA and r['model']==str(model),'partial/wrong suite/model report')
    require(r['device']=='cpu' and r['dtype']=='bfloat16' and r['transformers']=='5.12.1' and r['torch']=='2.13.0a0+8145d630e8.nv26.06' and r['official_harness_backend'] is False,'CPU backend/version drift')
    require(r['parameters_unchanged'] is True and r['no_padding'] is True,'scoring changed parameters or padding')
    require(r['model_class']=='transformers.models.nemotron_h.modeling_nemotron_h.NemotronHForCausalLM','wrong scoring class')
    qs=r['questions'];hs=r['heldout']
    require(len(qs)==r['question_count']==len(s['questions'])==400 and len(hs)==r['heldout_window_count']==len(s['heldout'])==64,'incomplete suite coverage')
    require([q['id'] for q in qs]==[q['id'] for q in s['questions']] and len({q['id'] for q in qs})==400,'question IDs/order/duplicates mismatch')
    require([w['id'] for w in hs]==[w['id'] for w in s['heldout']] and len({w['id'] for w in hs})==64,'heldout IDs/order/duplicates mismatch')
    for q,expected in zip(qs,s['questions']):
        vals=q['choice_logp_full_vocab_sum'];gold=expected['gold']
        require(q['task']==expected['task'] and q['gold']==gold and len(vals)==len(expected['choices'])==4,'question gold/task/choices mismatch')
        require(all(math.isfinite(x) for x in [*vals,q['gold_logp'],q['gold_choice_conditional_logp']]),'nonfinite question score')
        pred=max(range(4),key=lambda i:vals[i]);predn=max(range(4),key=lambda i:vals[i]/len(expected['choices'][i]))
        require(q['prediction']==pred and q['prediction_char_norm']==predn and q['acc']==int(pred==gold) and q['acc_norm']==int(predn==gold) and q['gold_logp']==vals[gold],'score aggregation mismatch')
        mx=max(vals);conditional=vals[gold]-(mx+math.log(sum(math.exp(v-mx) for v in vals)))
        require(math.isclose(conditional,q['gold_choice_conditional_logp'],rel_tol=1e-10,abs_tol=1e-10),'conditional choice probability mismatch')
    for w,expected in zip(hs,s['heldout']):
        require(w['source']==expected['source'] and w['target_count']==expected['target_count']==1024,'heldout source/targets mismatch')
        require(math.isfinite(w['nll_sum']) and math.isfinite(w['nll_mean']) and math.isclose(w['nll_sum']/1024,w['nll_mean'],rel_tol=1e-10),'heldout nonfinite/normalization mismatch')
    require(sum(w['target_count'] for w in hs)==r['heldout_target_count']==65536,'heldout target count mismatch')
    nll=sum(w['nll_sum'] for w in hs)/65536
    require(math.isclose(nll,r['heldout_full_vocab_nll'],rel_tol=1e-10) and math.isclose(math.exp(nll),r['heldout_ppl'],rel_tol=1e-10),'heldout NLL/PPL mismatch')
    require(set(r['metrics'])==set(s['counts']),'benchmark metric groups mismatch')
    for task,count in s['counts'].items():
        subset=[q for q in qs if q['task']==task];metric=r['metrics'][task]
        require(len(subset)==metric['n']==count and metric['correct']==sum(q['acc'] for q in subset),'task denominator mismatch')
        require(math.isclose(metric['acc'],sum(q['acc'] for q in subset)/count) and math.isclose(metric['acc_norm'],sum(q['acc_norm'] for q in subset)/count),'task metric mismatch')
    return {'questions':400,'choices':sum(len(q['choice_logp_full_vocab_sum']) for q in qs),'heldout_windows':64,'heldout_targets':65536,'model':str(model),'suite_sha256':SUITE_SHA}

OWNER='nemochat.ablation-auto.owner'
STAGES=['schema','export','validate','eval-C','eval-B']
import threading
CREATE_LOCK=threading.RLock()
STOPPING=threading.Event()

def load_module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def engine():
    return load_module(ROOT/'results/pretrain_followup/engine_helper.py','auto_engine')

def load_suite(path=EOUT/'suite.json'):
    s=read(path);h=s.pop('suite_sha256')
    actual=hashlib.sha256(json.dumps(s,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    require(actual==h==SUITE_SHA,'frozen suite canonical digest mismatch');s['suite_sha256']=h;return s

def native_gate():
    for marker in ['intentional-stop.json','intentional-user-stop.json','failure.json']:
        require(not (B/marker).exists(),'B failed/intentionally stopped; no automatic restart')
    supersession=read(B/'supersession.json')
    require(supersession['current_generation']==B.name and supersession['fresh_original_BASE0'] is True,'B superseded')
    require(read(B/'engine-created.json')['id']==BENGINE,'wrong native B Engine receipt')
    inspect=engine().request('GET',f'/containers/{BENGINE}/json');clean_engine(inspect,BENGINE)
    require(inspect['Config']['Labels'].get('nemochat.ablation.lmonly1000.ordinary02.owner')=='student','native B label mismatch')
    require(read(B/'complete.json')['lm_coefficient']==1.0 and read(B/'complete.json')['kd_disabled'] is True,'ordinary B completion missing')
    lineage=read(B/'restart-lineage.json')
    require(lineage['source_hf']==str(MODELS['A']) and lineage['start_step']==0 and lineage['lm_coefficient']==1.0 and lineage['kd_disabled'] is True,'B source/objective changed')
    require(all(lineage[k] is True for k in ['model_only','optimizer_reset','rng_reset','cursor_reset']),'B not fresh original BASE')
    mapping=read(B/'replay-manifest.json')['accepted_physical_mapping']
    require(mapping==read(C/'final-dynamic-state.json')['committed_mapping'] and len(mapping)==len(set(mapping))==1000,'B/C accepted physical mapping mismatch')
    for rank in range(8):
        bu=rows(B/f'student-rank{rank}-updates.jsonl');cu=rows(C/f'student-rank{rank}-updates.jsonl')
        validate_rank(rank,bu,read(B/f'student-rank{rank}-complete.json'),mapping,lmonly=True)
        validate_rank(rank,cu,read(C/f'student-rank{rank}-complete.json'),mapping)
        require([x['lr_used'] for x in bu]==[x['lr_used'] for x in cu],'B/C Learning Rate mismatch')
        bs=rows(B/f'student-rank{rank}-samples.jsonl');cs=rows(C/f'student-rank{rank}-samples.jsonl')
        validate_samples(rank,bs,cs,mapping)
        for i,x in enumerate(bs):require(x['source_index']==mapping[i//16]*128+(i%16)*8+rank,'B physical source index mismatch')
    return {'B':validate_checkpoint(B,mapping,True),'C':validate_checkpoint(C,mapping,False),'calls_per_rank':1000,'positive_updates_per_rank':999,'zero_lr_calls_per_rank':1,'actual_input_samples_all_ranks':128000,'actual_input_digests_exact_C':True}

def guard_spec(spec):
    h=spec['HostConfig']
    require(not h.get('DeviceRequests') and not h.get('Devices') and h.get('Runtime','runc')!='nvidia','CPU job requested GPU')
    require(h['NetworkMode']=='none' and h['NanoCpus']==4000000000 and h['Memory']==137438953472,'CPU/network/memory boundary changed')
    spec['Labels'][OWNER]=UNIT

def owned_name(stage):
    require(stage in STAGES,'unknown owned stage');return UNIT.removesuffix('.service')+'-'+stage.lower()

def cleanup(request=None):
    # Names are fixed before any create; labels and saved ID prohibit collisions.
    STOPPING.set();request=request or engine().request;outcomes=[]
    with CREATE_LOCK:
        for stage in STAGES:
            name=owned_name(stage)
            try:a=request('GET',f'/containers/{name}/json')
            except RuntimeError as exc:
                if '404' in str(exc):continue
                raise
            require(a['Name']=='/'+name and a['Config']['Labels'].get(OWNER)==UNIT,'refusing cleanup of nonowned Engine')
            saved=OUT/(stage+'.owned.json')
            if saved.exists():require(read(saved)['id']==a['Id'],'owned ID changed')
            if a['State']['Running']:request('POST',f"/containers/{a['Id']}/stop?t=20")
            after=request('GET',f"/containers/{a['Id']}/json")
            require(not after['State']['Running'],'owned Engine still running after stop')
            outcomes.append({'id':a['Id'],'name':name,'state':after['State']})
    return outcomes

def convert_command(schema=False):
    command=['python',str(EXPORT_CODE/'convert_kd1000.py'),'--source',str(B/'checkpoint/iter_0001000'),'--output',str(BMODEL),'--report-dir',str(OUT/'B-export'),'--source-step','1000','--stage','midtraining_lm_only','--model-prefix','']
    return command+(['--schema-only'] if schema else [])

def preview(verify=True):
    if verify:verify_freeze()
    return {'status':'INERT_PREVIEW_PARENT_REVIEW_REQUIRED','production_started':False,'unit':UNIT,'uid':1001,'B_identity':BIDENTITY,'B_unit':BUNIT,'B_engine':BENGINE,'A_engine':AENGINE,'A_action':'reuse exact existing Engine, blocking wait; never restart','export_command':convert_command(),'schema_command':convert_command(True),'models_display_order':{a:str(MODELS[a]) for a in ['A','C','B']},'suite_sha256':SUITE_SHA,'start_cli':f'bash {HERE}/start.sh --start','controller_argv':[str(ROOT/'.venv/bin/python'),'-u',str(HERE/'controller.py'),'run'],'final_marker':str(OUT/'complete.json'),'failure_marker':str(OUT/'failure.json'),'runtime_max':'23h','restart':'no','backend':'unchanged existing CPU BF16 Transformers5.12.1; 4 CPU / 128GiB per Engine; no GPU or network','source_hashes':source_hashes() if verify else {}}

def pipeline(out,ops):
    phase='waiting_B';cleaned=False
    try:
        write(out/'state.json',{'phase':phase});ops.wait_b()
        phase='native_gate';write(out/'state.json',{'phase':phase});ops.native()
        phase='export_and_evaluate';write(out/'state.json',{'phase':phase})
        cj=ops.start_eval('C');ops.export_b();bj=ops.start_eval('B')
        ops.wait_a();ops.finish_eval('C',cj);ops.finish_eval('B',bj)
        phase='final_verification';write(out/'state.json',{'phase':phase});result=ops.final()
        require(result.get('status')=='verified_all_three_complete' and result.get('all_three_complete') is True,'partial final proof')
        ops.cleanup();cleaned=True
        write(out/'state.json',{'phase':'complete'})
        write(out/'complete.json',result)  # Last atomic publication, after readback and cleanup.
        return result
    except BaseException as exc:
        write(out/'failure.json',{'status':'failed_or_superseded','phase':phase,'type':type(exc).__name__,'error':str(exc),'no_retry_no_restart_no_suite_shrink':True})
        write(out/'state.json',{'phase':'failed'});raise
    finally:
        if not cleaned:ops.cleanup()

def cpu_job(stage,command,client=None):
    verify_freeze();client=client or engine();name=owned_name(stage)
    launcher=load_module(EVAL/'launch.py' if stage.startswith('eval-') else EXPORT_CODE/'engine_run.py','owned_'+stage.replace('-','_'))
    launcher.OUT=OUT;launcher.Connection=client.Connection
    def request(method,path,data=None,**kwargs):
        if method=='POST' and path.startswith('/containers/create'):
            with CREATE_LOCK:
                require(not STOPPING.is_set(),'controller stopping; refusing new Engine create')
                require(path=='/containers/create?name='+name,'unexpected Engine create target')
                guard_spec(data);write(OUT/(stage+'.intent.json'),{'name':name,'owner':UNIT,'spec':data})
                result=client.request(method,path,data,**kwargs)
                write(OUT/(stage+'.owned.json'),{'name':name,'id':result['Id'],'owner':UNIT});return result
        return client.request(method,path,data,**kwargs)
    launcher.request=request
    if stage.startswith('eval-'):code=launcher.run(name,command,network=False,seconds=21600)
    else:code=launcher.run(name,command,export_dir=BMODEL,receipt_dir=OUT/'B-export')
    cid=read(OUT/(stage+'.owned.json'))['id'];inspect=client.request('GET',f'/containers/{cid}/json')
    clean_engine(inspect,cid)
    require(code==0 and inspect['Config']['Labels'].get(OWNER)==UNIT,'owned launcher failed')
    h=inspect['HostConfig'];require(not h.get('DeviceRequests') and not h.get('Devices') and h['Memory']==137438953472 and h['NanoCpus']==4000000000 and h['NetworkMode']=='none','actual Engine CPU boundaries differ')
    write(OUT/(stage+'.final.json'),inspect);return inspect

def final_reports(paths,suite,out):
    reports={a:read(paths[a]) for a in ['A','C','B']}
    for a,r in reports.items():
        validate_report(r,suite,MODELS[a])
        # Read back every actual per-question evaluator receipt, not counts alone.
        for q in r['questions']:
            qp=paths[a].parent/(q['id'].replace('/','__')+'.json')
            require(read(qp)==q,'per-question execution receipt differs from full report')
    summary=load_module(HERE/'report.py','auto_report').render(reports,suite,out)
    require(summary['status']=='verified_all_three_complete','partial/fixture cannot be production accepted')
    return {'status':'verified_all_three_complete','verified_reports':{a:str(paths[a].resolve()) for a in ['A','C','B']},'report_sha256':{a:hashlib.sha256(paths[a].read_bytes()).hexdigest() for a in paths},'images':summary['images'],'summary':str((out/'summary.json').resolve()),'reports':[str((out/'FINAL_REPORT.md').resolve()),str((out/'FINAL_REPORT.ko.md').resolve())],'suite_sha256':SUITE_SHA,'coverage':summary['coverage'],'checkpoint_integrity':'not_checked_explicit_user_waiver','all_three_complete':True}

def allowlist():
    files=[HERE/n for n in ['controller.py','report.py','test_controller.py','start.sh','README.md','README.ko.md']]
    files += [EVAL/n for n in ['eval.py','launch.py','compare.py','test_suite.py','README.md','README.ko.md']]
    files += [EXPORT_CODE/n for n in ['convert_kd1000.py','validate_hf.py','engine_run.py','README.md']]
    files += [ROOT/'results/pretrain_followup/engine_helper.py',ROOT/'results/pretraining/nemochat_baby/hf_transformers_conversion/convert_base.py',ROOT/'results/pretraining/nemochat_baby/hf_transformers_conversion/runtime_sources/nemotron_h_bridge.py',ROOT/'checkpoints/exports/nemochat_baby_base_bf16/model_config.json',EOUT/'suite.json',EOUT/'ablation-eval-A-full01.create.json']
    files += [B/n for n in ['replay-manifest.json','run-config.json','restart-lineage.json','source-freeze.json','ownership.json','engine-created.json','config-comparison-to-C.json']]
    files += [ROOT/'results/midtraining/hf-kd1000-conversion-01'/n for n in ['export_result.json','validation_result.json','final_acceptance.json']]
    return sorted(files)

def source_hashes():
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in allowlist()}

def verify_freeze():
    require(read(OUT/'source-freeze.json')==source_hashes(),'reviewed source allowlist changed; parent re-review required')
    load_suite()

def unit_mainpid(unit):
    result=subprocess.run(['systemctl','--user','show',unit,'-p','MainPID','--value'],text=True,capture_output=True)
    return int(result.stdout.strip() or '0')

def a_health():
    creation=read(EOUT/'ablation-eval-A-full01.create.json')
    require(creation['id']==AENGINE,'external A Engine receipt changed')
    a=engine().request('GET',f'/containers/{AENGINE}/json')
    require(a['Id']==AENGINE and a['Config']['Cmd']==creation['spec']['Cmd'],'external A command changed')
    if not a['State']['Running']:clean_engine(a,AENGINE)
    return a

class ProductionOps:
    def __init__(self):
        self.pool=concurrent.futures.ThreadPoolExecutor(max_workers=2)
        self.deadline=time.monotonic()+22*3600
        self.suite=load_suite()
        self.paths={'A':EOUT/'A-full01/report.json','C':OUT/'C-full01/report.json','B':OUT/'B-full01/report.json'}
    def wait_b(self):
        pid=unit_mainpid(BUNIT)
        require(pid in [0,BIDENTITY['pid']],'B service replaced by different supervisor')
        if pid:wait_exact(BIDENTITY,min(12*3600,self.deadline-time.monotonic()-8*3600))
        elif Path(f"/proc/{BIDENTITY['pid']}").exists():wait_exact(BIDENTITY,12*3600)
        # Missing/zombie alone is never successful completion; native() decides.
    def native(self):
        verify_freeze();a_health();proof=native_gate();write(OUT/'B-native-accepted.json',proof)
        accepted=ROOT/'results/midtraining/hf-kd1000-conversion-01'
        validate_hf(read(accepted/'validation_result.json'),read(accepted/'export_result.json'),MODELS['C'],C/'checkpoint/iter_0001000')
        require(read(accepted/'final_acceptance.json')['status']=='accepted_CPU_model_executed','C not parent accepted')
    def start_eval(self,arm):
        require(arm in ['C','B'],'never launch/restart external A')
        require(not self.paths[arm].parent.exists(),'fresh evaluation output required')
        command=['python',str(EVAL/'eval.py'),'score','--model',str(MODELS[arm]),'--suite',str(EOUT/'suite.json'),'--out',str(self.paths[arm].parent)]
        return self.pool.submit(cpu_job,'eval-'+arm,command)
    def export_b(self):
        BMODEL.mkdir(exist_ok=False);(OUT/'B-export').mkdir(exist_ok=False)
        cpu_job('schema',convert_command(True));s=read(OUT/'B-export/schema_result.json')
        require(s['status']=='passed' and s['schema_exact'] is True and s['source']==str(B/'checkpoint/iter_0001000') and s['native_model_tensor_records']==619,'B metadata-only schema gate failed')
        cpu_job('export',convert_command())
        cpu_job('validate',['python',str(EXPORT_CODE/'validate_hf.py'),'--model',str(BMODEL),'--result',str(OUT/'B-export/validation_result.json'),'--export-result',str(OUT/'B-export/export_result.json')])
        v=read(OUT/'B-export/validation_result.json');e=read(OUT/'B-export/export_result.json');validate_hf(v,e,BMODEL,B/'checkpoint/iter_0001000')
        require(e['stage']=='midtraining_lm_only','B export stage mismatch');write(OUT/'B-hf-accepted.json',{'status':'accepted_CPU_model_executed','model':str(BMODEL),'validation':str(OUT/'B-export/validation_result.json')})
    def wait_a(self):
        require(read(EOUT/'ablation-eval-A-full01.create.json')['id']==AENGINE,'external A Engine changed')
        e=engine();a=e.request('GET',f'/containers/{AENGINE}/json')
        require(a['Id']==AENGINE and a['Config']['Cmd']==read(EOUT/'ablation-eval-A-full01.create.json')['spec']['Cmd'],'external A command changed')
        if a['State']['Running']:
            result=e.request('POST',f'/containers/{AENGINE}/wait?condition=not-running',timeout=min(21660,self.deadline-time.monotonic()))
            require(result['StatusCode']==0 and not result.get('Error'),'existing A failed; no retry')
        a=e.request('GET',f'/containers/{AENGINE}/json');clean_engine(a,AENGINE)
        write(OUT/'A-external-final.json',a);validate_report(read(self.paths['A']),self.suite,MODELS['A'])
    def finish_eval(self,arm,job):
        job.result(timeout=max(1,self.deadline-time.monotonic()))
        validate_report(read(self.paths[arm]),self.suite,MODELS[arm])
    def final(self):
        verify_freeze();native_gate()
        for stage in STAGES:
            cid=read(OUT/(stage+'.owned.json'))['id'];clean_engine(engine().request('GET',f'/containers/{cid}/json'),cid)
        clean_engine(engine().request('GET',f'/containers/{AENGINE}/json'),AENGINE)
        result=final_reports(self.paths,self.suite,OUT)
        result['native_proof']=str(OUT/'B-native-accepted.json');result['HF_validation']=str(OUT/'B-export/validation_result.json')
        return result
    def cleanup(self):
        try:write(OUT/'cleanup.json',cleanup())
        finally:self.pool.shutdown(wait=True,cancel_futures=True)

def preflight():
    verify_freeze();require(os.getuid()==1001,'unexpected controller UID')
    for p in [OUT/'run.lock',OUT/'state.json',OUT/'failure.json',OUT/'complete.json',BMODEL]:require(not p.exists(),'fresh-only controller refuses existing '+str(p))
    require(unit_mainpid(UNIT)==0,'controller unit already running')
    pid=unit_mainpid(BUNIT);require(pid in [0,BIDENTITY['pid']],'B service replaced')
    if pid:check_identity(process_identity(pid),BIDENTITY)
    else:native_gate()
    e=engine();a=e.request('GET',f'/containers/{AENGINE}/json')
    require(a['Id']==AENGINE,'wrong A Engine')
    if not a['State']['Running']:clean_engine(a,AENGINE);validate_report(read(EOUT/'A-full01/report.json'),load_suite(),MODELS['A'])
    return preview()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',nargs='?',choices=['preview','preflight','run','cleanup','freeze'],default='preview');a=p.parse_args()
    if a.action=='freeze':
        require(not (OUT/'source-freeze.json').exists(),'freeze already exists; parent review required');write(OUT/'source-freeze.json',source_hashes());return
    if a.action=='preview':print(json.dumps(preview(),indent=2));return
    if a.action=='preflight':print(json.dumps(preflight(),indent=2));return
    if a.action=='cleanup':
        try:write(OUT/'cleanup.json',cleanup())
        finally:
            if (OUT/'run.lock').exists() and not (OUT/'complete.json').exists() and not (OUT/'failure.json').exists():write(OUT/'failure.json',{'status':'failed_or_superseded','error':'supervisor exited/stopped without full final acceptance','no_retry':True})
        return
    # This branch is only reachable via explicit parent-reviewed start.
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'run.lock').open('x') as f:f.write(str(os.getpid()))
    try:
        verify_freeze()
        require(os.getuid()==1001 and unit_mainpid(UNIT)==os.getpid(),'not exact durable controller unit/UID')
        actual=process_identity(os.getpid());require(actual['argv']==[str(ROOT/'.venv/bin/python'),'-u',str(HERE/'controller.py'),'run'],'wrong controller argv')
        write(OUT/'ownership.json',{'unit':UNIT,**actual,'prerequisite':BIDENTITY})
        def stopped(signum,frame):raise InterruptedError('controller stopped; prerequisites untouched')
        signal.signal(signal.SIGTERM,stopped);signal.signal(signal.SIGINT,stopped)
        pipeline(OUT,ProductionOps())
    except BaseException as exc:
        if not (OUT/'failure.json').exists():write(OUT/'failure.json',{'status':'failed_or_superseded','error':str(exc),'type':type(exc).__name__,'no_retry':True})
        raise

if __name__=='__main__':main()
