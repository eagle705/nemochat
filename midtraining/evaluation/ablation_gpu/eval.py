"""Frozen source-derived lm-eval v0.4.8 subset evaluator (not official leaderboard)."""
import argparse,collections,hashlib,json,math,os,re,struct,time,random,inspect
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'results/midtraining/offline2b-ablation-eval-01'
BASE=ROOT/'checkpoints/exports/nemochat_baby_base_hf_bf16_02'
SUBJECTS=['abstract_algebra','college_computer_science','college_mathematics','formal_logic','high_school_physics','high_school_chemistry','high_school_biology','high_school_world_history','professional_medicine','business_ethics']
def sha(b):return hashlib.sha256(b).hexdigest()
def canonical(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(x,f,indent=2,ensure_ascii=False,allow_nan=False)
def read_suite(p):
 x=json.loads(Path(p).read_text());h=x.pop('suite_sha256');assert sha(canonical(x))==h,'suite changed';x['suite_sha256']=h;return x
def pair(tok,ctx,cont):
 # Exact TemplateLM._encode_pair causal branch; no BOS/chat template.
 n=len(ctx)-len(ctx.rstrip())
 if n:cont=ctx[-n:]+cont;ctx=ctx[:-n]
 c=tok.encode(ctx,add_special_tokens=False);w=tok.encode(ctx+cont,add_special_tokens=False)
 assert w[:len(c)]==c,'context boundary retokenization';assert len(w)>len(c)
 return c,w[len(c):]
def mmlu_text(d):return d['question'].strip()+'\n'+'\n'.join(f'{c}. {v}' for c,v in zip('ABCD',d['choices']))+'\nAnswer:'
def clean(t):return re.sub(r'\[.*?\]','',t.strip().replace(' [title]','. ')).replace('  ',' ')
def direct_bytes(p):
 # Direct I/O avoids trusting a repeated buffered read on this faulting host.
 import mmap
 fd=os.open(str(p),os.O_RDONLY|os.O_DIRECT);size=os.fstat(fd).st_size;buf=mmap.mmap(-1,((size+4095)//4096)*4096)
 try:
  n=os.readv(fd,[buf]);assert n==size,(p,n,size);return bytes(buf[:size])
 finally:buf.close();os.close(fd)
def heldout():
 import numpy as np
 manifest_path=ROOT/'datasets/midtraining/super35-kd-01/offline1920-v1/data/manifest.json';mb=manifest_path.read_bytes();m=json.loads(mb);assert m['audit']['train_heldout_token_document_overlap']==0
 windows=[];evidence=[]
 for src in m['source_order']:
  v=m['sources'][src]['validation'];parts={}
  for ext in ['bin','idx']:
   p=Path(v['prefix']+'.'+ext);b=direct_bytes(p);assert sha(b)==v[ext+'_sha256'],('source direct digest mismatch',src,ext)
   parts[ext]=b
  idx=parts['idx'];assert idx[:9]==b'MMIDIDX\x00\x00';assert struct.unpack_from('<Q',idx,9)[0]==1;assert idx[17]==4
  ns,nd=struct.unpack_from('<QQ',idx,18);sizes=np.frombuffer(idx,dtype='<i4',count=ns,offset=34).copy();ptr=np.frombuffer(idx,dtype='<i8',count=ns,offset=34+4*ns).copy()
  tokens=np.frombuffer(parts['bin'],dtype='<i4').copy();assert len(tokens)==v['tokens'];assert len(sizes)==v['documents'];assert int(sizes.sum())==len(tokens);assert np.all(ptr==np.cumsum(np.r_[0,sizes[:-1]],dtype=np.int64)*4)
  assert not np.isin(tokens,np.arange(18,30)).any(),'teacher alias IDs in heldout';assert tokens.min()>=0 and tokens.max()<131072
  for i in range(16):
   a=i*1024;b=a+1025;ids=tokens[a:b].tolist();assert len(ids)==1025
   docs=[]
   for j,(p,n) in enumerate(zip(ptr//4,sizes)):
    lo=max(a,int(p));hi=min(b,int(p+n))
    if lo<hi:docs.append({'document':j,'doc_token_start':lo-int(p),'doc_token_end':hi-int(p)})
   windows.append({'id':f'{src}:{a}:{b}','source':src,'flat_token_start':a,'flat_token_end':b,'document_ranges':docs,'ids':ids,'target_count':1024,'input_sha256':sha(np.array(ids,dtype='<i4').tobytes())})
  evidence.append({'source':src,'prefix':v['prefix'],'bin_sha256':sha(parts['bin']),'idx_sha256':sha(idx),'read':'O_DIRECT authenticated immutable bytes','reserved':True})
 return windows,{'manifest_sha256':sha(mb),'source_audit_overlap':0,'sources':evidence,'note':'Inherited reserved-corpus exclusion evidence, not a new exhaustive raw text decontamination scan.'}
def prepare(path):
 from datasets import load_dataset,DatasetDict
 from transformers import AutoTokenizer
 from lm_eval.api.task import ConfigurableTask
 from lm_eval.utils import load_yaml_config
 import lm_eval
 tok=AutoTokenizer.from_pretrained(BASE,local_files_only=True,trust_remote_code=False);pins=json.loads((OUT/'research/pins.json').read_text());r=Path(lm_eval.__file__).parent;rows=[];samples={};checks=[]
 class LocalTask(ConfigurableTask):
  def download(self,*args,**kwargs):self.dataset=self.frozen_dataset
 for subject in SUBJECTS+['arc_challenge','hellaswag']:
  ism=subject in SUBJECTS;arc=subject=='arc_challenge';repo='cais/mmlu' if ism else 'allenai/ai2_arc' if arc else 'Rowan/hellaswag';name=subject if ism else 'ARC-Challenge' if arc else None;split='test' if ism or arc else 'validation';n=20 if ism else 100
  ds=load_dataset(repo,name,split=split,revision=pins['datasets'][repo]);dev=load_dataset(repo,name,split='dev',revision=pins['datasets'][repo]) if ism else None
  taskname='mmlu_'+subject if ism else subject;samples[taskname]=list(range(n))
  cfg=load_yaml_config(str(r/('tasks/mmlu/default/mmlu_'+subject+'.yaml' if ism else 'tasks/arc/arc_challenge.yaml' if arc else 'tasks/hellaswag/hellaswag.yaml')))
  # Route frozen already-pinned parquet rows into installed official task logic. No remote code.
  LocalTask.frozen_dataset=DatasetDict({split:ds,**({'dev':dev} if ism else {'train':load_dataset(repo,name,split='train',revision=pins['datasets'][repo])})});task=LocalTask(config=cfg);task.set_fewshot_seed(1234)
  for i in range(n):
   raw=dict(ds[i]);d=dict(raw)
   if ism:
    prompt='The following are multiple choice questions (with answers) about '+subject.replace('_',' ')+'.\n\n'+''.join(mmlu_text(ex)+' '+ 'ABCD'[ex['answer']]+'\n\n' for ex in dev.select(range(5)))+mmlu_text(d);choices=list('ABCD');gold=int(d['answer'])
   elif arc:prompt='Question: '+d['question']+'\nAnswer:';choices=d['choices']['text'];gold=d['choices']['label'].index(d['answerKey'])
   else:
    prompt=clean(d['activity_label']+': '+d['ctx_a']+' '+d['ctx_b'].capitalize());choices=[clean(x) for x in d['endings']];gold=int(d['label']);d.update(query=prompt,choices=choices)
   official=task.fewshot_context(d,5 if ism else 0);assert prompt==official,(taskname,i,'prompt mismatch')
   req=task.construct_requests(d,official);assert [x.args for x in req]==[(prompt,' '+c) for c in choices]
   assert int(task.doc_to_target(d))==gold
   encoded=[]
   for choice in choices:
    c,t=pair(tok,prompt,' '+choice);encoded.append({'context_ids':c,'continuation_ids':t,'continuation':' '+choice})
   assert task.process_results(d,[(float(j),False) for j in range(len(choices))])['acc']==float(gold==len(choices)-1)
   rows.append({'id':f'{taskname}/{split}/{i}','task':taskname,'dataset':repo,'revision':pins['datasets'][repo],'config':name,'split':split,'row_index':i,'native_id':raw.get('id',raw.get('ind',None)),'raw':raw,'raw_sha256':sha(canonical(raw)),'prompt':prompt,'choices':choices,'gold':gold,'fewshot_ids':list(range(5)) if ism else [],'fewshot_raw':[dict(x) for x in dev.select(range(5))] if ism else [],'encoded':encoded})
  checks.append({'task':taskname,'questions':n,'prompt_request_gold_metric_parity':'passed_all_real_selected_rows','config_sha256':sha(canonical({k:str(v) for k,v in cfg.items()}))})
 windows,hold=heldout();assets={p.name:sha(p.read_bytes()) for p in BASE.iterdir() if p.is_file() and (p.name.startswith('tokenizer') or p.name in ['special_tokens_map.json','added_tokens.json'])}
 code={str(p.relative_to(r)):sha(p.read_bytes()) for p in r.rglob('*.py') if ('api/' in str(p.relative_to(r)) or 'tasks/hellaswag/' in str(p.relative_to(r)))}
 x={'schema':1,'protocol':'nonstandard deterministic first-N subset, no leaderboard/full-MMLU claim','pins':pins,'installed_harness':'nvidia_lm_eval 25.11.1 / lm_eval 0.4.8','installed_code_sha256':code,'tokenizer_assets':assets,'tokenizer_vocab_sha256':sha(canonical(tok.get_vocab())),'bos':False,'chat_template':False,'samples':samples,'counts':dict(collections.Counter(x['task'] for x in rows)),'questions':rows,'heldout':windows,'heldout_evidence':hold,'parity':checks,'contamination':'Benchmark train contamination unproven; known reserved corpus exclusions only.','native_model_payload_integrity':'not_checked_explicit_user_waiver'}
 x['suite_sha256']=sha(canonical(x));save(path,x);print(json.dumps({'suite_sha256':x['suite_sha256'],'questions':len(rows),'heldout_windows':len(windows),'targets':sum(w['target_count'] for w in windows),'parity':checks},indent=2))
def loading_metadata(x):
 if isinstance(x,dict):return {k:loading_metadata(v) for k,v in x.items()}
 if isinstance(x,(set,tuple,list)):return [loading_metadata(v) for v in (sorted(x) if isinstance(x,set) else x)]
 return x
def score(a):
 import torch,importlib.metadata as im
 from transformers import AutoModelForCausalLM,AutoTokenizer
 torch.set_num_threads(4);torch.manual_seed(17);assert torch.cuda.is_available();assert torch.cuda.device_count()==1;device='cuda:0';torch.cuda.manual_seed_all(17);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=False;torch.use_deterministic_algorithms(False);suite=read_suite(a.suite);tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True,trust_remote_code=False);assert sha(canonical(tok.get_vocab()))==suite['tokenizer_vocab_sha256']
 assert not Path(a.out).exists();Path(a.out).mkdir(parents=True);start=time.monotonic();model,info=AutoModelForCausalLM.from_pretrained(a.model,trust_remote_code=False,local_files_only=True,dtype=torch.bfloat16,attn_implementation='eager',output_loading_info=True);assert not any(info.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys']);model.to(device);model.eval();assert all(p.device.type=='cuda' and p.dtype==torch.bfloat16 for p in model.parameters());assert all(torch.isfinite(p).all() for p in model.parameters());assert not info.get('error_msgs');versions=[p._version for p in model.parameters()];print('MODEL_LOADED',time.monotonic()-start,flush=True)
 selected=suite['questions'] if not a.smoke else [suite['questions'][0],next(x for x in suite['questions'] if x['task']=='arc_challenge')];windows=suite['heldout'] if not a.smoke else suite['heldout'][:1]
 maxlen=model.config.max_position_embeddings
 def likelihood(context,targets):
  ids=context+targets;assert len(ids)<=maxlen,'no silent truncation';assert context and targets
  with torch.inference_mode():
   x=torch.tensor([ids[:-1]],dtype=torch.long,device=device);logits=model(input_ids=x,use_cache=False,logits_to_keep=0).logits
   # Only cast score-bearing positions; normalization still covers full vocab.
   logits=logits[0,len(context)-1:].float();assert logits.shape[0]==len(targets);assert torch.isfinite(logits).all();y=torch.tensor(targets,device=device);assert logits.device==y.device==x.device;nll=torch.nn.functional.cross_entropy(logits,y,reduction='none');assert torch.isfinite(nll).all();return -float(nll.double().sum())
 questions=[]
 for row in selected:
  vals=[]
  for ch,e in zip(row['choices'],row['encoded']):
   c,t=pair(tok,row['prompt'],' '+ch);assert c==e['context_ids'] and t==e['continuation_ids'];vals.append(likelihood(c,t))
  pred=max(range(len(vals)),key=lambda i:vals[i]);norm=[v/len(c) for v,c in zip(vals,row['choices'])];predn=max(range(len(vals)),key=lambda i:norm[i]);mx=max(vals);gold_cond=vals[row['gold']]-(mx+math.log(sum(math.exp(v-mx) for v in vals)))
  q={'id':row['id'],'task':row['task'],'gold':row['gold'],'choice_logp_full_vocab_sum':vals,'gold_logp':vals[row['gold']],'gold_choice_conditional_logp':gold_cond,'prediction':pred,'prediction_char_norm':predn,'acc':int(pred==row['gold']),'acc_norm':int(predn==row['gold'])};questions.append(q);print(json.dumps(q),flush=True);save(Path(a.out)/(row['id'].replace('/','__')+'.json'),q)
 losses=[]
 for row in windows:
  ids=row['ids'];lp=likelihood(ids[:1],ids[1:]);q={'id':row['id'],'source':row['source'],'target_count':len(ids)-1,'nll_sum':-lp,'nll_mean':-lp/(len(ids)-1)};losses.append(q);print(json.dumps(q),flush=True)
 assert versions==[p._version for p in model.parameters()];assert not any(p.grad is not None for p in model.parameters());n=sum(x['target_count'] for x in losses);loss=sum(x['nll_sum'] for x in losses)/n
 metrics={t:{'n':len(xs),'correct':sum(x['acc'] for x in xs),'acc':sum(x['acc'] for x in xs)/len(xs),'acc_norm':sum(x['acc_norm'] for x in xs)/len(xs)} for t in {x['task'] for x in questions} for xs in [[x for x in questions if x['task']==t]]}
 report={'status':'passed_smoke' if a.smoke else 'complete','suite_sha256':suite['suite_sha256'],'model':str(Path(a.model).resolve()),'model_payload_integrity':'not_checked_explicit_user_waiver','model_class':type(model).__module__+'.'+type(model).__name__,'device':device,'loading_info':loading_metadata(info),'gpu_uuid':str(torch.cuda.get_device_properties(0).uuid),'gpu_name':torch.cuda.get_device_name(0),'cuda':torch.version.cuda,'tf32_matmul':torch.backends.cuda.matmul.allow_tf32,'tf32_cudnn':torch.backends.cudnn.allow_tf32,'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),'cudnn_deterministic':torch.backends.cudnn.deterministic,'cudnn_benchmark':torch.backends.cudnn.benchmark,'attention':'eager','all_loaded_tensors_finite':True,'dtype':'bfloat16','torch':torch.__version__,'transformers':im.version('transformers'),'question_count':len(questions),'heldout_window_count':len(losses),'heldout_target_count':n,'heldout_full_vocab_nll':loss,'heldout_ppl':math.exp(loss),'questions':questions,'heldout':losses,'metrics':metrics,'elapsed_seconds':time.monotonic()-start,'parameters_unchanged':True,'no_padding':True,'official_harness_backend':False,'backend_note':'GPU source-derived official-task-parity scoring, not unmodified harness. Frozen task request/gold/metric parity on all selected rows; official Transformers eager BF16, no native/conversion logit parity claim.'};save(Path(a.out)/'report.json',report);print(json.dumps(report,indent=2),flush=True)
def main():
 p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='mode',required=True);q=s.add_parser('prepare');q.add_argument('--suite',required=True);q=s.add_parser('score');q.add_argument('--model',required=True);q.add_argument('--suite',required=True);q.add_argument('--out',required=True);q.add_argument('--smoke',action='store_true');a=p.parse_args();prepare(a.suite) if a.mode=='prepare' else score(a)
if __name__=='__main__':main()
