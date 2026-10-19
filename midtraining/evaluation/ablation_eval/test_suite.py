"""Real frozen-row parity against the downloaded official v0.4.8 code."""
import ast,collections,hashlib,importlib.util,json,random,types,unittest
from pathlib import Path
import eval as ev
class SuiteTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.s=ev.read_suite(ev.OUT/'suite.json')
 def test_counts_and_ids(self):
  s=self.s;self.assertEqual(len(s['questions']),400);self.assertEqual(len({q['id'] for q in s['questions']}),400);self.assertEqual(sum(s['counts'][k] for k in s['counts'] if k.startswith('mmlu_')),200)
  self.assertEqual(s['counts']['arc_challenge'],100);self.assertEqual(s['counts']['hellaswag'],100)
  self.assertEqual(len(s['heldout']),64);self.assertEqual(sum(w['target_count'] for w in s['heldout']),65536)
  for w in s['heldout']:
   self.assertEqual(len(w['ids']),1025);self.assertFalse(set(w['ids'])&set(range(18,30)))
   import numpy as np
   self.assertEqual(ev.sha(np.array(w['ids'],dtype='<i4').tobytes()),w['input_sha256'])
  for q in s['questions']:
   self.assertEqual(ev.sha(ev.canonical(q['raw'])),q['raw_sha256']);self.assertEqual(q['fewshot_ids'],list(range(5)) if q['task'].startswith('mmlu_') else [])
 def test_official_source_on_all_real_frozen_rows(self):
  import lm_eval.api.task as tm
  from lm_eval.api.task import ConfigurableTask
  from lm_eval.utils import load_yaml_config
  from datasets import Dataset,DatasetDict
  import lm_eval
  r=Path(lm_eval.__file__).parent
  source=(ev.OUT/'research/lm_eval__api__task.py').read_text();tree=ast.parse(source);cl=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ConfigurableTask');namespace=dict(vars(tm));methods={}
  for name in ['fewshot_context','construct_requests','process_results']:
   node=next(n for n in cl.body if isinstance(n,ast.FunctionDef) and n.name==name);exec(compile(ast.Module(body=[node],type_ignores=[]),'<official-v0.4.8>','exec'),namespace);methods[name]=namespace[name]
  class Local(ConfigurableTask):
   def download(self,*a,**kw):self.dataset=self.frozen_dataset
  Official=type('OfficialV048',(Local,),methods)
  for taskname in self.s['samples']:
   rows=[q for q in self.s['questions'] if q['task']==taskname];ism=taskname.startswith('mmlu_');arc=taskname=='arc_challenge';split=rows[0]['split'];data=Dataset.from_list([q['raw'] for q in rows]);dd={split:data,'train':data}
   if ism:dd['dev']=Dataset.from_list(rows[0]['fewshot_raw'])
   Local.frozen_dataset=DatasetDict(dd);cfg=load_yaml_config(str(r/('tasks/mmlu/default/'+taskname+'.yaml' if ism else 'tasks/arc/arc_challenge.yaml' if arc else 'tasks/hellaswag/hellaswag.yaml')));t=Official(config=cfg);t.set_fewshot_seed(1234)
   for q in rows:
    d=dict(q['raw'])
    if taskname=='hellaswag':d.update(query=q['prompt'],choices=q['choices'])
    ctx=t.fewshot_context(d,5 if ism else 0);self.assertEqual(ctx,q['prompt']);self.assertEqual([i.args for i in t.construct_requests(d,ctx)],[(q['prompt'],' '+c) for c in q['choices']]);self.assertEqual(int(t.doc_to_target(d)),q['gold'])
    vals=[-float(i+1)**2 for i in range(len(q['choices']))];result=t.process_results(d,[(v,False) for v in vals]);self.assertEqual(result['acc'],float(q['gold']==0))
    if not ism:self.assertEqual(result['acc_norm'],float(q['gold']==max(range(len(vals)),key=lambda i:vals[i]/len(q['choices'][i]))))
 def test_official_pair_tokenization_all_real_choices(self):
  from transformers import AutoTokenizer
  tok=AutoTokenizer.from_pretrained(ev.BASE,local_files_only=True,trust_remote_code=False)
  s=(ev.OUT/'research/lm_eval__api__model.py').read_text();tree=ast.parse(s);node=next(n for c in tree.body if isinstance(c,ast.ClassDef) for n in c.body if isinstance(n,ast.FunctionDef) and n.name=='_encode_pair');node.decorator_list=[]
  import transformers
  ns={'Tuple':tuple,'List':list,'transformers':transformers};exec(compile(ast.Module(body=[node],type_ignores=[]),'<official-pair>','exec'),ns)
  fake=types.SimpleNamespace(tok_encode=lambda x,**kw:tok.encode(x,add_special_tokens=False))
  for q in self.s['questions']:
   for c,e in zip(q['choices'],q['encoded']):
    a,b=ns['_encode_pair'](fake,q['prompt'],' '+c);self.assertEqual(a,e['context_ids']);self.assertEqual(b,e['continuation_ids']);self.assertEqual((a,b),ev.pair(tok,q['prompt'],' '+c))
 def test_nll_full_vocabulary_shift(self):
  import torch
  torch.manual_seed(3);logits=torch.randn(1,8,19);ids=torch.randint(0,19,(1,9));context=5
  ce=torch.nn.functional.cross_entropy(logits[0,context-1:].float(),ids[0,context:],reduction='none');lp=torch.log_softmax(logits,dim=-1)[0].gather(-1,ids[:,1:].T).squeeze(-1)
  self.assertAlmostEqual(float(ce.sum()),float(-lp[context-1:].sum()),places=5)
if __name__=='__main__':unittest.main(verbosity=2)
