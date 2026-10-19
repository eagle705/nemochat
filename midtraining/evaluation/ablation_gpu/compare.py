"""Paired full-suite comparisons; positive Accuracy/gold-logp is better for right arm."""
import argparse,collections,json,random,math
from pathlib import Path
from eval import read_suite,save

def interval(values,rng,iters=5000):
 groups=list(values.values());n=sum(len(g) for g in groups)
 samples=sorted(sum(sum(rng.choice(g) for _ in g) for g in groups)/n for _ in range(iters))
 return [samples[int(.025*iters)],samples[int(.975*iters)]]
def compare(left,right,suite,allow_smoke=False):
 assert left['suite_sha256']==right['suite_sha256']==suite['suite_sha256']
 if not allow_smoke:assert left['status']==right['status']=='complete','partial/smoke not a full-suite comparison'
 l={q['id']:q for q in left['questions']};r={q['id']:q for q in right['questions']};expected={q['id'] for q in suite['questions']}
 assert set(l)==set(r);assert set(l)==expected or allow_smoke
 assert len(l)==len(left['questions'])==len(right['questions'])
 rng=random.Random(1729);groups=collections.defaultdict(list);paired=[]
 for k,a in l.items():
  b=r[k];assert a['gold']==b['gold'] and a['task']==b['task']
  row={'id':k,'task':a['task'],'acc_delta':b['acc']-a['acc'],'acc_norm_delta':b['acc_norm']-a['acc_norm'],'gold_logp_delta':b['gold_logp']-a['gold_logp']};paired.append(row);groups['mmlu200' if a['task'].startswith('mmlu_') else a['task']].append(row)
 scores={}
 for group,rows in groups.items():
  metric='acc_delta' if group=='mmlu200' else 'acc_norm_delta'
  strat=collections.defaultdict(list)
  for row in rows:strat[row['task']].append(row[metric])
  scores[group]={'n':len(rows),'primary_metric':metric,'delta':sum(row[metric] for row in rows)/len(rows),'paired_stratified_bootstrap_95pct':interval(strat,rng),'gold_logp_mean_delta':sum(row['gold_logp_delta'] for row in rows)/len(rows)}
 lh={w['id']:w for w in left['heldout']};rh={w['id']:w for w in right['heldout']};assert set(lh)==set(rh)
 if not allow_smoke:assert set(lh)=={w['id'] for w in suite['heldout']}
 strata=collections.defaultdict(list);windows=[]
 for k,a in lh.items():
  b=rh[k];assert a['target_count']==b['target_count']==1024 and a['source']==b['source'];delta=b['nll_mean']-a['nll_mean'];strata[a['source']].append(delta);windows.append({'id':k,'source':a['source'],'nll_delta':delta})
 return {'status':'smoke_selfcheck_only' if allow_smoke else 'complete_paired_comparison','left_model':left['model'],'right_model':right['model'],'suite_sha256':suite['suite_sha256'],'benchmark_groups':scores,'heldout_nll_delta':right['heldout_full_vocab_nll']-left['heldout_full_vocab_nll'],'heldout_nll_paired_source_stratified_bootstrap_95pct':interval(strata,rng),'paired_questions':paired,'paired_windows':windows,'bootstrap':{'seed':1729,'iterations':5000,'scope':'paired question bootstrap within MMLU subject; paired window bootstrap within source; not training-seed uncertainty'},'interpretation':'B ordinary 1.0 LM vs C deployed 0.5 LM + 0.5 KD is practical recipe comparison, not matched-LM-coefficient KD-term isolation. No extra 0.5 LM control requested.'}
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--left',required=True);p.add_argument('--right',required=True);p.add_argument('--suite',required=True);p.add_argument('--out',required=True);p.add_argument('--allow-smoke',action='store_true');a=p.parse_args();x=compare(json.loads(Path(a.left).read_text()),json.loads(Path(a.right).read_text()),read_suite(a.suite),a.allow_smoke);save(a.out,x);print(json.dumps({k:v for k,v in x.items() if not k.startswith('paired_')},indent=2))
