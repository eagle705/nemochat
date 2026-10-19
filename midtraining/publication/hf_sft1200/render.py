"""Accepted 1..1248 trajectory only, eight-rank checks and distinct LR streams."""
import csv,json,math,re
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path('/raid/yoon/nemochat'); OUT=ROOT/'results/midtraining/hf-sft1200-publication-01'
A=ROOT/'results/midtraining/sft-after-kd-runtime05'; B=ROOT/'results/midtraining/sft-continue1500-02'
def roll(v):return [sum(v[max(0,i-15):i+1])/min(16,i+1) for i in range(len(v))]
def main():
 ranks=[]
 for rank in range(8):
  rows=[]
  for run,lo,hi in [(A,1,905),(B,906,1248)]:
   part=[json.loads(s) for s in (run/'main'/f'rank-{rank}-updates.jsonl').read_text().splitlines() if s]
   assert [v['update'] for v in part]==list(range(lo,hi+1))
   rows+=part
  assert all(math.isfinite(r[k]) for r in rows for k in ['loss','lr','grad_norm','iteration_seconds'])
  ranks.append(rows)
 for other in ranks[1:]:
  for a,b in zip(ranks[0],other):
   for k in ['loss','lr','grad_norm','global_loss_tokens','global_processed_positions']:
    assert math.isclose(a[k],b[k],rel_tol=1e-7,abs_tol=1e-8),(a['update'],k)
   if a['update']>905:
    assert a['optimizer_step_param_group_lrs']==b['optimizer_step_param_group_lrs']
    assert a['post_scheduler_param_group_lrs']==b['post_scheduler_param_group_lrs']
 rows=ranks[0];sdk={}
 for line in (A/'main-engine.log').read_text(errors='replace').splitlines():
  m=re.search(r'iteration\s+(\d+)/\s*\d+.*?learning rate:\s*([\d.eE+\-]+)',line)
  if m:sdk[int(m[1])]=float(m[2])
 assert set(sdk)==set(range(1,906))
 held=[]
 for run in [A,B]:
  base=[json.loads(s) for s in (run/'main/rank-0-heldout.jsonl').read_text().splitlines() if s]
  for rank in range(1,8):
   other=[json.loads(s) for s in (run/'main'/f'rank-{rank}-heldout.jsonl').read_text().splitlines() if s]
   assert [(x['update'],x['heldout_ce']) for x in other]==[(x['update'],x['heldout_ce']) for x in base]
  held+=base
 assert all(x['update']!=1248 for x in held)
 fields=['update','loss','callback_lr','sdk_post_step_printed_lr','optimizer_boundary_lr','post_scheduler_lr','grad_norm','iteration_seconds','global_loss_tokens','global_processed_positions']
 with (OUT/'observed-metrics.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for r in rows:
   w.writerow(dict(update=r['update'],loss=r['loss'],callback_lr=r['lr'],sdk_post_step_printed_lr=sdk.get(r['update'],''),optimizer_boundary_lr=r.get('optimizer_step_param_group_lrs',[''])[0],post_scheduler_lr=r.get('post_scheduler_param_group_lrs',[''])[0],**{k:r[k] for k in fields[-4:]}))
 (OUT/'validation-observations.json').write_text(json.dumps(held,indent=2)+'\n')
 steps=[r['update'] for r in rows];loss=[r['loss'] for r in rows]
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18})
 fig,axes=plt.subplots(2,2,figsize=(16,10),layout='constrained')
 ax=axes[0,0];ax.plot(steps,loss,alpha=.22,lw=.7,color='#60a5fa',label='Raw native log');ax.plot(steps,roll(loss),color='#1d4ed8',lw=1.7,label='Trailing 16-update mean');ax.set(title='Training LM Loss',ylabel='Native logged LM Loss');ax.legend()
 ax=axes[0,1];ax.plot([v['update'] for v in held],[v['heldout_ce'] for v in held],'o-',color='#059669');ax.set(title='Native validation — fixed 128-dialogue window',ylabel='Native Validation LM Loss',xlim=(0,1270));ax.text(.03,.95,'Interval observations only; no final-1248 evaluation\nNot the pretraining-retention NLL benchmark',transform=ax.transAxes,va='top',fontsize=9)
 for v in held:ax.annotate(f"{v['heldout_ce']:.4f}",(v['update'],v['heldout_ce']),xytext=(4,6),textcoords='offset points',fontsize=8)
 ax=axes[1,0];ax.plot(steps[:905],[r['lr'] for r in rows[:905]],label='1–905: callback record',color='#7c3aed');ax.plot(steps[:905],[sdk[s] for s in steps[:905]],'--',label='1–905: SDK printed post-step',color='#ea580c');ax.plot(steps[905:],[r['optimizer_step_param_group_lrs'][0] for r in rows[905:]],label='906–1248: optimizer boundary',color='#0891b2');ax.plot(steps[905:],[r['post_scheduler_param_group_lrs'][0] for r in rows[905:]],'--',label='906–1248: post-scheduler',color='#db2777');ax.set(title='Observed LR — measurement boundaries differ',ylabel='Learning rate');ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0));ax.legend(fontsize=8,loc='lower right');ax.text(.03,.97,'No reconciled actual optimizer LR claim for 1–905.\nPost-scheduler value is not the current applied LR.',transform=ax.transAxes,va='top',fontsize=8)
 ax=axes[1,1];norm=[r['grad_norm'] for r in rows];ax.plot(steps,norm,alpha=.3,lw=.7,color='#f59e0b',label='Raw');ax.plot(steps,roll(norm),color='#b45309',label='Trailing 16-update mean');ax.set(title='Grad Norm — before native clipping',ylabel='Grad Norm',yscale='log');ax.legend()
 for ax in axes.flat:ax.axvline(905,color='#334155',ls=':',lw=1.3);ax.set_xlabel('Cumulative SFT optimizer update')
 fig.suptitle('FINAL SFT1248 — 905 + 343 full-state-resumed updates',fontsize=18,fontweight='bold')
 fig.supxlabel('Dotted boundary: full optimizer / scheduler / RNG / cursor resume at 905. Calibration excluded. Eight ranks matched.',fontsize=10)
 fig.savefig(OUT/'sft1248-training-summary.png',dpi=150);plt.close(fig)
 summary=dict(updates=len(rows),ranks_verified=8,contiguous_updates=True,rank_metrics_matched=True,assistant_target_exposures=sum(r['global_loss_tokens'] for r in rows),processed_positions_including_padding=sum(r['global_processed_positions'] for r in rows),unique_conversations=len(rows)*128,first16_mean_loss=sum(loss[:16])/16,last16_mean_loss=sum(loss[-16:])/16,validation=held,full_state_resume_step=905,segment_updates=343,calibration_included=False)
 assert summary['assistant_target_exposures']==117409886 and summary['processed_positions_including_padding']==2617245696
 (OUT/'training-summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
