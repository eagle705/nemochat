"""Render the completed SFT main only; never combine calibration attempts."""
import csv,json,math,re
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[3]
RUN=ROOT/'results/midtraining/sft-after-kd-runtime05'
OUT=RUN/'charts-final905'

def rolling(values,window=16):
    return [sum(values[max(0,i-window+1):i+1])/min(window,i+1) for i in range(len(values))]

def load_rows(run=RUN,expected=905):
    ranks=[]
    for rank in range(8):
        rows=[json.loads(s) for s in (run/'main'/f'rank-{rank}-updates.jsonl').read_text().splitlines() if s]
        if len(rows)!=expected or [v['update'] for v in rows]!=list(range(1,expected+1)):
            raise ValueError('incomplete/noncontiguous rank '+str(rank))
        for v in rows:
            if not all(math.isfinite(v[k]) for k in ['loss','lr','grad_norm','iteration_seconds']):raise ValueError('nonfinite metric')
        ranks.append(rows)
    for other in ranks[1:]:
        for a,b in zip(ranks[0],other):
            for key in ['loss','lr','grad_norm','global_loss_tokens','global_processed_positions']:
                if not math.isclose(a[key],b[key],rel_tol=1e-7,abs_tol=1e-8):raise ValueError('rank metric mismatch: '+key)
    return ranks[0]

def main():
    rows=load_rows();OUT.mkdir(parents=True,exist_ok=False)
    sdk={}
    for line in (RUN/'main-engine.log').read_text(errors='replace').splitlines():
        m=re.search(r'iteration\s+(\d+)/\s*\d+.*?learning rate:\s*([\d.eE+\-]+)',line)
        if m:sdk[int(m[1])]=float(m[2])
    assert set(sdk)==set(range(1,906))
    held=[json.loads(s) for s in (RUN/'main/rank-0-heldout.jsonl').read_text().splitlines() if s]
    fields=['update','loss','lr','sdk_logged_lr','grad_norm','iteration_seconds','global_loss_tokens','global_processed_positions']
    with (OUT/'observed-metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in rows:w.writerow({**{k:r[k] for k in fields if k!='sdk_logged_lr'},'sdk_logged_lr':sdk[r['update']]})
    (OUT/'validation-observations.json').write_text(json.dumps(held,indent=2)+'\n')
    steps=[r['update'] for r in rows];loss=[r['loss'] for r in rows]
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','axes.facecolor':'#f8fafc','axes.grid':True,'grid.alpha':.18})
    fig,axes=plt.subplots(2,2,figsize=(15,9),layout='constrained')
    ax=axes[0,0];ax.plot(steps,loss,color='#60a5fa',alpha=.25,lw=.7,label='Raw native log');ax.plot(steps,rolling(loss),color='#1d4ed8',lw=1.9,label='Trailing 16-step mean');ax.set(title='Training LM Loss',xlabel='Optimizer update',ylabel='Native logged Loss');ax.legend()
    ax=axes[0,1];ax.plot([v['update'] for v in held],[v['heldout_ce'] for v in held],'o-',color='#059669',lw=1.8);ax.set(title='Validation LM Loss — 128-dialogue window',xlabel='Optimizer update',ylabel='Native logged validation Loss',xlim=(0,925));ax.text(.04,.94,'Recorded only at interval evaluations\nNo final-905 validation point',transform=ax.transAxes,va='top',fontsize=10)
    for v in held:ax.annotate(f"{v['heldout_ce']:.4f}",(v['update'],v['heldout_ce']),xytext=(5,7),textcoords='offset points',fontsize=10)
    ax=axes[1,0];ax.plot(steps,[r['lr'] for r in rows],color='#7c3aed',lw=1.8,label='Callback record');ax.plot(steps,[sdk[x] for x in steps],color='#ea580c',lw=1.3,ls='--',label='SDK printed record');ax.set(title='Learning Rate — two distinct recorded streams',xlabel='Optimizer update',ylabel='Learning Rate');ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0));ax.legend();ax.text(.03,.95,'Do not equate either stream with a reconciled\noptimizer-boundary LR measurement.',transform=ax.transAxes,va='top',fontsize=9)
    ax=axes[1,1];norm=[r['grad_norm'] for r in rows];ax.plot(steps,norm,color='#f59e0b',alpha=.35,lw=.8,label='Raw');ax.plot(steps,rolling(norm),color='#b45309',lw=1.6,label='Trailing 16-step mean');ax.set(title='Grad Norm',xlabel='Optimizer update',ylabel='Grad Norm (before native clipping)',yscale='log');ax.legend()
    fig.suptitle('SFT from LM+KD C — final main checkpoint 905',fontsize=19,fontweight='bold')
    fig.supxlabel('Main trajectory only; calibration excluded. Observed logs, not inference quality or benchmark evidence.',fontsize=10)
    fig.savefig(OUT/'sft905-training-summary.png',dpi=150);fig.savefig(OUT/'sft905-training-summary.svg');plt.close(fig)
    summary=dict(updates=905,first16_mean_loss=sum(loss[:16])/16,last16_mean_loss=sum(loss[-16:])/16,assistant_target_exposures=sum(r['global_loss_tokens'] for r in rows),processed_positions_including_padding=sum(r['global_processed_positions'] for r in rows),validation=held,lr_note='Both recorded streams retained; no reconciled optimizer-boundary LR claim.',calibration_included=False)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2));print(OUT/'sft905-training-summary.png')
if __name__=='__main__':main()
