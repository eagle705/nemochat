"""Plot only fully observed, finite native optimizer calls on all eight ranks."""
import argparse
import json
import math
from pathlib import Path
import statistics


def collect(output,limit=None):
    ranks=[]
    for rank in range(8):
        path=output/f'student-rank{rank}-updates.jsonl'
        if not path.exists():return []
        data=path.read_text();lines=data.splitlines()
        if data and not data.endswith('\n'):lines=lines[:-1]
        rows=[json.loads(line) for line in lines]
        if any(row['rank']!=rank or row['update']!=i+1 for i,row in enumerate(rows)):
            raise ValueError('noncontiguous native update ownership')
        ranks.append(rows)
    n=min(map(len,ranks))
    if limit is not None:n=min(n,limit)
    result=[]
    keys={'lm_loss':'lm loss','kd_loss':'logits distillation loss','total_loss':'total loss'}
    for i in range(n):
        rows=[r[i] for r in ranks]
        if any(r['skipped'] or r['grad_norm'] is None or r['grad_norm']<=0 or
               not all(math.isfinite(v) for v in [r['lr_used'],r['grad_norm'],*r['metrics'].values()]) for r in rows):
            raise ValueError('not a finite real optimizer call')
        if any(r['allrank_masked_tokens']!=524288 for r in rows):raise ValueError('position count mismatch')
        if len({r['lr_used'] for r in rows})!=1:raise ValueError('rank LR mismatch')
        result.append(dict(update=i+1,positions=(i+1)*524288,lr_used=rows[0]['lr_used'],
            grad_norm_min=min(r['grad_norm'] for r in rows),grad_norm_max=max(r['grad_norm'] for r in rows),
            seconds_max=max(r['seconds'] for r in rows),
            **{out:statistics.mean(r['metrics'][metric] for r in rows) for out,metric in keys.items()}))
    return result


def summarize_ranks(output,limit):
    summary=[]
    for rank in range(8):
        rows=[json.loads(x) for x in (output/f'student-rank{rank}-updates.jsonl').read_text().splitlines()[:limit]]
        if len(rows)!=limit:raise ValueError('rank summary incomplete')
        summary.append(dict(rank=rank,calls=len(rows),positive_lr_updates=sum(r['lr_used']>0 for r in rows),
            zero_lr_calls=sum(r['lr_used']==0 for r in rows),skipped=sum(r['skipped'] for r in rows),
            nonfinite=sum(not all(math.isfinite(v) for v in [r['grad_norm'],r['lr_used'],*r['metrics'].values()]) for r in rows),
            grad_norm_min=min(r['grad_norm'] for r in rows),grad_norm_max=max(r['grad_norm'] for r in rows),
            final_metrics=rows[-1]['metrics'],lr_last_used=rows[-1]['lr_used']))
    return summary


def render(output,limit=64):
    rows=collect(output,limit)
    if len(rows)<limit:raise ValueError(f'only {len(rows)} all-rank calls; require {limit}')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    x=[r['update'] for r in rows]
    fig,axes=plt.subplots(2,1,figsize=(11,7.5),sharex=True,gridspec_kw={'height_ratios':[1,2]})
    axes[0].plot(x,[r['lr_used'] for r in rows],color='#2962b8',lw=2,label='actual LR used by native optimizer')
    axes[0].set_ylabel('Learning Rate');axes[0].ticklabel_format(axis='y',style='sci',scilimits=(0,0));axes[0].legend(loc='lower right')
    for key,label,color in [('lm_loss','LM','#3267c2'),('kd_loss','KD (Ghost-tail)','#de8126'),('total_loss','Total = 0.5 LM + 0.5 KD','#21824a')]:
        values=[r[key] for r in rows]
        axes[1].plot(x,values,color=color,alpha=.30,lw=1,label=label+' raw')
        rolling=[statistics.mean(values[max(0,i-7):i+1]) for i in range(len(values))]
        axes[1].plot(x,rolling,color=color,lw=2,label=label+' trailing-8 mean')
    axes[1].set_ylabel('Native token-mean training Loss');axes[1].set_xlabel('Native optimizer calls (call 1 uses LR=0)')
    axes[1].legend(ncol=3,fontsize=8,loc='upper center',bbox_to_anchor=(0.5,-0.20))
    for axis in axes:axis.grid(alpha=.22)
    fig.suptitle(f'Offline cached KD — first {limit} all-8-rank optimizer calls\nFull sampler: 468,736 samples; WSD horizon: 3,662 calls (warmup 37, decay 366)',fontsize=12)
    fig.text(.5,.01,f'Observed only; no future Loss prediction. {rows[-1]["positions"]:,} processed positions. Raw loss excludes separate router auxiliary regularizer.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.035,1,.93))
    image=output/f'lr-loss-first{limit}.png';fig.savefig(image,dpi=170);plt.close(fig)
    report=dict(ranks=summarize_ranks(output,limit),all_rank_completed_calls=limit,positive_lr_updates=sum(r['lr_used']>0 for r in rows),
        zero_lr_calls=sum(r['lr_used']==0 for r in rows),skipped=0,nonfinite=0,
        processed_positions=rows[-1]['positions'],schedule_horizon=3662,sampler_budget=[468736,120,8],
        rows=rows,image=str(image),metric_scope='LM/KD/primary Total; native router auxiliary remains enabled but is separate',
        checkpoint_transaction=str(output/f'checkpoint-step{limit}-transaction.json'),
        checkpoint_saved=(output/f'checkpoint-step{limit}-transaction.json').exists())
    path=output/f'lr-loss-first{limit}.json';path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
    return image


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--limit',type=int,default=64)
    args=parser.parse_args();render(args.output,args.limit)
