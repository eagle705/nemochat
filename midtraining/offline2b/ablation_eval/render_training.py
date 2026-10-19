"""Read-only, same-step LM-only versus KD observed LR / LM Loss comparison."""
import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import statistics


def complete_rows(path):
    text = path.read_text()
    lines = text.splitlines()
    if text and not text.endswith('\n'):
        lines = lines[:-1]
    return [json.loads(line) for line in lines]


def snapshot(root):
    root = Path(root)
    b = root / 'results/midtraining/offline2b-lmonly-1000-02'
    c = root / 'results/midtraining/offline2b-prefixtrain-1000-01'
    br = [complete_rows(b/f'student-rank{i}-updates.jsonl') for i in range(8)]
    cr = [complete_rows(c/f'student-rank{i}-updates.jsonl') for i in range(8)]
    n = min([len(x) for x in br] + [len(x) for x in cr])
    assert 0 < n <= 1000
    verified = 0
    for rank in range(8):
        for rows in (br[rank], cr[rank]):
            assert [x['update'] for x in rows[:n]] == list(range(1, n+1))
            for i, row in enumerate(rows[:n]):
                assert not row['skipped'] and row['grad_norm'] > 0
                assert all(math.isfinite(x) for x in [row['grad_norm'], row['lr_used'], *row['metrics'].values()])
                assert row['physical_iteration'] == cr[0][i]['physical_iteration']
                assert row['lr_used'] == cr[0][i]['lr_used']
                assert row['allrank_masked_tokens'] == 524288
        bs = complete_rows(b/f'student-rank{rank}-samples.jsonl')[:n*16]
        cs = complete_rows(c/f'student-rank{rank}-samples.jsonl')[:n*16]
        assert len(bs) == len(cs) == n*16
        for x, y in zip(bs, cs):
            assert all(x[k] == y[k] for k in ['micro','iteration','rank','sha256','masked_tokens','logical_iteration','physical_iteration'])
            verified += 1
    for i in range(n):
        assert all(x[i]['metrics'] == br[0][i]['metrics'] for x in br)
        assert all(x[i]['metrics'] == cr[0][i]['metrics'] for x in cr)
    return dict(snapshot_at=datetime.now().astimezone().isoformat(), calls=n, B=br[0][:n], C=cr[0][:n], C_completed_calls=min(map(len, cr)), input_digest_rows_verified=verified, max_lr_difference=max(abs(x['lr_used']-y['lr_used']) for x,y in zip(br[0][:n],cr[0][:n])), positions_per_arm=n*524288, B_source=str(b), C_source=str(c))


def render(data, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    n = data['calls']; steps = np.arange(1,n+1)
    b = np.array([r['metrics']['lm loss'] for r in data['B']])
    c = np.array([r['metrics']['lm loss'] for r in data['C']])
    lr = np.array([r['lr_used'] for r in data['B']])
    smooth = lambda a: np.array([np.mean(a[max(0,i-15):i+1]) for i in range(len(a))])
    sb, sc = smooth(b), smooth(c)
    fig, ax = plt.subplots(3,1,figsize=(13,11),sharex=True,gridspec_kw={'height_ratios':[1,2,1.2]},layout='constrained')
    blue, orange = '#2369bd', '#d78621'
    ax[0].plot(steps, lr, color=blue, linewidth=2.5, label='B: actual LR used')
    ax[0].plot(steps, [r['lr_used'] for r in data['C']], color=orange, linestyle='--',linewidth=1.6,label='C: actual LR used (identical)')
    ax[0].axvspan(1,min(38,n),color='#e8edf9',alpha=.8)
    ax[0].set_ylabel('Learning Rate'); ax[0].ticklabel_format(axis='y',style='sci',scilimits=(0,0)); ax[0].set_ylim(bottom=0); ax[0].legend(loc='lower right',fontsize=10)
    ax[1].plot(steps,b,color=blue,alpha=.2,linewidth=.7)
    ax[1].plot(steps,c,color=orange,alpha=.2,linewidth=.7)
    ax[1].plot(steps,sb,color=blue,linewidth=2,label='B: LM-only (1.0 LM), 16-call mean')
    ax[1].plot(steps,sc,color=orange,linewidth=2,label='C: KD (0.5 LM + 0.5 KD), 16-call mean')
    ax[1].set_ylabel('Unweighted Training LM Loss (native recorded scalar)'); ax[1].legend(loc='upper right',fontsize=10)
    ax[2].plot(steps,sb-sc,color='#7954a4',linewidth=1.8,label='16-call mean difference: B LM Loss minus C LM Loss')
    ax[2].axhline(0,color='#666666',linestyle='--',linewidth=1)
    ax[2].fill_between(steps,0,sb-sc,where=(sb>=sc),color=orange,alpha=.13)
    ax[2].fill_between(steps,0,sb-sc,where=(sb<sc),color=blue,alpha=.13)
    ax[2].set_ylabel('LM Loss difference (B - C)')
    ax[2].set_xlabel('Same accepted native optimizer call / same physical data order')
    ax[2].legend(loc='upper left',fontsize=9)
    ax[2].text(.99,.18,'Above zero: KD has lower Training LM Loss; below zero: LM-only has lower.',transform=ax[2].transAxes,ha='right',va='bottom',fontsize=8)
    for a in ax: a.grid(alpha=.18); a.set_xlim(1,max(n,2))
    fig.suptitle(f"LM-only vs KD: matched calls 1–{n} | B {n}/1000, C completed {data['C_completed_calls']}/1000\n{data['snapshot_at'][:19]} KST | faint: raw; solid: trailing16 means | Training Loss is NOT a benchmark",fontsize=12)
    fig.savefig(out/'lr_lm_comparison.png',dpi=160);fig.savefig(out/'lr_lm_comparison.svg');plt.close(fig)
    rows=[]
    for i,(x,y) in enumerate(zip(data['B'],data['C'])):
        rows.append(dict(update=i+1,physical_iteration=x['physical_iteration'],lr_B=x['lr_used'],lr_C=y['lr_used'],lm_B=float(b[i]),lm_C=float(c[i]),lm_B_minus_C=float(b[i]-c[i]),rolling16_B=float(sb[i]),rolling16_C=float(sc[i])))
    with (out/'observed.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    window=min(16,n); mb=float(np.mean(b[-window:]));mc=float(np.mean(c[-window:]))
    summary={k:v for k,v in data.items() if k not in ['B','C']}
    summary.update(last16=dict(window_calls=window,B=dict(lm_loss=mb),C=dict(lm_loss=mc),B_minus_C=mb-mc,C_reduction_vs_B_percent=(mb-mc)/mb*100),last_raw=dict(B=float(b[-1]),C=float(c[-1])),actual_lr=float(lr[-1]),positive_B=sum(x['lr_used']>0 for x in data['B']),native_skipped=0,interpretation='Matched training inputs and steps; ordinary recipe comparison, not fixed-LM-coefficient isolation, not heldout or benchmark evidence.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (out/'observed.json').write_text(json.dumps(rows,indent=2)+'\n')
    return summary


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('/raid/yoon/nemochat'));p.add_argument('--out',type=Path,required=True);a=p.parse_args();print(json.dumps(render(snapshot(a.root),a.out),indent=2))
