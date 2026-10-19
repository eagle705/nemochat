"""Verified paired comparison and PNGs. No model execution or GPU access."""
import collections
import math
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import controller as c
sys.path.insert(0,str(c.EVAL))
from compare import compare

ORDER=['A','C','B']
LABELS={'A':'[1] BASE (pre-midtraining)','C':'[2] LM+KD (0.5 LM + 0.5 KD)','B':'[3] LM-only (1.0 LM)'}
IDS='BASE19074_A__KD1000_C__LM1000_B'
CAVEAT='Nonstandard diagnostic subsets: MMLU 10 subjects x 20 test questions (5-shot), ARC-Challenge 100 test (0-shot), HellaSwag 100 validation (0-shot); not full MMLU or leaderboard scores. Benchmark contamination unknown. Fixed heldout: 64 windows, 65,536 target tokens, 16 windows/source. B 1.0 LM vs C 0.5 LM + 0.5 KD is a practical recipe comparison, not coefficient-matched causal KD isolation. Bootstrap uncertainty is over paired questions/windows, not training seeds. Checkpoint SHA/CRC not checked by explicit user waiver.'

def render(reports,suite,out,fixture=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    proofs={a:c.validate_report(reports[a],suite,c.MODELS[a]) for a in ORDER}
    pairs={}
    for left,right in [('A','C'),('A','B'),('B','C')]:
        result=compare(reports[left],reports[right],suite)
        c.require(result['status']=='complete_paired_comparison' and len(result['paired_questions'])==400 and len(result['paired_windows'])==64,'paired comparison incomplete')
        name=f'paired-{left}-vs-{right}.json';c.write(out/name,result);pairs[f'{left}-vs-{right}']=str(out/name)
    groups=[('MMLU diagnostic\n200 questions, 5-shot',lambda q:q['task'].startswith('mmlu_'),'acc'),('ARC-Challenge\n100 questions, 0-shot',lambda q:q['task']=='arc_challenge','acc_norm'),('HellaSwag\n100 questions, 0-shot',lambda q:q['task']=='hellaswag','acc_norm')]
    scores={};sources=sorted({w['source'] for w in suite['heldout']});held={}
    for a in ORDER:
        scores[a]=[sum(q[metric] for q in reports[a]['questions'] if pred(q))/sum(pred(q) for q in reports[a]['questions']) for _,pred,metric in groups]
        held[a]={s:sum(w['nll_sum'] for w in reports[a]['heldout'] if w['source']==s)/sum(w['target_count'] for w in reports[a]['heldout'] if w['source']==s) for s in sources}
        held[a]={'Overall':reports[a]['heldout_full_vocab_nll'],**held[a]}
    title='CPU TEST FIXTURE — NOT MODEL RESULTS' if fixture else 'NemoChat same-suite GPU BF16 evaluation'
    colors=['#4477aa','#ee7733','#228833'];fig,ax=plt.subplots(figsize=(13,7))
    for i,a in enumerate(ORDER):
        bars=ax.bar([j+(i-1)*.24 for j in range(3)],[100*x for x in scores[a]],width=.23,color=colors[i],label=LABELS[a])
        ax.bar_label(bars,fmt='%.1f%%',padding=4,fontsize=10)
    ax.set_xticks(range(3),[x[0] for x in groups]);ax.set_ylim(0,105);ax.set_ylabel('Accuracy (%) — MMLU acc; ARC / HellaSwag acc_norm');ax.set_title(title+'\nBASE update 19,074; midtraining endpoints 1,000 calls');ax.legend(loc='upper right');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.text(.05,.035,'400 questions / 1,600 choices | Not full MMLU | Contamination unknown | GPU Transformers 5.12.1\nPractical recipe comparison, not coefficient-matched KD isolation',fontsize=10)
    fig.subplots_adjust(bottom=.23,top=.85);bench=out/f'benchmark-{IDS}.png';fig.savefig(bench,dpi=140);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(14,10));names=['Overall',*sources]
    for i,a in enumerate(ORDER):
        vals=[held[a][s] for s in names]
        for ax,ys in zip(axes,[vals,[math.exp(x) for x in vals]]):
            bars=ax.bar([j+(i-1)*.24 for j in range(len(names))],ys,width=.23,color=colors[i],label=LABELS[a]);ax.bar_label(bars,fmt='%.3f',padding=3,fontsize=8)
    for ax,label in zip(axes,['Full-vocabulary token NLL (lower is better)','Perplexity exp(NLL) (lower is better)']):
        ax.set_xticks(range(len(names)),['Overall\n65,536 targets',*[s.replace('_',' ')+'\n16,384 targets' for s in sources]],fontsize=9);ax.set_ylabel(label);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.margins(y=.2)
    axes[0].legend(fontsize=9);fig.suptitle(title+'\nIdentical fixed heldout windows: 64 x 1,024 targets; original tokenizer',fontsize=14)
    fig.text(.05,.02,'BASE19074 (A) | LM+KD1000 (C) | ordinary LM-only1000 (B)\nReserved corpus diagnostic, not benchmark accuracy; checkpoint payload SHA/CRC explicitly waived.',fontsize=10);fig.tight_layout(rect=(0,.06,1,.94));hold=out/f'heldout-{IDS}.png';fig.savefig(hold,dpi=140);plt.close(fig)
    images=[str(bench),str(hold)]
    for p in images:
        with Image.open(p) as im:im.verify()
    summary={'status':'CPU_TEST_FIXTURE_NOT_MODEL_RESULTS' if fixture else 'verified_all_three_complete','display_order':ORDER,'labels':LABELS,'model_checkpoints':{a:str(c.MODELS[a]) for a in ORDER},'suite_sha256':suite['suite_sha256'],'coverage':proofs,'benchmark_accuracy':scores,'heldout_nll':held,'heldout_ppl':{a:{s:math.exp(v) for s,v in held[a].items()} for a in ORDER},'paired_reports':pairs,'images':images,'limitations':CAVEAT}
    c.write(out/'summary.json',summary)
    text='# '+title+'\n\n'+CAVEAT+'\n\n| Model / checkpoint | MMLU acc /200 | ARC acc_norm /100 | HellaSwag acc_norm /100 | Heldout NLL /65,536 | PPL |\n|---|---:|---:|---:|---:|---:|\n'
    for a in ORDER:text+='| '+LABELS[a]+' / `'+c.MODELS[a].name+'` | '+' | '.join(f'{100*v:.2f}%' for v in scores[a])+f" | {held[a]['Overall']:.6f} | {math.exp(held[a]['Overall']):.6f} |\n"
    text+='\n## Paired statistics\n'+ '\n'.join(f'- {k}: `{v}`' for k,v in pairs.items())+'\n\n'+ '\n'.join(f'![{Path(p).stem}]({Path(p).name})' for p in images)+'\n'
    (out/'FINAL_REPORT.md').write_text(text)
    (out/'FINAL_REPORT.ko.md').write_text('# 동일 GPU 평가 결과\n\n'+('CPU TEST FIXTURE: 실제 모델 평가 결과가 아닙니다.\n\n' if fixture else '')+'표시 순서: [1] BASE → [2] LM+KD → [3] LM-only. 모든 모델은 동일한 400문항/1,600선택지 및 64개 heldout window/65,536 target token을 사용합니다. 전체 MMLU 또는 공식 leaderboard 결과가 아니며 benchmark contamination은 확인되지 않았습니다. B=1.0 LM과 C=0.5 LM+0.5 KD의 실용 recipe 비교이며 KD의 coefficient-matched 인과 효과가 아닙니다. Checkpoint SHA/CRC는 사용자 요청으로 검사하지 않았습니다.\n\n'+text)
    return summary
