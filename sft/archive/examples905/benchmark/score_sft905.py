"""Use the frozen scorer unchanged, adapting only its full-vocabulary identity guard.

The SFT tokenizer intentionally renames reserved IDs18..29. Require exact ordinary
IDs and every frozen prompt/choice encoding; never replace the migrated tokenizer.
"""
import importlib.util,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];OUT=ROOT/'results/midtraining/sft905-benchmark-gpu-01';MODEL=ROOT/'checkpoints/exports/nemochat_sft905_hf_bf16_01'
CODE=ROOT/'midtraining/evaluation/ablation_gpu/eval.py'
spec=importlib.util.spec_from_file_location('frozen_gpu_scorer',CODE);evaluation=importlib.util.module_from_spec(spec);spec.loader.exec_module(evaluation)
original_read_suite=evaluation.read_suite

def migrated_guard(path):
    from transformers import AutoTokenizer
    suite=original_read_suite(path)
    assert suite['suite_sha256']=='07609a0cf53e4a63d555a3e50938adfed1cb05471d02b37fb93461e7879af19e'
    old=AutoTokenizer.from_pretrained(evaluation.BASE,local_files_only=True,trust_remote_code=False)
    actual=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False)
    oldv=old.get_vocab();newv=actual.get_vocab()
    assert evaluation.sha(evaluation.canonical(oldv))==suite['tokenizer_vocab_sha256']
    oi={v:k for k,v in oldv.items()};ni={v:k for k,v in newv.items()}
    assert len(oi)==len(oldv)==len(ni)==len(newv)==131072 and set(oi)==set(ni)
    changed=sorted(i for i in oi if oi[i]!=ni[i]);assert changed==list(range(18,30)),changed
    choices=0
    for row in suite['questions']:
        for choice,encoded in zip(row['choices'],row['encoded']):
            c,t=evaluation.pair(actual,row['prompt'],' '+choice)
            assert c==encoded['context_ids'] and t==encoded['continuation_ids'],row['id']
            choices+=1
    assert choices==1600
    assert all(not(set(w['ids']) & set(changed)) for w in suite['heldout'])
    migrated_hash=evaluation.sha(evaluation.canonical(newv))
    receipt={'status':'all_frozen_raw_prompt_choice_encodings_identical','suite_sha256':suite['suite_sha256'],'original_vocab_sha256':suite['tokenizer_vocab_sha256'],'migrated_vocab_sha256':migrated_hash,'allowed_and_actual_changed_ids':changed,'aliases':{i:{'old':oi[i],'new':ni[i]} for i in changed},'all_ordinary_token_IDs_identical':True,'encoded_choices_checked':choices,'heldout_contains_renamed_reserved_ids':False,'chat_template_applied':False,'scorer':str(CODE),'adapter':'Only replace full-vocab equality-guard expected hash in memory after exhaustive alias and frozen-encoding proof. Original suite file and scoring/aggregation functions remain unchanged.'}
    (OUT/'tokenizer-compatibility.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
    suite['tokenizer_vocab_sha256']=migrated_hash
    return suite

if __name__=='__main__':
    assert '--model' in sys.argv and sys.argv[sys.argv.index('--model')+1]==str(MODEL)
    export=json.loads((ROOT/'results/midtraining/sft905-examples-01/export_result.json').read_text());assert export['source_step']==905
    evaluation.read_suite=migrated_guard
    evaluation.main()
