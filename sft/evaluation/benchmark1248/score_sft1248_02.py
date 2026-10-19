"""Use the frozen scorer unchanged, adapting only its full-vocabulary identity guard.

The SFT tokenizer intentionally renames reserved IDs18..29. Require exact ordinary
IDs and every frozen prompt/choice encoding; never replace the migrated tokenizer.
"""
import importlib.util,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'results/midtraining/sft1248-benchmark-gpu-01';MODEL=ROOT/'checkpoints/exports/nemochat_sft1248_hf_bf16_01'
CODE=ROOT/'midtraining/evaluation/ablation_gpu/eval.py'
spec=importlib.util.spec_from_file_location('frozen_gpu_scorer',CODE);evaluation=importlib.util.module_from_spec(spec);spec.loader.exec_module(evaluation)
original_read_suite=evaluation.read_suite

def migrated_guard(path):
    from transformers import AutoTokenizer
    suite=original_read_suite(path)
    assert suite['suite_sha256']=='07609a0cf53e4a63d555a3e50938adfed1cb05471d02b37fb93461e7879af19e'
    old=AutoTokenizer.from_pretrained(ROOT/'checkpoints/exports/nemochat_baby_base_bf16/tokenizer',local_files_only=True,trust_remote_code=False)
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
    receipt={'status':'all_frozen_raw_prompt_choice_encodings_identical','suite_sha256':suite['suite_sha256'],'original_vocab_sha256':suite['tokenizer_vocab_sha256'],'migrated_vocab_sha256':migrated_hash,'allowed_and_actual_changed_ids':changed,'aliases':{i:{'old':oi[i],'new':ni[i]} for i in changed},'all_ordinary_token_IDs_identical':True,'encoded_choices_checked':choices,'heldout_contains_renamed_reserved_ids':False,'chat_template_applied':False,'scorer':str(CODE),'comparison_only_original_tokenizer_path':str(ROOT/'checkpoints/exports/nemochat_baby_base_bf16/tokenizer'),'failed_BASE_export_tokenizer_path':str(evaluation.BASE),'comparison_vocab_identity_proof':'canonical vocabulary SHA must equal immutable suite; evaluated model tokenizer remains migrated SFT1248','adapter':'Only replace full-vocab equality-guard expected hash in memory after exhaustive alias and frozen-encoding proof. Original suite file and scoring/aggregation functions remain unchanged.'}
    (OUT/'tokenizer-compatibility.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
    suite['tokenizer_vocab_sha256']=migrated_hash
    return suite

if __name__=='__main__':
    assert '--model' in sys.argv and sys.argv[sys.argv.index('--model')+1]==str(MODEL)
    export=json.loads((ROOT/'results/midtraining/sft1248-benchmark-gpu-01/export_result.json').read_text());assert export['source_step']==1248
    from transformers import AutoModelForCausalLM
    import torch
    original_load=AutoModelForCausalLM.from_pretrained
    def inspected_load(*args,**kwargs):
        result=original_load(*args,**kwargs)
        model,info=result
        assert str(args[0])==str(MODEL)
        assert sum(p.numel() for p in model.parameters())==9790967416
        assert len(model.model.layers)==52 and model.config.time_step_min==0.0
        state=model.state_dict()
        routers={k:v for k,v in state.items() if 'e_score_correction_bias' in k}
        assert len(routers)==23 and all(v.dtype==torch.float32 for v in routers.values())
        assert len(json.loads((MODEL/'model.safetensors.index.json').read_text())['weight_map'])==6289
        assert len(state)==447 and all(torch.isfinite(v).all() for v in state.values())
        assert not any(info.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs'])
        (OUT/'load-receipt.json').write_text(json.dumps({'model':str(MODEL),'native_source':export['source'],'source_step':1248,'parameter_numel':9790967416,'layers':52,'runtime_tensor_records':len(state),'serialized_tensor_records':len(json.loads((MODEL/'model.safetensors.index.json').read_text())['weight_map']),'router_correction_count':len(routers),'router_correction_dtype':'torch.float32','all_loaded_tensors_finite':True,'time_step_min':model.config.time_step_min,'loading_info':evaluation.loading_metadata(info),'trust_remote_code':False,'native_HF_logit_parity':'not_run'},indent=2))
        return result
    AutoModelForCausalLM.from_pretrained=inspected_load
    evaluation.read_suite=migrated_guard
    evaluation.main()
