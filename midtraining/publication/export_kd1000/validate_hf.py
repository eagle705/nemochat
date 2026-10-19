"""Full offline official Transformers CPU reload, bilingual forward/cache gate."""
import argparse, importlib.metadata as im, json, os, platform, time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
ROOT=Path('/raid/yoon/nemochat')
DEFAULT_MODEL=ROOT/'checkpoints/exports/nemochat_mid_super35_distill_hf_bf16_01'
DEFAULT_RESULT=ROOT/'results/midtraining/hf-kd1000-conversion-01/validation_result.json'
ORIGINAL=ROOT/'checkpoints/exports/nemochat_baby_base_bf16/tokenizer'
def main(model_path=DEFAULT_MODEL,result_path=DEFAULT_RESULT,original_path=ORIGINAL,export_result=None):
    start=time.monotonic();torch.set_num_threads(4);torch.manual_seed(17)
    model_path=Path(model_path).resolve();result_path=Path(result_path)
    assert not result_path.exists(),'fresh validation result required'
    export_result=Path(export_result) if export_result else result_path.parent/'export_result.json'
    export_receipt=json.loads(export_result.read_text())
    assert Path(export_receipt['output']).resolve()==model_path
    assert export_receipt['all_source_model_keys_mapped'] and export_receipt['all_source_tensors_finite']
    assert not torch.cuda.is_available(),'GPU must not be visible'
    assert not any(model_path.glob('*.py')),'no hub Python code allowed'
    original_path=Path(original_path)
    for asset in original_path.iterdir():
        if asset.is_file():assert asset.read_bytes()==(model_path/asset.name).read_bytes(),asset.name
    original=AutoTokenizer.from_pretrained(original_path,local_files_only=True,trust_remote_code=False)
    tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True,trust_remote_code=False)
    assert tokenizer.get_vocab()==original.get_vocab()
    texts=['The capital of France is','대한민국의 수도는']
    assert all(tokenizer.encode(s,add_special_tokens=False)==original.encode(s,add_special_tokens=False) for s in texts)
    print('Full official AutoModelForCausalLM reload',model_path,flush=True)
    model,info=AutoModelForCausalLM.from_pretrained(model_path,local_files_only=True,trust_remote_code=False,dtype=torch.bfloat16,output_loading_info=True,attn_implementation='eager')
    info={k:sorted(v) if isinstance(v,set) else v for k,v in info.items()}
    assert not info.get('missing_keys') and not info.get('unexpected_keys') and not info.get('mismatched_keys') and not info.get('error_msgs'),info
    assert type(model).__module__=='transformers.models.nemotron_h.modeling_nemotron_h'
    assert all(t.device.type=='cpu' for t in model.state_dict().values())
    for key,tensor in model.state_dict().items():
        if not torch.isfinite(tensor).all():raise ValueError('nonfinite reloaded tensor '+key)
    biases={k:t for k,t in model.state_dict().items() if 'e_score_correction_bias' in k}
    assert len(biases)==23 and all(t.dtype==torch.float32 for t in biases.values())
    model.eval();load_seconds=time.monotonic()-start;rows=[]
    with torch.inference_mode():
        for text in texts:
            batch=tokenizer(text,return_tensors='pt',add_special_tokens=False)
            t=time.monotonic();out=model(**batch,use_cache=False,logits_to_keep=0)
            assert torch.isfinite(out.logits).all() and out.logits.shape[:2]==batch['input_ids'].shape
            logits=out.logits.float();top=logits[0,-1].topk(5)
            row={'prompt':text,'input_ids':batch['input_ids'].tolist(),'shape':list(logits.shape),'finite':True,'logits_min':logits.min().item(),'logits_max':logits.max().item(),'logits_mean':logits.mean().item(),'top5_ids':top.indices.tolist(),'top5_logits':top.values.tolist(),'forward_seconds':time.monotonic()-t}
            logits_path=result_path.with_name(result_path.stem+'_hf_logits_'+str(len(rows))+'.pt')
            torch.save({'input_ids':batch['input_ids'],'logits':logits},logits_path)
            row['logits_artifact']=str(logits_path.resolve())
            t=time.monotonic();generation=model.generate(**batch,max_new_tokens=4,do_sample=False,use_cache=True,pad_token_id=original.pad_token_id if original.pad_token_id is not None else original.eos_token_id,return_dict_in_generate=True,output_scores=True)
            assert all(torch.isfinite(score).all() for score in generation.scores)
            assert generation.past_key_values is not None
            generated=generation.sequences
            assert generated.shape[1]>batch['input_ids'].shape[1]
            row.update({'generation_scores_finite':True,'generated_ids':generated.tolist(),'generated_text':tokenizer.decode(generated[0],skip_special_tokens=False),'new_tokens':generated.shape[1]-batch['input_ids'].shape[1],'generation_seconds':time.monotonic()-t,'generation_cache':True,'cache_class':type(generation.past_key_values).__name__})
            print(json.dumps(row,ensure_ascii=False),flush=True);rows.append(row)
    report={'status':'passed','source':export_receipt['source'],'source_step':export_receipt['source_step'],'model_prefix':export_receipt['model_prefix'],'export_receipt':str(export_result.resolve()),'model':str(model_path),'class':type(model).__module__+'.'+type(model).__name__,'official_transformers':True,'trust_remote_code':False,'device':'cpu','cuda_visible':torch.cuda.is_available(),'full_model_reload':True,'loading_info':info,'all_reloaded_tensors_finite':True,'tokenizer_bytes_exact_match':True,'tokenizer_vocab_exact_match':True,'torch':torch.__version__,'versions':{p:im.version(p) for p in ['transformers','megatron-bridge','megatron-core','mamba-ssm','safetensors']},'python':platform.python_version(),'load_seconds':load_seconds,'elapsed_seconds':time.monotonic()-start,'parameter_numel':sum(p.numel() for p in model.parameters()),'router_bias_dtypes':sorted(set(str(b.dtype) for b in biases.values())),'router_correction_count':len(biases),'cases':rows,'native_forward_parity':'not_run_no_GPU_allocation_to_preserve_LM_only_priority','source_integrity':'not_checked_explicit_user_waiver'}
    result_path.write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',type=Path,default=DEFAULT_MODEL);p.add_argument('--result',type=Path,default=DEFAULT_RESULT);p.add_argument('--original-tokenizer',type=Path,default=ORIGINAL);p.add_argument('--export-result',type=Path);a=p.parse_args();main(a.model,a.result,a.original_tokenizer,a.export_result)
