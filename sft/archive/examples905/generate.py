"""One full official HF load; six frozen manual prompts, once each."""
import json,time,importlib.metadata as im
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer,GenerationConfig
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'results/midtraining/sft905-examples-01'
MODEL=ROOT/'checkpoints/exports/nemochat_sft905_hf_bf16_01'
def safe(v):
    if isinstance(v,dict):return {k:safe(x) for k,x in v.items()}
    if isinstance(v,(set,tuple,list)):return [safe(x) for x in v]
    return v
if __name__=='__main__':
    start=time.monotonic();torch.set_num_threads(4);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    assert not (OUT/'examples.json').exists() and not (OUT/'generations.jsonl').exists()
    frozen=json.loads((OUT/'prompts-predeclared.json').read_text());export=json.loads((OUT/'export_result.json').read_text())
    assert export['source_step']==905 and export['all_source_model_keys_mapped']
    original=Path(export['tokenizer_path'])
    for name in ['tokenizer.json','tokenizer_config.json','chat_template.jinja']:
        assert (original/name).read_bytes()==(MODEL/name).read_bytes()
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False)
    eot=tok.convert_tokens_to_ids('<|im_end|>');assert tok.eos_token_id==eot and tok.eos_token=='<|im_end|>'
    plan=[]
    for case in frozen['cases']:
        messages=[{'role':'user','content':case['prompt']}]
        ids=tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=False)
        rendered=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        assert rendered.endswith('<|im_start|>assistant\n<think></think>')
        plan.append(dict(case,messages=messages,prompt_ids=ids,rendered_prompt=rendered))
    config=GenerationConfig.from_pretrained(MODEL,local_files_only=True)
    inherited=config.to_dict();config.update(do_sample=False,max_new_tokens=128,eos_token_id=eot,pad_token_id=tok.pad_token_id,use_cache=True,num_beams=1)
    (OUT/'generation-plan-before-load.json').write_text(json.dumps({'cases':plan,'inherited_generation_config':inherited,'effective_generation_config':config.to_dict(),'seed':42,'official_eot':{'token':tok.eos_token,'id':eot},'template_mode':'official apply_chat_template enable_thinking=False'},ensure_ascii=False,indent=2))
    model,info=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False,dtype=torch.bfloat16,output_loading_info=True,attn_implementation='eager',device_map={'':'cuda:0'})
    info=safe(info);assert not any(info.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']),info
    assert type(model).__module__=='transformers.models.nemotron_h.modeling_nemotron_h'
    assert im.version('transformers')=='5.12.1'
    biases={k:t for k,t in model.state_dict().items() if 'e_score_correction_bias' in k}
    assert len(biases)==23 and all(t.dtype==torch.float32 for t in biases.values())
    for name,t in model.state_dict().items():assert torch.isfinite(t).all(),name
    model.eval();load_seconds=time.monotonic()-start
    (OUT/'load-receipt.json').write_text(json.dumps({'status':'full_load_passed','loading_info':info,'all_reloaded_tensors_finite':True,'class':type(model).__module__+'.'+type(model).__name__,'parameter_numel':sum(p.numel() for p in model.parameters()),'load_seconds':load_seconds,'router_buffers_fp32':len(biases)},indent=2))
    rows=[]
    with torch.inference_mode():
        for case in plan:
            batch={'input_ids':torch.tensor([case['prompt_ids']],device='cuda'),'attention_mask':torch.ones((1,len(case['prompt_ids'])),dtype=torch.long,device='cuda')}
            t=time.monotonic();torch.cuda.synchronize()
            result=model.generate(**batch,generation_config=config,return_dict_in_generate=True,output_scores=True)
            torch.cuda.synchronize();elapsed=time.monotonic()-t
            assert all(torch.isfinite(s).all() for s in result.scores)
            ids=result.sequences[0,len(case['prompt_ids']):].tolist()
            row=dict(case,generated_ids=ids,generated_text=tok.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False),generated_text_skip_special_tokens=tok.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False),new_tokens=len(ids),eos_reached=bool(ids and ids[-1]==eot),generation_scores_finite=True,generation_seconds=elapsed,cache_class=type(result.past_key_values).__name__)
            with (OUT/'generations.jsonl').open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
            rows.append(row);print(json.dumps(row,ensure_ascii=False),flush=True)
    report={'status':'passed_six_once_actual_generations','source':export['source'],'source_step':905,'native_generation':export['native_generation'],'initial_source_C':export['initial_source_C'],'model':str(MODEL),'tokenizer_path':str(original),'tokenizer_revision':original.name,'tokenizer_assets_exact':True,'official_transformers':im.version('transformers'),'torch':torch.__version__,'class':type(model).__module__+'.'+type(model).__name__,'trust_remote_code':False,'precision':'BF16 weights with 23 FP32 router correction buffers','backend':'official Transformers eager attention and official PyTorch Mamba fallback use_mamba_kernels=False','device':'cuda:0 (physical GPU0 only)','gpu_name':torch.cuda.get_device_name(0),'seed':42,'loading_info':info,'load_seconds':load_seconds,'elapsed_seconds':time.monotonic()-start,'generation_config':config.to_dict(),'source_integrity':'not_checked_explicit_user_waiver','native_HF_logit_parity':'not_run','quality_scope':'six_manual_heldout_prompts_not_a_benchmark','cases':rows}
    (OUT/'examples.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print('SIX_ONCE_COMPLETE',flush=True)
