"""Generate literal, predeclared bilingual continuations and update cards only."""
import argparse
from datetime import datetime
import importlib.util
import json
from pathlib import Path
import re
import time

ROOT=Path('/raid/yoon/nemochat')
OUT=ROOT/'results/midtraining/hf-kd1000-examples-01'
MODEL=ROOT/'checkpoints/exports/nemochat_mid_super35_distill_hf_bf16_01'
REPO='eagle705/nemochat-10BA1B-BF16-mid-super3.5-distill'
REVISION='4f58cd035ba83c415552a24820b62667b2b3cb44'
GENERATION=dict(max_new_tokens=64,do_sample=False,num_beams=1,use_cache=True)
PROMPTS=[
    dict(id='en-geography',language='en',prompt='The capital of France is'),
    dict(id='en-algorithm',language='en',prompt='Binary search is an algorithm that'),
    dict(id='en-code',language='en',prompt='def is_even(n):\n    '),
    dict(id='ko-geography',language='ko',prompt='대한민국의 수도는'),
    dict(id='ko-algorithm',language='ko',prompt='이진 탐색 알고리즘은'),
    dict(id='ko-language-model',language='ko',prompt='인공지능 언어 모델은'),
]


def fenced(text):
    longest=max([len(x) for x in re.findall(r'`+',text)]+[2])
    mark='`'*(longest+1)
    return mark+'text\n'+text+('' if text.endswith('\n') else '\n')+mark+'\n'


def generate():
    import torch
    import transformers
    from transformers import AutoModelForCausalLM,AutoTokenizer
    OUT.mkdir(parents=True,exist_ok=True)
    target=OUT/'generation_examples.json'
    assert not target.exists(), 'Refuse regenerating/cherry-picking an existing sample set'
    torch.set_num_threads(4);torch.manual_seed(17)
    assert not torch.cuda.is_available(), 'CPU-only examples must not compete with the learner'
    start=time.monotonic()
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False)
    model,info=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False,dtype=torch.bfloat16,attn_implementation='eager',output_loading_info=True)
    assert all(not info.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs'])
    model.eval();versions=[p._version for p in model.parameters()];cases=[]
    for prompt in PROMPTS:
        x=tok(prompt['prompt'],return_tensors='pt',add_special_tokens=False);n=x['input_ids'].shape[1]
        with torch.inference_mode():
            y=model.generate(**x,**GENERATION,pad_token_id=tok.eos_token_id,return_dict_in_generate=True,output_scores=True)
        new=y.sequences[0,n:];assert new.numel()>0
        finite=all(torch.isfinite(score).all().item() for score in y.scores)
        assert finite
        case={**prompt,'input_ids':x['input_ids'][0].tolist(),'generated_ids':new.tolist(),'new_tokens':new.numel(),'continuation':tok.decode(new,skip_special_tokens=False,clean_up_tokenization_spaces=False),'full_text':tok.decode(y.sequences[0],skip_special_tokens=False,clean_up_tokenization_spaces=False),'scores_finite':finite,'cache_class':type(getattr(y,'past_key_values',None)).__name__,'eos_generated':bool(new[-1].item()==tok.eos_token_id)}
        cases.append(case);print('ACTUAL_GENERATION',json.dumps(case,ensure_ascii=False),flush=True)
    assert versions==[p._version for p in model.parameters()] and not any(p.grad is not None for p in model.parameters())
    value=dict(status='generated_actual_model',generated_at=datetime.now().astimezone().isoformat(),repo_id=REPO,model_revision=REVISION,source_native_step=1000,device='cpu',dtype='bfloat16',transformers=transformers.__version__,torch=torch.__version__,trust_remote_code=False,chat_template=False,add_special_tokens=False,generation=GENERATION,token_decode='skip_special_tokens=False, clean_up_tokenization_spaces=False',selection_policy='All six prompts declared before generation; one greedy generation each, no retries/selection/editing',parameters_unchanged=True,cases=cases,elapsed_seconds=time.monotonic()-start)
    with target.open('x') as f:json.dump(value,f,indent=2,ensure_ascii=False)
    print('EXAMPLES_COMPLETE',str(target),flush=True)


def section(value,korean=False):
    assert value['status']=='generated_actual_model' and len(value['cases'])==6
    assert value['generation']==GENERATION and value['model_revision']==REVISION
    heading='## 실제 영어·한국어 이어쓰기 예시' if korean else '## Actual English and Korean continuation examples'
    text='\n\n'+heading+'\n\n'
    if korean:
        text+='동일 KD step1000 HF 모델을 CPU에서 실제 실행한 결과입니다. **Base 모델이며 instruction/chat 모델이 아닙니다.** 영어3개·한국어3개 prompt를 미리 고정하고 각각 한 번만 greedy 생성했습니다. 답이 틀리거나 반복·언어 혼합이 있어도 수정하거나 더 좋은 출력으로 교체하지 않았습니다. 정답 예시나 benchmark 점수가 아닙니다.\n\n'
    else:
        text+='These are **actual CPU generations from this KD step1000 checkpoint**, not reference answers or benchmark scores. This is a **base completion model, not an instruction/chat model**. All three English and three Korean prompts were declared before one greedy generation per prompt. Incorrect, repetitive or mixed-language continuations are retained; no retries, output editing or best-of selection. A short prompt can continue into document-like text rather than answer conversationally.\n\n'
    if korean:
        text+='**이번 고정 예시에서 관측된 실패:** 영어 출력에는 반복이 있었고, 코드 출력은 올바른 `is_even` 구현이 아닙니다. 한국어 세 prompt 모두 `1000...` 숫자 반복으로 이어졌습니다. 이를 성공적인 한국어 응답으로 제시하지 않습니다.\n\n'
    else:
        text+='**Observed failures in this fixed sample batch:** English continuations repeat; the code continuation is not a correct `is_even` implementation; all three Korean prompts enter a `1000...` numeric repetition loop. These are not successful Korean answers or evidence of reliable bilingual instruction following.\n\n'
    text+='Settings: Transformers '+value['transformers']+', BF16 CPU, `trust_remote_code=False`, eager attention; `max_new_tokens=64`, `do_sample=False`, `num_beams=1`, `use_cache=True`; no chat template or extra BOS. The output below is **only newly generated tokens**, decoded with special tokens preserved. The 64-token cap can end mid-sentence.\n\n'
    text+='[Machine-readable prompts, token IDs and exact outputs](examples/generation_examples.json). Weights were fixed at revision `'+REVISION+'`; this update changes documentation only.\n\n'
    for row in value['cases']:
        assert row['scores_finite'] and row['new_tokens']>0
        text+='### '+row['id']+' ('+str(row['new_tokens'])+' new tokens)\n\n**Input**\n\n'+fenced(row['prompt'])+'\n**Output — continuation only**\n\n'+fenced(row['continuation'])+'\n'
    return text


def publish():
    import os
    os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'
    from huggingface_hub import HfApi,CommitOperationAdd,hf_hub_download
    spec=importlib.util.spec_from_file_location('publication',ROOT/'midtraining/publication/hf_kd1000/publish.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    token=m.credential();api=HfApi(token=token);before=api.repo_info(REPO,files_metadata=True)
    assert before.private is True and before.sha==REVISION, 'Remote changed; review and merge rather than clobber'
    value=json.loads((OUT/'generation_examples.json').read_text());assert value['repo_id']==REPO
    updates={}
    for name,ko in [('README.md',False),('README.ko.md',True)]:
        p=Path(hf_hub_download(REPO,filename=name,revision=REVISION,token=token,local_dir=OUT/'original-card-readback'))
        original=p.read_text();assert ('## 실제 영어·한국어 이어쓰기 예시' if ko else '## Actual English and Korean continuation examples') not in original
        target=OUT/name;target.write_text(original+section(value,ko));updates[name]=target
    updates['examples/generation_examples.json']=OUT/'generation_examples.json'
    commit=api.create_commit(REPO,repo_type='model',revision='main',parent_commit=REVISION,operations=[CommitOperationAdd(path_in_repo=name,path_or_fileobj=str(path)) for name,path in updates.items()],commit_message='Add six actual English/Korean greedy continuation examples with generation provenance',create_pr=False)
    (OUT/'commit_returned.json').write_text(json.dumps(dict(repo_id=REPO,revision=commit.oid,parent_revision=REVISION))+'\n')
    after=api.repo_info(REPO,revision=commit.oid,files_metadata=True);assert after.private and after.sha==commit.oid
    previous={x.rfilename:x.size for x in before.siblings};expected={**previous,**{name:path.stat().st_size for name,path in updates.items()}};assert {x.rfilename:x.size for x in after.siblings}==expected
    for name,path in updates.items():
        remote=Path(hf_hub_download(REPO,filename=name,revision=commit.oid,token=token,local_dir=OUT/'final-readback'));assert remote.read_bytes()==path.read_bytes()
    # Check every weight LFS object remains the exact previous object; no payload reads.
    oldweights={x.rfilename:dict(x.lfs) for x in before.siblings if x.rfilename.endswith('.safetensors')}
    newweights={x.rfilename:dict(x.lfs) for x in after.siblings if x.rfilename.endswith('.safetensors')}
    assert oldweights==newweights and len(oldweights)==12
    receipt=dict(status='published_and_verified',repo_id=REPO,private=True,revision=commit.oid,parent_revision=REVISION,examples=len(value['cases']),english=3,korean=3,all_outputs_actual=True,weights_unchanged_lfs_metadata=True,readback_exact=True,updated_files=list(updates),remote_weights_redownloaded=False)
    (OUT/'publication_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['generate','publish']);a=p.parse_args()
    generate() if a.action=='generate' else publish()
