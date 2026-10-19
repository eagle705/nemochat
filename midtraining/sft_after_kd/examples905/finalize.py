"""Read back exact Engine targets; assemble reports from real generation JSON."""
import json,sys,subprocess,time
from pathlib import Path
ROOT=Path('/raid/yoon/nemochat');OUT=ROOT/'results/midtraining/sft905-examples-01';MODEL=ROOT/'checkpoints/exports/nemochat_sft905_hf_bf16_01'
sys.path.insert(0,str(ROOT/'results/pretrain_followup'))
from engine_helper import request
if __name__=='__main__':
    # Label the untouched upstream tokenizer card accurately, not as this model's card.
    if (MODEL/'README.md').exists():
        assert not (MODEL/'TOKENIZER_UPSTREAM_README.md').exists()
        (MODEL/'README.md').rename(MODEL/'TOKENIZER_UPSTREAM_README.md')
    examples=json.loads((OUT/'examples.json').read_text());rows=examples['cases']
    planned=json.loads((OUT/'prompts-predeclared.json').read_text())['cases']
    stream=[json.loads(s) for s in (OUT/'generations.jsonl').read_text().splitlines()]
    assert len(rows)==len(stream)==len(planned)==6
    assert rows==stream and [x['id'] for x in rows]==[x['id'] for x in planned]
    assert all(a['prompt']==b['prompt'] for a,b in zip(rows,planned))
    assert all(r['new_tokens']==len(r['generated_ids'])==128 and r['generation_scores_finite'] for r in rows)
    engines={}
    for name in ['nemochat-sft905-metadata-01','nemochat-sft905-schema-01','nemochat-sft905-export-01']:
        created=json.loads((OUT/(name+'.create.json')).read_text());cid=created['id'];actual=request('GET',f'/containers/{cid}/json')
        state=actual['State'];assert state['ExitCode']==0 and not state['Running'] and not state['OOMKilled']
        assert not actual['HostConfig'].get('DeviceRequests')
        engines[name]={'id':cid,'state':state,'image':actual['Config']['Image']}
    created=json.loads((OUT/'generation-engine.create.json').read_text());cid=created['id'];actual=request('GET',f'/containers/{cid}/json');state=actual['State']
    assert state['ExitCode']==0 and not state['Running'] and not state['OOMKilled']
    devices=actual['HostConfig']['DeviceRequests'];assert len(devices)==1 and devices[0]['DeviceIDs']==['0'] and devices[0]['Count']==0 and devices[0]['Driver']=='nvidia' and devices[0]['Capabilities']==[['gpu']]
    engines['generation']={'id':cid,'state':state,'image':actual['Config']['Image'],'device_requests':devices}
    trained=request('GET','/containers/33f23666c0ab27f8fb758d08986a55dbb464d2da92cbd11859508d3845a4a28e/json')['State']
    assert not trained['Running'] and trained['ExitCode']==0 and not trained['OOMKilled']
    load=json.loads((OUT/'load-receipt.json').read_text());assert not any(load['loading_info'].values())
    index=json.loads((MODEL/'model.safetensors.index.json').read_text());assert len(index['weight_map'])==6289
    judgments={
      'en_sky':{'en':'Starts with a relevant Rayleigh-scattering explanation, but violates one-sentence instruction, becomes repetitive, and is cut off at the cap.','ko':'Rayleigh scattering 설명으로 시작하지만 한 문장 지시를 어기고 반복하며 토큰 상한에서 잘렸습니다.'},
      'ko_sky':{'en':'Fails to explain the sky color; nonsensical Korean repetition.','ko':'하늘 색을 설명하지 못하고 무의미한 한국어를 반복합니다.'},
      'en_python':{'en':'Contains the correct evenness expression initially, but repeats definitions, uses a Markdown fence despite code-only, and ends in incomplete code.','ko':'처음에는 올바른 짝수 판별식을 쓰지만 함수 정의를 반복하고 코드만 출력하라는 지시 대신 Markdown fence를 쓰며 불완전하게 끝납니다.'},
      'ko_python':{'en':'Does not provide the requested function; repetitive English prose/list, not code-only.','ko':'요청한 함수를 제공하지 못하고 영어 설명과 목록을 반복하므로 코드만 출력 조건도 실패합니다.'},
      'en_json':{'en':'Contains the requested values inside a JSON block, but adds prose, Markdown fences and a repetitive table; the complete response is not JSON-only.','ko':'JSON 블록 안의 값은 요청과 일치하지만 설명, Markdown fence, 반복 표를 추가해 응답 전체는 JSON-only가 아닙니다.'},
      'ko_json':{'en':'Does not output a JSON object; repetitive Korean phrase.','ko':'JSON 객체를 출력하지 않고 한국어 구절을 반복합니다.'}}
    for lang in ['en','ko']:
        title='# SFT905 actual six-response examples' if lang=='en' else '# SFT905 실제 6개 응답 예시'
        intro=('Actual final native SFT905, not C/base/calibration. Each frozen prompt was generated once; continuations below are full and unedited. All six reached the 128-token cap without official EOS/EOT. These are manual examples, not a benchmark.' if lang=='en' else 'C/base/calibration이 아닌 실제 최종 native SFT905입니다. 고정된 프롬프트를 각각 한 번만 생성했으며 아래 continuation은 전체 원문입니다. 6개 모두 공식 EOS/EOT 없이 128-token 상한에 도달했습니다. 수동 예시이며 benchmark가 아닙니다.')
        parts=[title,'',intro,'',f"Source: `{examples['source']}`",f"Local HF: `{MODEL}`",f"Tokenizer revision: `{examples['tokenizer_revision']}`",'Runtime: official Transformers 5.12.1; trust_remote_code=False; BF16 + FP32 router buffers; GPU0; eager attention; official PyTorch Mamba fallback; seed42; greedy; max_new_tokens=128.','Template: official apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False).',f"Generation Engine: `{cid}`, exit0, OOMKilled=false.",'Source payload integrity: not_checked_explicit_user_waiver. Native/HF logit parity: not run. Operational HF compatibility only.','']
        for row in rows:
            parts += [f"## {row['id']}",'','Prompt:','',row['prompt'],'','Full unedited continuation:','','````text',row['generated_text'],'````','',judgments[row['id']][lang],'']
        (OUT/f'examples.{lang}.md').write_text('\n'.join(parts))
    receipt={'status':'complete_actual_FINAL_SFT905_full_HF_reload_six_once','time':time.time(),'source':examples['source'],'source_step':905,'initial_source_C':examples['initial_source_C'],'native_generation':examples['native_generation'],'training_engine':{'id':'33f23666c0ab27f8fb758d08986a55dbb464d2da92cbd11859508d3845a4a28e','state':trained},'engines':engines,'native_model_records':619,'HF_runtime_records':447,'serialized_records':len(index['weight_map']),'parameter_numel':load['parameter_numel'],'loading_info':load['loading_info'],'model':str(MODEL),'tokenizer_revision':examples['tokenizer_revision'],'generation_case_count':len(rows),'total_generated_tokens':sum(r['new_tokens'] for r in rows),'eos_reached_count':sum(r['eos_reached'] for r in rows),'once_no_retries':True,'judgments':judgments,'source_integrity':'not_checked_explicit_user_waiver','native_HF_logit_parity':'not_run','training_or_remote_write_performed':False,'files':{p.name:p.stat().st_size for p in MODEL.iterdir() if p.is_file()},'postrun_doc_labeling':'Copied upstream tokenizer README renamed TOKENIZER_UPSTREAM_README.md; converter source updated for accurate future naming; no weights/config/tokenizer runtime assets modified.'}
    (OUT/'final-acceptance.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2));print(json.dumps(receipt,ensure_ascii=False,indent=2))
    print(subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv'],text=True))
