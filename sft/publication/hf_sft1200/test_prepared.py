"""Small-asset acceptance only: never hash/read/redownload source weight payloads."""
import ast,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'results/midtraining/hf-sft1200-publication-01';STAGE=ROOT/'checkpoints/exports/nemochat_sft1248_publication_01';MODEL=ROOT/'checkpoints/exports/nemochat_sft1248_hf_bf16_01'
def main():
 expected=json.loads((OUT/'expected-inventory.json').read_text());actual={str(p.relative_to(STAGE)):p.stat().st_size for p in STAGE.rglob('*') if p.is_file()};assert actual==expected
 shards=[n for n in expected if n.endswith('.safetensors')];assert len(shards)==12
 for n in shards:assert (STAGE/n).is_symlink() and (STAGE/n).resolve()==MODEL/n
 assert not any(n.endswith(('.distcp','.pt','.npy')) or any(t in n.lower() for t in ['credential','secret','optimizer','rng_state']) for n in expected)
 manifest=json.loads((STAGE/'publication_manifest.json').read_text());assert manifest['repo_id']=='eagle705/nemochat-10BA1B-BF16-SFT-1200' and manifest['private'] and manifest['actual_source_step']==1248 and manifest['resume_step']==905 and manifest['segment_updates']==343
 assert manifest['source_payload_integrity']=='not_checked_explicit_user_waiver' and manifest['native_HF_logit_parity']=='not_run'
 export=json.loads((STAGE/'training/export_result.json').read_text());assert export['source_step']==1248 and export['source'].endswith('/iter_0001248') and export['all_source_model_keys_mapped']
 examples=json.loads((STAGE/'examples/examples.json').read_text());original=json.loads((OUT/'examples.json').read_text());assert examples['cases']==original['cases']
 assert examples['source_step']==1248 and examples['model']=='checkpoints/exports/nemochat_sft1248_hf_bf16_01'
 assert len(examples['cases'])==6 and all(c['new_tokens']==128 and len(c['generated_ids'])==128 and not c['eos_reached'] and c['generation_scores_finite'] and c['cache_class']=='DynamicCache' for c in examples['cases'])
 prompts=json.loads((STAGE/'examples/prompts-predeclared.json').read_text());assert [r['prompt'] for r in examples['cases']]==[r['prompt'] for r in prompts['cases']]
 assert examples['generation_config']['eos_token_id']==examples['generation_config']['pad_token_id']==11
 assert not examples['generation_config']['do_sample'] and examples['generation_config']['max_new_tokens']==128
 assert json.loads((STAGE/'examples/generation-engine.state.json').read_text())['state']['ExitCode']==0
 snippets=[]
 for name,section in [('README.md','## Runnable loading and generation (private repository)'),('README.ko.md','## 실행 예제와 검증 범위')]:
  text=(STAGE/name).read_text();assert '@@' not in text
  for link in re.findall(r'\]\(([^)]+)\)',text):
   if not link.startswith(('https:','http:','#')):assert (STAGE/link).is_file(),(name,link)
  code=text.split(section,1)[1].split('```python',1)[1].split('```',1)[0];ast.parse(code)
  for required in ['AutoModelForCausalLM.from_pretrained','enable_thinking=False','add_generation_prompt=True','trust_remote_code=False','eos_token_id=tok.eos_token_id']:
   assert required in code,(name,required)
  snippets.append(code)
 assert snippets[0]==snippets[1]
 t=json.loads((STAGE/'training/training-summary.json').read_text());assert t['updates']==1248 and t['ranks_verified']==8 and t['assistant_target_exposures']==117409886 and t['processed_positions_including_padding']==2617245696 and t['unique_conversations']==159744
 assert [r['update'] for r in t['validation']]==[250,500,750,1000] and all(r['fixed_heldout_window_reused'] for r in t['validation'])
 assert len((STAGE/'training/observed-metrics.csv').read_text().splitlines())==1249
 for name in ['sft1248-training-summary.png','benchmark-five-arms-SFT1248.png','heldout-five-arms-SFT1248.png']:
  assert (STAGE/'graphs'/name).read_bytes()[:8]==b'\x89PNG\r\n\x1a\n'
 # Validate text-like payloads have no absolute project path or token-shaped secret.
 for name in expected:
  if name.endswith(('.md','.json','.jsonl','.csv','.jinja')):
   text=(STAGE/name).read_text();assert '/raid/yoon/' not in text,name
   assert not re.search(r'\bhf_[A-Za-z0-9]{25,}\b',text),name
 receipt=dict(status='passed_local_model_only_publication_acceptance',files=len(expected),payload_bytes=sum(expected.values()),weight_shards=12,weight_file_checks='stat_and_readable_symlink_only_no_payload_audit',new_once_generations=6,raw_generation_cases_exact=True,bilingual_runtime_snippets_identical_and_syntax_valid=True,all_main_card_relative_links_exist=True,graphs_verified_png_headers=3,source_step=1248,secret_pattern_scan='no_token_shaped_strings',source_payload_SHA_CRC='not_checked_explicit_user_waiver')
 (OUT/'local-acceptance.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
