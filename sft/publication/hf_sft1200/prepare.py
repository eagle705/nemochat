"""Prepare new model-only publication with accepted export symlinks and small evidence."""
import json,os,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];CODE=ROOT/'sft/publication/hf_sft1200';OUT=ROOT/'results/midtraining/hf-sft1200-publication-01';BENCH=ROOT/'results/midtraining/sft1248-benchmark-gpu-01';MODEL=ROOT/'checkpoints/exports/nemochat_sft1248_hf_bf16_01';STAGE=ROOT/'checkpoints/exports/nemochat_sft1248_publication_01'
REPO='eagle705/nemochat-10BA1B-BF16-SFT-1200'
def clean(x):
 if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
 if isinstance(x,list):return [clean(v) for v in x]
 if isinstance(x,str):return x.replace(str(ROOT)+'/', '')
 return x
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(clean(x),indent=2,ensure_ascii=False)+'\n')
def copy(a,b):b.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(a,b)
def main():
 assert not STAGE.exists(),'Refuse stale staging; inspect explicitly'
 export=json.loads((BENCH/'export_result.json').read_text());accepted=json.loads((BENCH/'complete.json').read_text());examples=json.loads((OUT/'examples.json').read_text());state=json.loads((OUT/'generation-engine.state.json').read_text());training=json.loads((OUT/'training-summary.json').read_text());native=json.loads((ROOT/'results/midtraining/sft-continue1500-02/complete.json').read_text())
 assert export['source_step']==accepted['source_step']==examples['source_step']==native['final_step']==1248
 assert accepted['local_export']==str(MODEL) and examples['model']==str(MODEL)
 assert export['all_source_model_keys_mapped'] and accepted['all_loaded_tensors_finite']
 assert state['state']['ExitCode']==0 and not state['state']['OOMKilled']
 assert [c['id'] for c in examples['cases']]==[c['id'] for c in json.loads((OUT/'prompts-predeclared.json').read_text())['cases']]
 assert len(examples['cases'])==6 and all(c['generation_scores_finite'] and c['new_tokens']==128 and not c['eos_reached'] for c in examples['cases'])
 assert not any(examples['loading_info'].values()) and examples['tokenizer_assets_exact']
 assert len((OUT/'generations.jsonl').read_text().splitlines())==6
 assert training['updates']==1248 and training['ranks_verified']==8 and training['contiguous_updates']
 for name,size in export['files'].items():
  p=MODEL/name;assert p.stat().st_size==size and os.access(p,os.R_OK)
 assert len([n for n in export['files'] if n.endswith('.safetensors')])==12
 STAGE.mkdir()
 for name in export['files']:(STAGE/name).symlink_to(MODEL/name)
 # Do not ship local absolute host paths in export metadata.
 save(STAGE/'nemochat_export_source.json',json.loads((MODEL/'nemochat_export_source.json').read_text()))
 for name in ['prompts-predeclared.json','generation-plan-before-load.json','examples.json','load-receipt.json','generation-engine.state.json']:
  save(STAGE/'examples'/name,json.loads((OUT/name).read_text()))
 (STAGE/'examples/generations.jsonl').write_text(''.join(json.dumps(clean(json.loads(s)),ensure_ascii=False)+'\n' for s in (OUT/'generations.jsonl').read_text().splitlines()))
 # Four fixed-order prefixes; no best-case selection; all six full outputs retained.
 rows={r['id']:r for r in examples['cases']};blocks=[]
 for ident,limit in [('en_sky',320),('ko_sky',180),('en_python',280),('ko_json',180)]:
  c=rows[ident];blocks.append(f"### {ident} — {c['language']}\n\nPrompt: {c['prompt']}\n\n**Verbatim generated prefix excerpt; 128 tokens total, EOS not reached:**\n\n````text\n{c['generated_text'][:limit]}\n````\n\n[Display excerpt ends here; see the full raw continuation. / 화면 발췌 종료; 전체 원문은 raw artifact 참조.]\n")
 blocks.append('| Case | New tokens | EOS reached | Observed limitation |\n|---|---:|---|---|\n| en_sky |128|No|Ignored one-sentence request, repetition|\n| ko_sky |128|No|Incorrect Korean repetition|\n| en_python |128|No|Duplicated code and repetitive output, extra formatting|\n| ko_python |128|No|Wrong language/content; repetitive “100% correct”, no requested function|\n| en_json |128|No|Wrong key, repeated keys, unclosed JSON/code fence|\n| ko_json |128|No|Repetitive prose, not JSON|\n')
 en=(CODE/'README.template.md').read_text().replace('@@EXAMPLES@@','\n'.join(blocks))
 (STAGE/'README.md').write_text(en)
 snippet='```python'+en.split('## Runnable loading and generation (private repository)',1)[1].split('```python',1)[1].split('```',1)[0]+'```'
 ko=(CODE/'README.ko.template.md').read_text().replace('@@EXAMPLES@@','\n'.join(blocks)).replace('@@SNIPPET@@',snippet)
 (STAGE/'README.ko.md').write_text(ko)
 prior=ROOT/'checkpoints/exports/nemochat_mid_super35_distill_publication_01'
 constraints=(prior/'USE_CONSTRAINTS.md').read_text().replace('The tokenizer is included unchanged.','That BASE release included the original tokenizer unchanged; the current SFT release instead preserves its accepted migrated Super3.5 assets at revision fc067b34f6b514014d2f8bddba855d2e4bfdfba6, without embedding row changes.')
 constraints+='''\n\n## SFT1248 additions and exact repository naming\n\nThis private release eagle705/nemochat-10BA1B-BF16-SFT-1200 contains actual\nFINAL native step1248, model-only, after905+343 full-state-resumed SFT updates.\nIt adds no learned-weight license grant or commercial/redistribution authorization.\nSelected SFT sources, recorded as OpenMDW1.1 in the frozen preparation manifest:\n- NVIDIA Nemotron-SFT-Instruction-Following-Chat-v4, revision1cd362945d7a00b136b95b96892bbfea8798686e: chat, instruction_following_for_response_length_control, multi_turn_instruction_following.\n- NVIDIA Nemotron-SFT-Agentic-v4, revisiona4593a26ea83a700035bbe0ec1edc9ef3810669d: tool_calling and knowledge_agent.\n- NVIDIA Nemotron-SFT-BBQ-v1, revision4efdb045cda11918f29065a2e0fd5000b73b8c22: train.\n\nReview https://openmdw.ai/license/1-1/ and the exact pinned upstream cards.\nLong-Context-Transduction and reserved RL seed were not used in SFT.\nTokenizer attribution for the accepted migrated Super3.5 revision is retained in\nTOKENIZER_UPSTREAM_README.md; it is not the local trained-model card. The original\nBASE/tokenizer lineage and inherited ClimbMix noncommercial constraints remain.\nNo teacher weights, native optimizer/RNG, raw training data or credentials are included.\nNoncommercial educational/research scope and no-new-grant restrictions remain.\n'''
 (STAGE/'USE_CONSTRAINTS.md').write_text(constraints)
 copy(prior/'README.md',STAGE/'training/inherited_midtraining_modelcard.md')
 for name in ['sft1248-training-summary.png']:
  copy(OUT/name,STAGE/'graphs'/name)
 for name in ['benchmark-five-arms-SFT1248.png','heldout-five-arms-SFT1248.png']:copy(BENCH/name,STAGE/'graphs'/name)
 for name in ['observed-metrics.csv','validation-observations.json','training-summary.json']:copy(OUT/name,STAGE/'training'/name)
 for name in ['scores.csv','paired-question-deltas.csv','paired-question-deltas.json']:copy(BENCH/name,STAGE/'evaluation'/name)
 for name in ['summary.json','load-receipt.json','tokenizer-compatibility03.json','recovery-disclosure.json','recovery03-disclosure.json']:
  save(STAGE/'evaluation'/name,json.loads((BENCH/name).read_text()))
 save(STAGE/'evaluation/E-report.json',json.loads((BENCH/'E-full01/report.json').read_text()))
 for name in ['FINAL_REPORT.md','FINAL_REPORT.ko.md']:
  text=(BENCH/name).read_text()
  for image in ['benchmark-five-arms-SFT1248.png','heldout-five-arms-SFT1248.png']:text=text.replace(']('+image+')','](../graphs/'+image+')')
  (STAGE/'evaluation'/name).write_text(text.replace(str(ROOT)+'/', ''))
 m=json.loads((ROOT/'datasets/nemotron_sft/nemochat_baby/prepared-full-03/data/manifest.json').read_text())
 keys=['schema','format','status','sequence_length','train_dialogues','heldout_dialogues','train_loss_tokens','heldout_loss_tokens','nonpadding_tokens','per_source','reasoning_policy','mask_policy','long_context_policy','split_policy','training_repetition','excluded_datasets','tokenizer_revision','decontamination','source_schema_policy']
 data={k:m[k] for k in keys};data['selected_sources']=[{k:s[k] for k in ['dataset','revision','path','license','key']} for s in m['sources'] if s['purpose']=='sft' and not m['per_source'].get(s['key'],{}).get('excluded')];assert len(data['selected_sources'])==6
 data['pool_counts_not_consumed_proportions']=True;save(STAGE/'training/data_provenance.json',data)
 config=json.loads((ROOT/'results/midtraining/sft-continue1500-02/main-config.json').read_text())
 config={k:config[k] for k in ['sequence_length','context_parallel_size','expert_model_parallel_size','global_batch_size','micro_batch_size','epochs','allow_repetition','max_updates','learning_rate','min_learning_rate','weight_decay','seed','validation_interval','validation_max_dialogues','reasoning_policy','format']}
 config.update(actual_cumulative_updates=1248,segment_updates=343,resume_step=905,restore_optimizer_scheduler_rng_cursor=True,resume_cursor=115840,scheduler_horizon_not_rebased=True,optimizer='native_distributed_AdamW',tensor_parallel_size=1,pipeline_parallel_size=1,bridge_version='0.6.1',objective='selected_assistant_LM_plus_native_router_aux_no_KD_in_SFT',lr_measurement_note='1..905 callback and SDK printed post-step distinct;906..1248 optimizer boundary and post-scheduler explicitly observed',processed_positions_including_padding=2617245696,assistant_target_exposures=117409886,unique_consumed_conversations=159744)
 save(STAGE/'training/hyperparameters.json',config)
 save(STAGE/'training/native_final_save_receipt.json',native)
 save(STAGE/'training/export_result.json',export)
 save(STAGE/'training/benchmark_acceptance.json',accepted)
 manifest=dict(repo_id=REPO,private=True,repository_name_milestone=1200,actual_source_step=1248,actual_source_checkpoint='results/midtraining/sft-continue1500-02/main/checkpoint/iter_0001248',native_final_save_status=native['status'],native_final_engine_exit=0,native_final_engine_finished_at='2026-10-19T05:39:16.910261485Z',resume_step=905,segment_updates=343,source_weights='actual_FINAL_SFT1248_not_SFT905_or_C',model_only=True,native_optimizer_rng_distributed=False,format='HF safetensors',parameter_numel=9790967416,shards=12,source_payload_integrity='not_checked_explicit_user_waiver',native_HF_logit_parity='not_run',weight_redownload_for_verification=False,new_generation_cases=6,new_generation_once_each=True,generation_engine_id=state['id'],generation_engine_exit=0,generation_eos_reached=0,graph_visual_review='three figures readable and unclipped',source_paths='project_relative_sanitized_no_host_prefix')
 save(STAGE/'publication_manifest.json',manifest)
 expected={str(p.relative_to(STAGE)):p.stat().st_size for p in STAGE.rglob('*') if p.is_file()}
 save(OUT/'expected-inventory.json',expected)
 assert not any('optimizer' in n.lower() or n.endswith('.distcp') for n in expected)
 from huggingface_hub import ModelCard
 card=ModelCard.load(str(STAGE/'README.md'));card.validate()
 save(OUT/'preparation-receipt.json',dict(status='prepared_modelcard_metadata_validated',repo_id=REPO,private=True,actual_source_step=1248,files=len(expected),bytes=sum(expected.values()),safetensors_shards=12,card_metadata_validation=True,generation_once_six=True))
 print(json.dumps(dict(stage=str(STAGE),files=len(expected),bytes=sum(expected.values()),source_step=1248),indent=2))
if __name__=='__main__':main()
