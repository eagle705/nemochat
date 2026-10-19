"""Authenticate retained CPU producer, tokenizer and selected-mask lineage."""
import hashlib
import json
from pathlib import Path
import sys
import verify
sys.path.insert(0,str(verify.ROOT/'src'))
from nemochat.tokenization.posttraining import compare_tokenizers

root=verify.ROOT;out=verify.OUT

def receipt(path):
 p=root/path;b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}

base=Path('results/pretraining/nemochat_baby/nemotron_sft/cpu-full-03')
hand=json.loads((root/base/'FULL_HANDOFF.json').read_text());m=json.loads(verify.MANIFEST.read_text());c=json.loads(verify.CORPUS.read_text())
audit_path=verify.MANIFEST.parent/'counted-source-mask-audit.json';audit=json.loads(audit_path.read_text())
records={}
for name,expected in hand['source_code_sha256'].items():
 r=receipt(name);r['historical_expected_sha256']=expected;r['matches_historical']=r['sha256']==expected;records[name]=r
for name in ['full-preparation.final-state.json','full-preparation.started-state.json','full-preparation.wait.json','full-preparation.command.json','preparation-controller-state.json','preparation-controller.log','FULL_HANDOFF.json','publish_ready.py','full_workload.py','prepare_full_parallel.py','run_full_preparation.py']:
 records[str(base/name)]=dict(receipt(base/name),**({'historical_expected_sha256':records[str(base/name)]['historical_expected_sha256'],'matches_historical':records[str(base/name)]['matches_historical']} if str(base/name) in records else {}))
for p in [audit_path,verify.MANIFEST,verify.CORPUS,root/m['source_schema_policy']['full_source_audit'],root/'results/pretraining/nemochat_baby/nemotron_sft/cpu-full-01/acquisition-completed.json',root/'datasets/nemotron_sft/nemochat_baby/prepared-full-03/verification/result.json',root/m['tokenizer_manifest'],root/'configs/tokenizers/nemotron_super35.yaml',root/'src/nemochat/tokenization/posttraining.py']:
 records[str(p.relative_to(root))]=receipt(p)
engine=json.loads((root/base/'full-preparation.final-state.json').read_text());wait=json.loads((root/base/'full-preparation.wait.json').read_text());start=json.loads((root/base/'full-preparation.started-state.json').read_text());command=json.loads((root/base/'full-preparation.command.json').read_text());create=json.loads((root/base/'full-preparation.create.json').read_text())
producer_ok=(engine['Id']==start['Id']==create['Id'] and engine['State']['ExitCode']==wait['StatusCode']==0 and engine['State']['Running'] is False and engine['State']['OOMKilled'] is False and not command['HostConfig']['DeviceRequests'] and engine['Config']['Cmd']==command['Cmd'])
mig=json.loads((root/m['tokenizer_manifest']).read_text());assets={}
for name,expected in mig['backend_comparison']['new_assets'].items():
 r=receipt(Path(mig['tokenizer_path'])/name);r['expected']=expected;r['accepted']=r['bytes']==expected['bytes'] and r['sha256']==expected['sha256'];assets['SFT/'+name]=r
oldroot=Path('results/pretraining/nemochat_baby/tokenizer_migration/old-official')
for name,expected in mig['backend_comparison']['old_official_provenance']['official_assets'].items():
 r=receipt(oldroot/name);r['expected']=expected;r['accepted']=r['bytes']==expected['bytes'] and r['sha256']==expected['sha256'];assets['old/'+name]=r
for name,sha in c['tokenizer']['asset_sha256'].items():
 r=receipt(Path(c['tokenizer']['path'])/name);r['expected_sha256']=sha;r['accepted']=r['sha256']==sha;assets['C-corpus/'+name]=r
# Exported C's exact original tokenizer assets only; never enumerate weights.
cmodel=Path('checkpoints/exports/nemochat_mid_super35_distill_hf_bf16_01')
for name,expected in mig['backend_comparison']['old_assets'].items():
 r=receipt(cmodel/name);r['expected']=expected;r['accepted']=r['bytes']==expected['bytes'] and r['sha256']==expected['sha256'];assets['C-export-original/'+name]=r
old=json.loads((root/oldroot/'tokenizer.json').read_text());new=json.loads((root/mig['tokenizer_path']/'tokenizer.json').read_text());export=json.loads((root/cmodel/'tokenizer.json').read_text());corpus=json.loads((root/c['tokenizer']['path']/'tokenizer.json').read_text())
comparison=compare_tokenizers(old,new)
comparison.update(C_export_backend_exact_old=old==export,C_corpus_backend_exact_SFT=corpus==new,weight_embedding_permutation_resize=False)
source_audit=json.loads((root/m['source_schema_policy']['full_source_audit']).read_text());acq=json.loads((root/'results/pretraining/nemochat_baby/nemotron_sft/cpu-full-01/acquisition-completed.json').read_text())
result=dict(status='provenance_authenticated' if producer_ok and all(r.get('matches_historical',True) for r in records.values()) and all(r['accepted'] for r in assets.values()) else 'provenance_blocked',producer_engine_clean_exit=producer_ok,producer_id=engine['Id'],producer_state=engine['State'],producer_source_current_matches_historical_handoff=all(r.get('matches_historical',True) for r in records.values()),records=records,tokenizer_assets=assets,tokenizer_comparison=comparison,original_counted_audit=audit,source_audit_digest_matches_manifest=receipt(m['source_schema_policy']['full_source_audit'])['sha256']==m['source_schema_policy']['full_source_audit_sha256'],original_counts_match_manifest=audit['all_conversations_counted']==m['train_dialogues']+m['heldout_dialogues'] and audit['all_loss_tokens_counted']==m['train_loss_tokens']+m['heldout_loss_tokens'],source_schema_counts_match=source_audit['rejected_records']==m['source_schema_policy']['rejected_records']==audit['source_schema_rejections_verified'],acquisition_complete=acq['status']=='passed' and acq['completed_files']==7,checkpoint_payloads_read=False,old_publisher_failure={'log':str(base/'preparation-controller.log'),'reason':'stable prepared file byte/hash gate failed','exact_historical_failing_file':'not recoverable from retained exception: publisher logs neither filename nor digest, and stops at first mismatch','file_iteration_order':list(m['files']),'old_completion_marker_exists':(root/base/'completion-ready.json').exists()},mask_lineage={'every_retained_producer_row':'official vs instrumented full rendered bytes and token IDs equality, selected message-aligned assistant spans','independent_re_render_rows':216,'exhaustive_second_render':False,'historical_source_authenticated':all(r.get('matches_historical',True) for r in records.values())},transition_policy={'prior_C_corpus':'fresh zero occurrences required for alias meaning transition; original model-export aliases SPECIAL18..29, corpus was encoded using new official backend','new_SFT':'measure and classify 18..29 under new official aliases; NOT blanket zero; no resizing/permutation','prior_ClimbMix_proof':'retained migration receipt only, not newly rescanned in this task','fresh_bulk_acceptance':'pending separate full-file scans, provenance is not a data release'})
verify.save(out/'provenance.json',result)
print(json.dumps({k:result[k] for k in ['status','producer_engine_clean_exit','producer_source_current_matches_historical_handoff','source_audit_digest_matches_manifest','original_counts_match_manifest','source_schema_counts_match']},indent=2));print('bad assets',[(k,r) for k,r in assets.items() if not r['accepted']])
