"""Guarded private model-only upload; recover committed SHA read-only, never overwrite."""
import importlib.util,json,os,sys,time,traceback
from pathlib import Path
ROOT=Path('/raid/yoon/nemochat');OUT=ROOT/'results/midtraining/hf-sft1200-publication-01';STAGE=ROOT/'checkpoints/exports/nemochat_sft1248_publication_01';REPO='eagle705/nemochat-10BA1B-BF16-SFT-1200'
def save(name,x):(OUT/name).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def credential():
 spec=importlib.util.spec_from_file_location('existing_publisher',ROOT/'midtraining/publication/hf_kd1000/publish.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module.credential()
def inventory():return {str(p.relative_to(STAGE)):p.stat().st_size for p in STAGE.rglob('*') if p.is_file()}
def verify(api,token,revision,expected):
 from huggingface_hub import hf_hub_download
 info=api.repo_info(REPO,repo_type='model',revision=revision,files_metadata=True)
 assert info.private is True and info.sha==revision
 actual={p.rfilename:p.size for p in info.siblings if p.rfilename!='.gitattributes'}
 assert actual==expected,'Remote inventory mismatch'
 checked=[]
 for name in sorted(expected):
  if name.endswith('.safetensors'):continue
  assert expected[name]<50000000,'Refuse unbounded nonweight readback'
  path=Path(hf_hub_download(REPO,filename=name,revision=revision,token=token,local_dir=OUT/'small-readback'))
  assert path.read_bytes()==(STAGE/name).read_bytes(),'Small asset byte mismatch: '+name
  checked.append(name)
  save('verification-progress.json',dict(revision=revision,checked=checked,expected_small_assets=len(expected)-12))
 assert api.repo_info(REPO,repo_type='model').sha==revision,'Head changed after commit'
 receipt=dict(status='published_and_verified',repo_id=REPO,url='https://huggingface.co/'+REPO,revision=revision,private=True,actual_source_step=1248,requested_repo_name_step=1200,model_only=True,format='HF BF16 safetensors',payload_files=len(expected),payload_bytes=sum(expected.values()),safetensors_shards=12,remote_inventory_exact=True,small_assets_byte_equal=checked,small_assets_verified=len(checked),hub_generated_files={p.rfilename:p.size for p in info.siblings if p.rfilename=='.gitattributes'},source_checkpoint_payload_SHA_CRC='not_checked_explicit_user_waiver',native_HF_logit_parity='not_run',remote_weights_redownloaded=False,files=expected,verified_at_unix=time.time())
 save('publication_receipt.json',receipt);print(json.dumps(receipt,indent=2),flush=True)
def main():
 assert os.environ.get('HF_HUB_DISABLE_XET')=='1' and os.environ.get('HF_HUB_ENABLE_HF_TRANSFER')=='0'
 from huggingface_hub import HfApi,CommitOperationAdd
 from huggingface_hub.errors import RepositoryNotFoundError
 expected=json.loads((OUT/'expected-inventory.json').read_text());assert inventory()==expected
 assert json.loads((OUT/'preparation-receipt.json').read_text())['card_metadata_validation']
 manifest=json.loads((STAGE/'publication_manifest.json').read_text());assert manifest['actual_source_step']==1248 and manifest['private']
 token=credential();api=HfApi(token=token);assert api.whoami()['name']=='eagle705'
 returned=OUT/'commit_returned.json'
 if returned.exists():verify(api,token,json.loads(returned.read_text())['revision'],expected);return
 owned=OUT/'repo-created.json'
 try:
  current=api.repo_info(REPO,repo_type='model',files_metadata=True)
  assert owned.exists(),'Occupied target not created by this run; refusing overwrite'
 except RepositoryNotFoundError:
  api.create_repo(REPO,repo_type='model',private=True,exist_ok=False)
  current=api.repo_info(REPO,repo_type='model',files_metadata=True);assert current.private is True
  save('repo-created.json',dict(repo_id=REPO,private=True,initial_revision=current.sha,created_at_unix=time.time()))
 assert current.private is True
 assert all(p.rfilename=='.gitattributes' for p in current.siblings),'Target no longer empty; inspect, do not overwrite'
 save('upload-preflight.json',dict(repo_id=REPO,private=True,parent_commit=current.sha,files=expected,pid=os.getpid(),transport='HTTP_HF_HUB_DISABLE_XET1_HF_TRANSFER0',started_at_unix=time.time()))
 print('UPLOADING_ACTUAL_FINAL_SFT1248_TO_EXACT_PRIVATE_SFT1200_TARGET',json.dumps(dict(repo_id=REPO,files=len(expected),bytes=sum(expected.values()))),flush=True)
 operations=[CommitOperationAdd(path_in_repo=name,path_or_fileobj=str(STAGE/name)) for name in sorted(expected)]
 commit=api.create_commit(repo_id=REPO,repo_type='model',revision='main',operations=operations,commit_message='Publish actual FINAL SFT1248 BF16 model, bilingual cards, observed graphs and six once-only generations',parent_commit=current.sha,create_pr=False,num_threads=2)
 save('commit_returned.json',dict(repo_id=REPO,revision=commit.oid,returned_at_unix=time.time()))
 verify(api,token,commit.oid,expected)
if __name__=='__main__':
 try:main()
 except Exception as error:
  save('publication_failure.json',dict(type=type(error).__name__,http_status=getattr(getattr(error,'response',None),'status_code',None),frames=[dict(file=Path(f.filename).name,line=f.lineno,function=f.name) for f in traceback.extract_tb(error.__traceback__)]))
  print('PUBLICATION_FAILED',type(error).__name__,'see sanitized publication_failure.json',flush=True);sys.exit(1)
