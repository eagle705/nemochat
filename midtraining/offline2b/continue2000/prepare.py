"""One-shot CPU preparation; --pin is separate, explicit, never called at runtime."""
import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
import sys
REPO=Path('/raid/yoon/nemochat');sys.path.insert(0,str(REPO))
from midtraining.offline2b.continue2000.contract import OLD,SOURCE,OUT,plan,sha,extend_state
HERE=Path(__file__).resolve().parent
OWN_FILES=('contract.py','broker.py','binding.py','input_gate.py','native.py','teacher.py','new_cache.py',
    'merge.py','supervisor.py','start.py','prepare.py','cpu_probe.py','README.md','README.ko.md','REQUEST_PLAN.json')
DEPENDENCIES=('midtraining/offline2b/launch.py','midtraining/offline2b/native.py','midtraining/native_kd.py',
 'midtraining/longrun_native.py','midtraining/offline2b/prefix/contract.py',
 'midtraining/offline2b/prefix/dynamic/restart01/broker.py','midtraining/offline2b/prefix/dynamic/restart01/reader.py',
 'midtraining/offline2b/prefix/directio/binding.py','midtraining/offline2b/prefix/filtered/transport.py',
 'midtraining/patches/student_embedding_import.py','midtraining/patches/offline_kd_pr/gpu_smoke/optimizer_compat.py',
 'results/pretrain_followup/engine_helper.py','tests/test_offline2000_continuation.py')
SNAPSHOTS=('sdk/megatron/core/datasets/gpt_dataset.py','sdk/megatron/core/datasets/indexed_dataset.py',
 'sdk/megatron/core/datasets/blended_dataset.py','sdk/megatron/core/datasets/utils.py',
 'staged/src/megatron/bridge/data/samplers.py','staged/src/megatron/bridge/data/utils.py',
 'staged/src/megatron/bridge/training/config.py','staged/src/megatron/bridge/training/pretrain.py',
 'staged/src/megatron/bridge/training/training.py','staged/src/megatron/bridge/training/checkpointing.py',
 'staged/src/megatron/bridge/training/cached_kd/integration.py','staged/src/megatron/bridge/training/cached_kd/runtime.py',
 'staged/src/megatron/bridge/training/cached_kd/logits_saver.py','staged/src/megatron/bridge/training/cached_kd/cached_logits_loss.py',
 'staged/src/megatron/bridge/training/cached_kd/utils_logits.py')

def put(path,value):
    with path.open('x') as f:json.dump(value,f,sort_keys=True,indent=2);f.write('\n')
def prepare():
    prep=OUT/'preparation';prep.mkdir(parents=True,exist_ok=False)
    m=json.loads((OLD/'prefix-manifest.json').read_text());full=Path(m['full_expectations_path'])
    if sha(full)!=m['full_expectations_sha256']:raise ValueError('old expectations identity changed')
    original=json.loads(full.read_text());p=plan()
    if len(original['shards'])!=9144 or original['budget']!=json.loads((SOURCE/'run-config.json').read_text()):raise ValueError('original full C changed')
    tx=json.loads((OLD/'checkpoint-step16-transaction.json').read_text())
    extend_state(tx['binding'],m,sha(OLD/'prefix-manifest.json'),2064)
    if not tx['save_optim'] or not tx['save_rng'] or not tx['native_synchronous_save_completed']:raise ValueError('FULL saved transaction required')
    if (OLD/'checkpoint/latest_checkpointed_iteration.txt').read_text().strip()!='16':raise ValueError('checkpoint root no longer at16')
    scratch=Path(tempfile.mkdtemp(prefix='offline2000-snapshot-',dir='/home/yoon/.hermes/cache/scratch'))
    approved=Path(m['approved_scratch_source'])
    shutil.copytree(approved/'sdk',scratch/'sdk');shutil.copytree(approved/'staged',scratch/'staged')
    shutil.copyfile(full,prep/'old-expectations.json');shutil.copyfile(SOURCE/'run-config.json',OUT/'run-config.json')
    shutil.copyfile(OLD/'checkpoint-step16-transaction.json',OUT/'resume-ancestry.json')
    shutil.copytree(SOURCE/'dataset-cache',OUT/'dataset-cache')
    put(prep/'prepared.json',dict(plan=p,snapshot=str(scratch),approved_snapshot_source=str(approved),
        old_manifest_sha256=sha(OLD/'prefix-manifest.json'),old_expectations_sha256=sha(full),
        checkpoint_transaction_sha256=sha(OLD/'checkpoint-step16-transaction.json'),
        checkpoint_payload_read=False,gpu_launched=False,teacher_data_ready=False))
    print(json.dumps(dict(status='CPU_PREPARED_ONLY',output=str(OUT),snapshot=str(scratch))))
def pin():
    prep=OUT/'preparation';s=Path(json.loads((prep/'prepared.json').read_text())['snapshot'])
    paths=[HERE/n for n in OWN_FILES]+[REPO/n for n in DEPENDENCIES]
    originals=[OLD/'prefix-manifest.json',OLD/'checkpoint-step16-transaction.json',
        Path(json.loads((OLD/'prefix-manifest.json').read_text())['full_expectations_path']),
        SOURCE/'run-config.json',prep/'prepared.json',prep/'old-expectations.json',OUT/'run-config.json',OUT/'resume-ancestry.json']
    originals += [SOURCE/f'teacher-rank{r}-samples.jsonl' for r in range(8)]
    missing=[str(p) for p in paths+[s/n for n in SNAPSHOTS]+originals if not p.is_file()]
    if missing:raise ValueError('source allowlist incomplete: '+str(missing))
    put(prep/'source-pins.json',dict(source_sha256={str(p.relative_to(REPO)):sha(p) for p in paths},
        snapshot_sha256={str(s/n):sha(s/n) for n in SNAPSHOTS},ancestry_sha256={str(p):sha(p) for p in originals},
        checkpoint_payloads_hashed=False,gpu_launched=False))
    print(json.dumps(dict(source_pins_sha256=sha(prep/'source-pins.json'),source_count=len(paths),snapshot_count=len(SNAPSHOTS))))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pin',action='store_true');a=p.parse_args()
    pin() if a.pin else prepare()
