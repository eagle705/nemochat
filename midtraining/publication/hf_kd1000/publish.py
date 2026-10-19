"""Publish the accepted KD1000 HF export to one exact, new private Hub target."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil

ROOT = Path('/raid/yoon/nemochat')
REPO = 'eagle705/nemochat-10BA1B-BF16-mid-super3.5-distill'
RESULT = ROOT/'results/midtraining/hf-kd1000-publication-01'
CONVERSION = ROOT/'results/midtraining/hf-kd1000-conversion-01'
MODEL = ROOT/'checkpoints/exports/nemochat_mid_super35_distill_hf_bf16_01'
STAGING = ROOT/'checkpoints/exports/nemochat_mid_super35_distill_publication_01'
SOURCE = ROOT/'results/midtraining/offline2b-prefixtrain-1000-01/checkpoint/iter_0001000'


def validate(v):
    assert v['status']=='passed' and v['source_step']==1000
    assert Path(v['source'])==SOURCE and Path(v['model'])==MODEL
    assert v['full_model_reload'] and v['official_transformers'] and v['trust_remote_code'] is False
    assert v['parameter_numel']==9790967416 and v['all_reloaded_tensors_finite']
    assert not any(v['loading_info'].values())
    assert v['tokenizer_bytes_exact_match'] and v['tokenizer_vocab_exact_match']
    assert len(v['cases'])==2 and all(c['finite'] and c['generation_scores_finite'] and c['new_tokens']==4 and c['generation_cache'] for c in v['cases'])


def card(root, validation):
    validate(validation)
    return '''---
library_name: transformers
pipeline_tag: text-generation
base_model: eagle705/nemochat-10BA1B-BF16-base
base_model_relation: finetune
language:
- en
- ko
datasets:
- nvidia/Nemotron-Pretraining-Specialized-v1.1
- karpathy/climbmix-400b-shuffle
- nvidia/Nemotron-ClimbMix
tags:
- nemochat
- continued-pretraining
- knowledge-distillation
- base-model
- bf16
- mamba
- mixture-of-experts
- megatron-bridge
license: other
license_name: upstream-constraints-no-additional-grant
license_link: https://huggingface.co/eagle705/nemochat-10BA1B-BF16-mid-super3.5-distill/blob/main/USE_CONSTRAINTS.md
---

# NemoChat 10BA1B BF16 — mid-training with Nemotron 3.5 Super Base distillation

**Text-only continued-pretraining research checkpoint, not an instruction/chat model.**
This release contains the actual student weights after **1,000 native optimizer calls**
from the completed BASE19074 checkpoint. It is neither the unchanged initialization
nor the separate LM-only control. There were **999 positive-LR updates and one initial
zero-LR optimizer call**. No SFT, RLVR, WSM merge, vision encoder or teacher weights
are included. This is a model-only **HF safetensors** release, not native optimizer state.

The rounded name means **9,790,967,416 learned parameters**, approximately
964,054,648 active parameters under the inherited dense-plus-top-k weight-count
convention (not measured FLOPs). The architecture is unchanged from BASE: 52 serialized
Mamba/attention/latent-MoE blocks; hidden1024, latent512, 128 routed experts/top6,
shared expert width1024, routed width3072, vocabulary131072, untied embeddings,
16 attention heads/2 KV groups and head_dim128. Training context is **4096**;
no long-context or broad language-quality claim is made. BF16 weights retain the
23 native FP32 router correction buffers.

## Teacher and training lineage

- Initialization: `eagle705/nemochat-10BA1B-BF16-base`, completed BASE update19074,
  model-only warmstart with fresh optimizer, RNG, scheduler and input cursor.
- Exact teacher: `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-Base-BF16`, revision
  `4ba52e8ffa79e9693936c1a819ac89f551ceeef6`. **Base**, not Instruct/post-trained.
- A separately reviewed immutable text-backbone view excluded vision/projector/MTP
  records; the student consumed offline teacher scores, not visual inputs.
- Run: October18,2026, **18:48:02–21:41:52 KST**, elapsed **2h53m50s**, eight B300 GPUs.
- Final raw native checkpoint: **step1000**, not an averaged or intermediate checkpoint.

## Token accounting and data

**This mid-training stage processed 524,288,000 next-token positions:**
`1000 calls × global batch128 × sequence4096`. This includes the initial zero-LR
call; positions are not a claim of distinct source tokens or positive-LR-only exposure.
Global sequence samples: **128,000**. The preceding BASE processed10,000,269,312
positions; the sum across the two stages is10,524,557,312 exposures, **not unique tokens**,
because the current stage deliberately includes ClimbMix replay.

The prepared native training pool contained **1,929,598,341 tokens / 3,882,611 documents**.
The larger frozen sampler/scheduler horizon was3662 calls /1,919,942,656 positions;
**neither that full horizon nor the entire prepared pool was consumed by this release**.
The finite teacher prefix had1143 global source batches;1000 accepted unique physical
batches were learned without cycling, through physical source1090.91 rejected global
batches were skipped synchronously before any forward/optimizer/LR advance: seven
historical exclusions and84 newly observed runtime input/cache failures. Accepted
teacher/student input digests were matched for128,000 rank/microbatch rows.
This operational mitigation does **not** establish that the host read fault was repaired.

Nominal native sampling recipe (not measured post-quarantine proportions):

| Source | Weight | Prepared train tokens | Provenance |
|---|---:|---:|---|
| Code Concepts |67%|1,292,808,470|Specialized-v1.1, Code-Concepts selected11 parquet files|
| Unconditional Algorithmic |10%|192,967,490|Specialized-v1.1, selected first parquet|
| Formal Logic |3%|57,901,312|Specialized-v1.1, selected first parquet|
| ClimbMix replay |20%|385,921,069|Selected previously prepared BASE corpus|

Specialized dataset: `nvidia/Nemotron-Pretraining-Specialized-v1.1`, pinned revision
`13fa979be2e7f7e62913eee0ec5e97c8fd6e24af`.[1]
Replay repack: `karpathy/climbmix-400b-shuffle` revision
`915333b4f8b8684f39aeaafea600fea6f43fb703`; upstream `nvidia/Nemotron-ClimbMix`
revision`5eaa64b9c0c85b7f56af01d7dffdb0795816b12b`.
Economics and Multiple-Choice subsets were **not** used. Preparation preserved heldout
reservations, exact-token-document deduplication, seven authorized whole-document
Algorithmic reserved-ID exclusions and31 Replay roundtrip exclusions; no ID remapping.
Reserved validation pool:406,814 tokens/1077 documents, separate from training.
See [machine-readable provenance](training/data_provenance.json).

The corpus used the pinned3.5 teacher tokenizer with reserved IDs18..29 audited absent.
The **released tokenizer remains the unchanged original BASE assets**, from
`nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16` revision
`2dc98e2afe4face0e4ce40972a915c45368bd34a`. Ordinary-text alignment was verified;
reserved alias spellings were not migrated, and embeddings were not resized/permuted.
Native training used NullTokenizer vocab131072/EOD11 over prepared token IDs.

## Actual KD objective and essential Megatron Bridge configuration

**Primary objective: `0.5 × LM Loss + 0.5 × Ghost-tail KL`.**
Teacher probabilities are globally normalized at **T=1** before storing **Top32**
FP16 log-probabilities. The remaining vocabulary mass forms the native ghost-tail
bin; this is not conditional Top32 renormalization, full-vocabulary KL or a free
text SFT dataset. Router auxiliary loss/expert-bias handling remains native and separate.
There is **no online teacher** in the student read stage and no ModelOpt detached
LM/KD magnitude balancer in this cached objective.

The executed runtime was **Megatron Bridge0.6.1 / MCore0.19.1 / Transformers5.12.1**
with a scoped Core-first cached-KD patch. `CachedKDConfig` below is **not** stock
API in unmodified Bridge0.6.1. The [versioned patch](training/megatron-bridge-0.6.1-cached-kd.patch)
and [selected training settings](training/core_config.json) are included; this is not
an assertion that stock Bridge CLI accepts cache flags or that this excerpt alone
implements the exact whole-batch digest/quarantine/cursor protocol.

```python
# After applying the bundled reviewed patch to the pinned SDK copy:
from megatron.bridge.training.cached_kd.integration import (
    CachedKDConfig, CachedKDForwardStep,
)
from megatron.bridge.training.pretrain import pretrain

kd = CachedKDConfig(
    mode="read", cache_dir=cache_dir, cache_id=immutable_cache_id,
    alpha=0.5, top_k=32, top_p=None, min_k=1, save_dtype="fp16",
    decode_threads=1, prefetch_factor=2, msc_prefetch_depth=0,
    objective="megatron-lm-ghost-tail-fixed-alpha",
)
# runtime_factory binds the frozen full sampler, topology and live step.
# common_forward binds the actual GPT forward and accepted immutable inputs.
with CachedKDForwardStep(kd, runtime_factory, common_forward=common_forward) as step:
    pretrain(cfg, step, callbacks=callbacks)
```

T=1 is the cache/objective contract, **not** an invented temperature dataclass field.
The production adapter additionally commits physical-to-logical source mappings,
checks every rank before forward, and checkpoints optimizer/RNG/cache cursor together.

| Setting | Executed value |
|---|---|
| Sequence/global/micro batch |4096 /128 /1;16 microbatches per rank/call|
| Parallelism |TP1 /PP1 /CP1 /EP8 /DP8; no sequence parallelism|
| Optimizer |AdamW, beta1=.9, beta2=.999, eps1e-8, weight_decay=.01|
| Distributed optimizer |False; FP32 gradient reduction|
| Gradient clipping |1.0 native global norm|
| LR |peak1e-5, min1e-6;37 warmup calls, WSD full horizon3662/decay366|
| Actual LR at call1000 |1e-5; segment ends before decay, no rescaled1000-step schedule|
| RNG seed |43|
| Precision/dropout |BF16; hidden/attention dropout0|
| Router |native auxiliary coefficient1e-4 and expert-bias updates|
| Native save |synchronous model+optimizer+RNG; final1000, retention3; model-only HF here|

## Observed curves and results (limited scope)

![Actual LR and raw/smoothed training losses](graphs/lr_loss.png)

Faint lines show raw native observations; solid Loss lines show trailing16-call means.
Last16 means: **LM1.132520 / KD0.547897 / Primary Total0.840209**.
The final call's raw values were LM1.100863 / KD0.536768 / Total0.818816.
The original [1000-row observations](training/observed.csv) are included.
These are recorded native scalar diagnostics, not a derived benchmark score.

![Native gradient and callback timing](graphs/gradient_timing.png)

Grad Norm is pre-clip. Callback timings exclude save/evaluation/post-callback data
preflight; they are **not end-to-end throughput measurements**.

![Separate native validation observations](graphs/validation_snapshots.png)

At step1000 the periodic native Validation LM Loss was1.204911 and the separate
end-of-training call was0.9206597. These are two calls of a changing iterator,
**not fixed-input before/after evidence**; neither is silently discarded or used to
claim generalization gain. Existing BASE validation and this stage's values have
different data/scope and must not be presented as a matched quality comparison.

**KD benefit is not established yet.** An authorized ordinary **1.0 LM** control
is training from the same original BASE using the exact accepted input order and1000-call
budget. A frozen BASE/LM+KD/LM-only comparison is pending: MMLU200 total across10
subjects, ARC-Challenge100, HellaSwag100, plus65,536 fixed heldout target tokens.
No scores, claimed accuracy gains or significance are fabricated. The subset is
not full MMLU/leaderboard coverage; benchmark contamination is not exhaustively ruled out.

## HF loading and actual conversion validation

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
repo = "eagle705/nemochat-10BA1B-BF16-mid-super3.5-distill"
# Authenticate first: this repository is private.
tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=False)
model = AutoModelForCausalLM.from_pretrained(
    repo, dtype=torch.bfloat16, trust_remote_code=False,
    attn_implementation="eager",
).eval()  # Tested on CPU; approximately20GB of weights plus runtime overhead.
x = tokenizer("The capital of France is", return_tensors="pt")
with torch.inference_mode():
    y = model.generate(**x, max_new_tokens=4, do_sample=False,
                       use_cache=True, pad_token_id=tokenizer.eos_token_id)
print(tokenizer.decode(y[0], skip_special_tokens=True))
```

Official `NemotronHForCausalLM` fully reloaded in Transformers5.12.1, Python3.12.3,
PyTorch2.13.0a0+8145d630e8.nv26.06. Missing/unexpected/mismatched/error lists were empty;
all reloaded tensors and bilingual forward/generation scores were finite. Four-token
cached generations returned actual DynamicCache. Original tokenizer bytes matched.
The Korean diagnostic completion was incorrect: finite execution is **not** a quality test.

Conversion used the Bridge-registry-derived global-DCP adapter, not Bridge's ordinary
export CLI:619 native global model tensors →6289 serialized HF records →447 official
runtime records, including grouped experts. Mamba/QKV/expert transforms and FP32 router
buffers were preserved; no randomized missing-key fallback was used.
No remote model code is required. Optional Mamba fast kernels are disabled in this
portable configuration. GPU/fused-kernel execution, native/HF numerical logit parity,
vLLM, TensorRT-LLM, long context, safety and production suitability are **not verified**.
Source checkpoint SHA/CRC forensics remain `not_checked_explicit_user_waiver`;
operational reload and Hub transfer verification do not erase that waiver.

## Attribution and use constraints

This is a **noncommercial educational research** release with **no new learned-weight
license grant**. Specialized-v1.1 declares CC-BY4.0 for the selected subsets.[1]
The exact3.5 teacher card declares OpenMDW1.1.[2] Original BASE/tokenizer notices and
upstream ClimbMix CC-BY-NC4.0 conditions remain documented in
[USE_CONSTRAINTS.md](USE_CONSTRAINTS.md). A repack's MIT label does not waive upstream
conditions. No raw training rows, teacher-cache payload or training credentials are uploaded.

## Sources

[1] https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-Specialized-v1.1/raw/13fa979be2e7f7e62913eee0ec5e97c8fd6e24af/README.md

[2] https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-Base-BF16/raw/4ba52e8ffa79e9693936c1a819ac89f551ceeef6/README.md

[한국어 요약](README.ko.md)
'''


def prepare():
    v=json.loads((CONVERSION/'validation_result.json').read_text()); validate(v)
    accepted=json.loads((CONVERSION/'final_acceptance.json').read_text())
    assert accepted['source']==str(SOURCE) and accepted['output']==str(MODEL)
    assert {p.name:p.stat().st_size for p in MODEL.iterdir() if p.is_file()}==accepted['files']
    if STAGING.exists():
        assert (STAGING/'publication_manifest.json').exists(), 'incomplete staging; inspect before recovery'
        return inventory()
    STAGING.mkdir()
    for p in MODEL.iterdir():
        assert p.is_file() and os.access(p,os.R_OK)
        (STAGING/p.name).symlink_to(p.resolve())
    (STAGING/'README.md').write_text(card(ROOT,v))
    (STAGING/'README.ko.md').write_text('''# NemoChat mid-training + Super3.5 distillation\n\n[전체 모델카드](README.md)\n\n원본 BASE19074에서 새 optimizer/RNG/cursor로 시작한 **KD step1000 최종 HF 모델**입니다. SFT/채팅 모델이 아닙니다.\n\n- 실제 처리량: **524,288,000 positions /128,000 global samples**.1000 calls 중999회 Positive-LR, 초기1회 zero-LR.\n- Teacher: NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-Base-BF16, revision4ba52e8ffa79e9693936c1a819ac89f551ceeef6의 text backbone.\n- Primary objective: **0.5 LM +0.5 Ghost-tail KD**, Top32/T1/global-normalized FP16 log-probabilities.\n- 데이터: Specialized-v1.1의 Code Concepts67%/Algorithmic10%/Formal Logic3% + ClimbMix replay20%(명목 비율). 준비된1.9296B token pool 전체를 학습한 것이 아닙니다.\n- 원래3662-call WSD를 유지한1000-call 구간이며 실제 최종LR1e-5;37-call warmup 이후 stable 구간에서 종료했습니다.\n- 오류91 global batches는 forward 전에 all-rank 제외했고, 학습한 batch는 반복하지 않았습니다. 이는 저장장치/읽기 문제 해결 증거가 아닙니다.\n- 마지막16 calls 평균LM1.132520/KD0.547897/Primary Total0.840209.\n- 아래 그림은 실제 native raw 기록이며 진한 Loss 선은16-call 이동평균입니다. Validation은 바뀌는 iterator의 별도 호출이므로 고정 문항 성능 향상으로 해석하지 않습니다.\n\n![LR/Loss](graphs/lr_loss.png)\n![Grad Norm/부분 구간 시간](graphs/gradient_timing.png)\n![Native Validation 관측값](graphs/validation_snapshots.png)\n\n9,790,967,416개 parameter의 HF safetensors 변환 후 공식 Transformers5.12.1/NemotronH 전체 로드·finite 영/한 forward·4-token cached generation을 CPU에서 확인했습니다. trust_remote_code=False. 한국어 smoke 정답은 틀렸으며 이는 품질 검증이 아닙니다. native/HF logit parity 및 GPU/fused kernel/서빙 호환성은 미검증입니다.\n\n**KD 자체 품질 기여는 아직 미확정**입니다. 동일 입력1000-call ordinary1.0LM control과 BASE/KD/LM-only 고정 평가가 진행/대기 중이며 결과를 지어내지 않았습니다. 체크포인트 SHA/CRC 감사는 사용자 면제로 미수행입니다.\n\n이 저장소는 private 비상업 교육/연구용이며 새로운 weight license를 부여하지 않습니다. 원본 데이터/teacher/tokenizer 조건과 [사용 제약](USE_CONSTRAINTS.md)을 확인하세요.\n''')
    original=ROOT/'checkpoints/exports/nemochat_baby_base_bf16/USE_CONSTRAINTS.md'
    (STAGING/'USE_CONSTRAINTS.md').write_text(original.read_text()+'''\n## Continued pretraining and KD additions\n\nSelected Code-Concepts, Unconditional-Algorithmic and Formal-Logic data: NVIDIA\nNemotron-Pretraining-Specialized-v1.1 revision13fa979be2e7f7e62913eee0ec5e97c8fd6e24af;\nits pinned card declares CC-BY-4.0 for these subsets. Economics and Multiple-Choice\nwere not used. Preserve NVIDIA attribution and applicable notices.\n\nTeacher: NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-Base-BF16 revision\n4ba52e8ffa79e9693936c1a819ac89f551ceeef6, whose pinned model card declares\nOpenMDW1.1: https://openmdw.ai/license/1-1/ . Review applicable terms; this\npublication does not relabel the new teacher as governed by the older tokenizer\nlicense or grant additional rights to the student weights.\n\nNo teacher weights, teacher score cache or raw corpus rows are distributed.\nThe inherited noncommercial educational scope remains in force; no legal\nguarantee regarding trained-weight rights is made.\n''')
    shutil.copytree(RESULT/'figures',STAGING/'graphs')
    training=STAGING/'training';training.mkdir()
    cfg=json.loads((ROOT/'results/midtraining/offline2b-prefixtrain-1000-01/run-config.json').read_text())
    config={k:cfg[k] for k in ['sequence_length','global_batch_size','micro_batch_size','num_microbatches','seed','parallel','train_samples','train_iters','warmup_iters','decay_iters','weights']}
    config.update(completed_calls=1000,positive_lr_updates=999,processed_positions=524288000,optimizer=dict(lr=1e-5,min_lr=1e-6,adam_beta1=.9,adam_beta2=.999,adam_eps=1e-8,weight_decay=.01,clip_grad=1.,use_distributed_optimizer=False),kd=dict(mode='read',alpha=.5,top_k=32,top_p=None,min_k=1,save_dtype='fp16',decode_threads=1,prefetch_factor=2,msc_prefetch_depth=0,objective='megatron-lm-ghost-tail-fixed-alpha'),cache_temperature_contract=1,router_aux_loss_coefficient=1e-4)
    (training/'core_config.json').write_text(json.dumps(config,indent=2)+'\n')
    m=json.loads((ROOT/'datasets/midtraining/super35-kd-01/offline1920-v1/data/manifest.json').read_text())
    provenance=dict(specialized_repository=m['source_provenance']['repository'],specialized_revision=m['source_provenance']['revision'],raw_files=[{k:v for k,v in x.items() if k in ['key','path','size','sha256']} for x in m['source_provenance']['raw_files']],nominal_weights=m['weights'],prepared_train_tokens=m['audit']['train_tokens'],prepared_train_documents=m['audit']['train_documents'],reserved_validation_tokens=m['audit']['validation_tokens'],replay=dict(repository='karpathy/climbmix-400b-shuffle',revision='915333b4f8b8684f39aeaafea600fea6f43fb703',upstream='nvidia/Nemotron-ClimbMix',upstream_revision='5eaa64b9c0c85b7f56af01d7dffdb0795816b12b'),processed_positions_this_stage=524288000,cycling=False,quarantined_global_batches=91)
    (training/'data_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    shutil.copyfile(ROOT/'midtraining/patches/offline_kd_pr/megatron-bridge-0.6.1-cached-kd.patch',training/'megatron-bridge-0.6.1-cached-kd.patch')
    shutil.copyfile(ROOT/'results/midtraining/offline2b-prefixtrain-1000-01/charts/final-20261018-214512/observed.csv',training/'observed.csv')
    runtime={k:v[k] for k in ['status','source_step','class','official_transformers','trust_remote_code','device','full_model_reload','loading_info','all_reloaded_tensors_finite','tokenizer_bytes_exact_match','versions','parameter_numel','native_forward_parity','source_integrity']}
    runtime['cases']=[{k:c[k] for k in ['prompt','finite','generated_text','new_tokens','generation_cache','generation_scores_finite']} for c in v['cases']]
    (training/'hf_validation.json').write_text(json.dumps(runtime,indent=2,ensure_ascii=False)+'\n')
    manifest=dict(repo_id=REPO,private=True,format='HF Transformers safetensors',source_native_step=1000,parameter_numel=9790967416,checkpoint_integrity='not_checked_explicit_user_waiver',lm_only_comparison='pending',processed_positions=524288000,positive_lr_updates=999)
    (STAGING/'publication_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return inventory()


def inventory():
    return {str(p.relative_to(STAGING)):p.stat().st_size for p in STAGING.rglob('*') if p.is_file()}


def credential():
    spec=importlib.util.spec_from_file_location('existing_credential',ROOT/'pretraining/publication/replace_hf_base_transformers.py')
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    return value.credential()


def verify(api, token, revision, expected):
    from huggingface_hub import hf_hub_download
    info=api.repo_info(REPO,repo_type='model',revision=revision,files_metadata=True)
    assert info.private is True and info.sha==revision
    actual={p.rfilename:p.size for p in info.siblings if p.rfilename!='.gitattributes'}
    assert actual==expected, 'Remote payload inventory differs'
    for name in ['README.md','README.ko.md','config.json','model.safetensors.index.json','graphs/lr_loss.png','graphs/gradient_timing.png','graphs/validation_snapshots.png','training/core_config.json','training/data_provenance.json','publication_manifest.json']:
        p=Path(hf_hub_download(REPO,filename=name,revision=revision,token=token,local_dir=RESULT/'small-readback'))
        assert p.read_bytes()==(STAGING/name).read_bytes(), 'Small artifact readback differs'
    receipt=dict(status='published_and_verified',repo_id=REPO,revision=revision,private=True,format='HF Transformers safetensors',payload_files=len(expected),payload_bytes=sum(expected.values()),safetensors_shards=sum(x.endswith('.safetensors') for x in expected),remote_inventory_exact=True,card_figures_and_configs_exact_readback=True,source_step=1000,checkpoint_payload_audit='not_checked_explicit_user_waiver',remote_weights_redownloaded=False,files=expected)
    (RESULT/'publication_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2),flush=True)


def main():
    os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1';os.environ['HF_HOME']=str(RESULT/'hf-cache')
    os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepare-only',action='store_true');a=parser.parse_args()
    expected=prepare()
    if a.prepare_only:
        print(json.dumps(dict(stage=str(STAGING),files=len(expected),bytes=sum(expected.values())),indent=2));return
    from huggingface_hub import HfApi,CommitOperationAdd
    from huggingface_hub.errors import RepositoryNotFoundError
    token=credential();api=HfApi(token=token);assert api.whoami()['name']=='eagle705'
    returned=RESULT/'commit_returned.json'
    if returned.exists():
        verify(api,token,json.loads(returned.read_text())['revision'],expected);return
    owned=RESULT/'repo-created.json'
    try:
        current=api.repo_info(REPO,repo_type='model',files_metadata=True)
        assert owned.exists(), 'Occupied target was not created by this publication; refusing overwrite'
    except RepositoryNotFoundError:
        api.create_repo(REPO,repo_type='model',private=True,exist_ok=False)
        current=api.repo_info(REPO,repo_type='model',files_metadata=True)
        assert current.private is True
        owned.write_text(json.dumps(dict(repo_id=REPO,private=True,initial_revision=current.sha))+'\n')
    assert current.private is True
    assert all(p.rfilename=='.gitattributes' for p in current.siblings), 'Target not empty; inspect instead of overwriting'
    (RESULT/'upload-preflight.json').write_text(json.dumps(dict(repo_id=REPO,private=True,parent_commit=current.sha,files=expected),indent=2)+'\n')
    print('UPLOADING_VALIDATED_KD_STEP1000_HF_PAYLOAD',json.dumps(dict(files=len(expected),bytes=sum(expected.values()),repo=REPO)),flush=True)
    ops=[CommitOperationAdd(path_in_repo=n,path_or_fileobj=str(STAGING/n)) for n in sorted(expected)]
    commit=api.create_commit(repo_id=REPO,repo_type='model',revision='main',operations=ops,commit_message='Publish validated KD step1000 HF BF16 model with data, cached-KD recipe and observed curves',parent_commit=current.sha,create_pr=False,num_threads=4)
    returned.write_text(json.dumps(dict(repo_id=REPO,revision=commit.oid))+'\n')
    verify(api,token,commit.oid,expected)


if __name__=='__main__':
    try: main()
    except Exception as error:
        import traceback
        print('PUBLICATION_FAILED',type(error).__name__,'http_status',getattr(getattr(error,'response',None),'status_code',None),flush=True)
        for f in traceback.extract_tb(error.__traceback__): print('FRAME',Path(f.filename).name,f.lineno,f.name,flush=True)
        raise SystemExit(1)
