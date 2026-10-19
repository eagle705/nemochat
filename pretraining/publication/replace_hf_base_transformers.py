"""Guarded replacement of the explicitly approved private BASE repository."""
import json
import logging
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / 'checkpoints/exports/nemochat_baby_base_hf_bf16_02'
EXPORT = ROOT / 'checkpoints/exports/nemochat_baby_base_hf_publication_01'
RESULT = ROOT / 'results/pretraining/nemochat_baby/hf_transformers_publication'
CONVERSION = ROOT / 'results/pretraining/nemochat_baby/hf_transformers_conversion'
ORIGINAL = ROOT / 'checkpoints/exports/nemochat_baby_base_bf16'
REPO = 'eagle705/nemochat-10BA1B-BF16-base'
PREVIOUS = '94422e79dd31ede40e52a9940367115b690d5048'


def prepare():
    v = json.loads((CONVERSION / 'validation_result_02.json').read_text())
    assert v['status'] == 'passed' and v['full_model_reload']
    assert Path(v['model']) == MODEL and v['parameter_numel'] == 9790967416
    EXPORT.mkdir(exist_ok=True)
    # User-owned publication staging; immutable container-owned model files stay intact.
    for source in MODEL.iterdir():
        if not source.is_file():
            raise ValueError('unexpected directory in validated model bundle')
        target = EXPORT / source.name
        if target.exists() or target.is_symlink():
            assert target.is_symlink() and target.resolve() == source.resolve()
        else:
            target.symlink_to(source.resolve())
    assert v['official_transformers'] and v['trust_remote_code'] is False
    assert all(not x for x in v['loading_info'].values())
    assert v['tokenizer_vocab_exact_match'] and len(v['cases']) == 2
    assert all(x['finite'] and x['new_tokens'] > 0 and x['generation_cache'] for x in v['cases'])
    card = (ORIGINAL / 'README.md').read_text().split('## Files, loading, and compatibility')[0]
    card = card.replace('language:\n', 'library_name: transformers\npipeline_tag: text-generation\nlanguage:\n', 1)
    card = card.replace(
        'The exact provider fields, including the full hybrid layer pattern, are in\n'
        '`model_config.json`. This is a native configuration, **not** a Transformers\n'
        '`PretrainedConfig`; no fictitious `model_type` or AutoModel implementation is supplied.\n',
        'The converted model configuration, including the full hybrid layer pattern, is in\n'
        '`config.json` and uses the official Transformers `NemotronHConfig` /\n'
        '`NemotronHForCausalLM` implementation. The prior native provider configuration\n'
        'remains available in the original native revision listed below.\n',
    )
    card += '''## Learning Rate and Loss curves

The figures show recorded updates from the same completed BASE pretraining run, not SFT.

![Observed WSD Learning Rate](figures/learning_rate.png)

![Observed pretraining Loss and fixed-window Validation CE](figures/loss.png)

Training Loss includes raw updates and a trailing 100-update mean. Validation CE uses
one fixed, separately reserved ClimbMix heldout window reused across evaluations;
this is not a standardized benchmark. Green shading marks the final LR decay phase.

## Transformers loading

This revision replaces the previous native-DCP payload with **HF safetensors** and
the official Transformers `NemotronHForCausalLM` configuration. The original native
release remains in repository history at revision
`94422e79dd31ede40e52a9940367115b690d5048`; local native checkpoints are preserved.

The tested runtime is **Transformers 5.12.1**, Python 3.12.3, and PyTorch
`2.13.0a0+8145d630e8.nv26.06`. Compatibility with other versions is not established.
The complete 9,790,967,416-parameter BF16 model was loaded on **CPU**, with no missing,
unexpected, or mismatched keys. English and Korean finite forward passes and short
four-token greedy generations with KV cache succeeded. No remote model code is needed.

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

repo = "eagle705/nemochat-10BA1B-BF16-base"
# Authenticate beforehand for this private repository.
tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=False)
model = AutoModelForCausalLM.from_pretrained(
    repo,
    dtype=torch.bfloat16,
    trust_remote_code=False,
    attn_implementation="eager",
).eval()  # Tested on CPU; allow sufficient RAM for approximately 20 GB of weights.
inputs = tokenizer("The capital of France is", return_tensors="pt")
with torch.inference_mode():
    output = model.generate(
        **inputs, max_new_tokens=4, do_sample=False, use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
    )
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

Observed smoke outputs, not a benchmark or selected quality evaluation:
- `The capital of France is` → `The capital of France is Paris. It is`
- `대한민국의 수도는` → `대한민국의 수도는, 그 다음,`

## Conversion method and verification boundaries

Conversion uses the pinned Megatron Bridge Nemotron-H parameter registry plus an
explicit CPU global-DCP schema adapter for split Mamba records, group-interleaved
QKV, stacked expert tensors and the official Transformers module prefix. Bridge's
normal export entrypoint was **not** executed; this is not a claim that its unmodified
CLI supports this custom checkpoint. LatentMoE dimensions and original tokenizer
assets are preserved; FP32 router-correction buffers remain FP32. Native Mamba's
zero lower time-step bound is represented explicitly in the HF configuration.

The configuration disables optional Mamba fast kernels. GPU/fused-kernel execution,
native-versus-HF numerical logit parity, long-context behavior, vLLM and TensorRT-LLM
compatibility are **not verified**. Full load and short generation demonstrate
operational compatibility in the recorded CPU runtime, not broad answer quality or
exact equivalence to native kernels. The original checkpoint byte-integrity waiver
remains `not_checked_explicit_user_waiver`; no SHA/CRC forensic audit is claimed.

This is the **pretraining BASE**, not the separate SFT or RLVR model. Its pretraining
used LM Loss without teacher distillation. Original tokenizer revision
`2dc98e2afe4face0e4ce40972a915c45368bd34a` is unchanged, including reserved aliases.

## Use constraints

This is a noncommercial educational research artifact. The selected ClimbMix repack
has an upstream NVIDIA ClimbMix **CC-BY-NC-4.0** source; a repack's MIT label does not
remove upstream restrictions. See `USE_CONSTRAINTS.md` for attribution and use limits.
No additional license grant, safety assessment, or production-readiness claim is made.
'''
    (EXPORT / 'README.md').write_text(card)
    shutil.copyfile(ORIGINAL / 'USE_CONSTRAINTS.md', EXPORT / 'USE_CONSTRAINTS.md')
    (EXPORT / 'figures').mkdir(exist_ok=True)
    for name in ['learning_rate.png', 'loss.png']:
        shutil.copyfile(ROOT / 'results/pretraining/nemochat_baby/hf_card_figures' / name, EXPORT / 'figures' / name)
    manifest = {
        'repo_id': REPO, 'private_required': True, 'format': 'HF Transformers safetensors',
        'stage': 'pretraining_BASE_not_SFT', 'checkpoint_update': 19074,
        'previous_native_revision': PREVIOUS, 'parameter_numel': v['parameter_numel'],
        'training_positions': 10000269312, 'tokenizer_revision': v['source_tokenizer_revision'],
        'checkpoint_integrity': 'not_checked_explicit_user_waiver',
        'conversion': 'Bridge registry-derived CPU DCP adapter; normal Bridge export entrypoint not executed',
        'validation': {k: v[k] for k in ['class', 'official_transformers', 'trust_remote_code', 'device', 'full_model_reload', 'loading_info', 'versions', 'native_forward_parity']},
        'plots': ['figures/learning_rate.png', 'figures/loss.png'],
    }
    (EXPORT / 'publication_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    RESULT.mkdir(parents=True, exist_ok=True)
    return {str(p.relative_to(EXPORT)): p.stat().st_size for p in EXPORT.rglob('*') if p.is_file()}


def credential():
    token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
    if not token:
        for line in (ROOT / '.env').read_text().splitlines():
            if '=' not in line:
                continue
            key, value = line.split('=', 1)
            if key.strip() in {'HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN'}:
                token = value.strip().strip('\"\'')
                if token:
                    break
    if not token:
        raise RuntimeError('HF credential unavailable')
    return token


def verify(api, token, revision, expected):
    from huggingface_hub import hf_hub_download
    info = api.repo_info(REPO, repo_type='model', revision=revision, files_metadata=True)
    assert info.private is True and info.sha == revision
    actual = {f.rfilename: f.size for f in info.siblings if f.rfilename != '.gitattributes'}
    assert actual == expected, 'remote file inventory mismatch'
    for name in ['README.md', 'figures/learning_rate.png', 'figures/loss.png', 'config.json', 'publication_manifest.json']:
        local = Path(hf_hub_download(REPO, filename=name, revision=revision, repo_type='model', token=token, local_dir=RESULT / 'small_readback'))
        assert local.read_bytes() == (EXPORT / name).read_bytes(), 'small artifact readback mismatch'
    assert PREVIOUS == api.repo_info(REPO, revision=PREVIOUS, repo_type='model').sha
    receipt = {'repo_id': REPO, 'revision': revision, 'private': True, 'previous_native_revision_preserved': PREVIOUS, 'payload_files': len(expected), 'payload_bytes': sum(expected.values()), 'expected_files': expected, 'remote_inventory_exact': True, 'model_card_and_figures_exact_readback': True, 'remote_weight_redownload': False, 'native_logit_parity': 'not_run', 'status': 'published_and_verified'}
    (RESULT / 'publication_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2), flush=True)


def main():
    os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
    logging.getLogger('huggingface_hub').setLevel(logging.CRITICAL)
    from huggingface_hub import HfApi, CommitOperationAdd, CommitOperationDelete
    expected = prepare()
    token = credential()
    api = HfApi(token=token)
    returned = RESULT / 'commit_returned.json'
    if returned.exists():
        revision = json.loads(returned.read_text())['revision']
        verify(api, token, revision, expected)
        return
    current = api.repo_info(REPO, repo_type='model', revision='main', files_metadata=True)
    assert current.private is True, 'target must already be private'
    assert current.sha == PREVIOUS, 'remote main changed; refusing overwrite'
    operations = [CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(EXPORT / name)) for name in sorted(expected)]
    removed = [f.rfilename for f in current.siblings if f.rfilename not in expected and f.rfilename != '.gitattributes']
    operations += [CommitOperationDelete(path_in_repo=name, is_folder=False) for name in removed]
    (RESULT / 'preflight.json').write_text(json.dumps({'repo_id': REPO, 'private': True, 'parent_commit': current.sha, 'expected_files': expected, 'removed_current_tree_files': removed}, indent=2))
    print('ACCEPTED_LOCAL_AUTOMODEL_GATE; private target confirmed; uploading complete HF payload', flush=True)
    commit = api.create_commit(repo_id=REPO, repo_type='model', revision='main', operations=operations, commit_message='Replace native BASE payload with validated Transformers BF16 model and training curves', parent_commit=current.sha, create_pr=False, num_threads=4)
    revision = commit.oid
    returned.write_text(json.dumps({'repo_id': REPO, 'revision': revision}, indent=2) + '\n')
    verify(api, token, revision, expected)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        status = getattr(getattr(error, 'response', None), 'status_code', None)
        print(f'HF_REPLACEMENT_FAILED type={type(error).__name__} http_status={status}', file=sys.stderr)
        import traceback
        for frame in traceback.extract_tb(error.__traceback__):
            print(f'FRAME {Path(frame.filename).name}:{frame.lineno} {frame.name}', file=sys.stderr)
        sys.exit(1)
