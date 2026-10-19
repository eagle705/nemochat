# Ordinary LM-only control B02 — supersedes coefficient-matched B01

Latest user correction selects **B=1.0LM** versus deployed **C=0.5LM+0.5 ghost-tail KD**. This measures practical recipe value, not the isolated effect of adding KD at a constant common LM coefficient. Primary Total Loss equals unweighted LM Loss; router auxiliary loss / expert bias remain native and separate.

Source: `ordinary02/native.py`, native Bridge pretrain, forward, masked_next_token_loss, DDP, AdamW, scheduler and full-state save. Original BASE HF `checkpoints/exports/nemochat_baby_base_hf_bf16_02` fresh0; original tokenizer and reviewed embedding alias importer preserved. No learned B01/C weights loaded. Adam beta1=.9/beta2=.999/eps1e-8, WD.01/clip1, BF16 TP1/PP1/CP1/EP8/DP8, seq4096/GB128/MB1. Seed43; full sampler468736/[468736,120,8], full WSD3662/warm37/decay366/LR1e-5/min1e-6 unchanged.

`prepare.py` authenticates completed C all8 mapping, 128000 actual input hashes and historical teacher input oracle. 1000 accepted physical batches through1090 (91 excluded in C) are replayed exactly via physical*128+micro*8+rank. No teacher/cache decoding/model is required. All16 inputs per rank are cloned and verified collectively before each optimizer call; actual collated tensors are checked collectively again before every forward. Any new read/digest/schema/index fault is fatal with evidence; no data reselection, retries, rebasing or fallback.

Native checkpoints save optimizer/RNG at16 then64-periodic/final1000, retention3. Callback verifies8 native distcp shards plus metadata before allrank transaction. Checkpoint payload SHA/CRC stays explicitly waived; input hashes are mandatory. Final B checkpoint is reserved for parent-owned fixed heldout/benchmark comparisons and publication; this service executes neither benchmark nor upload.

B01 was superseded and is preserved unchanged. Its zero-update launch also exposed a structural adapter error cloning NumPy scalar metadata, not a tensor digest mismatch. B02 preserves such metadata while cloning tensors; a failing regression reproduces that error before correction. TDD loss/replay/error/save/native RED/GREEN logs are in `results/midtraining/offline2b-lmonly-1000-02-preflight/`.

Owned service/container: `nemochat-offline2b-lmonly-1000-02.service`, pinned image, independent user-systemd,12h watchdog, no retries, exact owned-container stop hook, cooperative all8 lock, repository read-only and separate output/scratch/compiler caches. Results: `results/midtraining/offline2b-lmonly-1000-02/`.
