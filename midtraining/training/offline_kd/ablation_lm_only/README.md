# Coefficient-matched LM-only control B

B uses **0.5 LM**, compared with accepted C's **0.5 LM + 0.5 ghost-tail KD**. This is not an ordinary 1.0-LM training recipe. Native router auxiliary loss, expert bias, clipping and regularization are unchanged and separate from primary Total Loss.

`prepare.py` freezes all eight completed C update mappings and 128,000 actual input digests, matching teacher-era digests. Native full GPTDataset sampler remains seed43, 468736 samples / [468736,120,8], sequence4096, GB128/MB1, TP1/PP1/CP1/EP8/DP8. Native sampler index is physical*128 + micro*8 + rank. B replays those 1000 accepted physical batches in exactly that order; new faults fail all ranks before forward, without quarantine, fallback, retry or expectation rebasing. Inputs are cloned once and actual collated tensor digests are collectively rechecked before each forward.

Original BASE HF `checkpoints/exports/nemochat_baby_base_hf_bf16_02` is imported with the original tokenizer identity and reviewed embedding-name alias importer. Optimizer, RNG, cursor and scheduler reset to zero; no learned KD weights are loaded. Full WSD3662/warm37/decay366, LR1e-5/min1e-6 and native AdamW/DDP settings match C.

`native.py` calls native Bridge pretrain and native forward / masked_next_token_loss, scaling only the backward LM scalar by 0.5. LM Loss logging is unweighted, token-weighted and independently cloned; Total Loss is 0.5LM. Native saves include optimizer/RNG: step16 then every64/final1000 with retention3. Checkpoint payload SHA/CRC remains explicitly waived; input verification is not waived.

`supervisor.py` is owned by named user-systemd unit `nemochat-offline2b-lmonly-1000-01.service`, cooperative all8 lock, pinned image, no network, read-only repository and isolated writable output/scratch. It uses existing reviewed task-local staged SDK, never edits installed SDK. Fresh compiler caches avoid old absolute-path artifacts. Watchdog12h, no automatic retry. ExecStopPost stops only its exact owned container.

Output: `results/midtraining/offline2b-lmonly-1000-01/`. Final B checkpoint is reserved for parent-owned fixed BASE/B/C heldout loss and benchmark comparisons. This adapter runs no benchmark, export, publication, data generation or teacher/cache decoding.
