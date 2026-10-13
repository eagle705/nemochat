# CPU-First Implementation Walkthrough

Start with [README](../../README.md) for exact runnable commands and [verification report](../IMPLEMENTATION_VERIFICATION.md) for real outputs.

## Why these files are small and explicit

The tutorial uses direct functions and CLI arguments instead of a trainer inheritance tree, generic backend registry or configuration factory. Data acquisition, metrics, model dimensions and NIAH have different responsibilities and are intentionally separate. Small explicit data classes hold model dimensions, loss totals and WSD policy (`WSDConfig`); their arithmetic and boundaries are directly readable.

### 1. Metrics

`metrics.py`: add eligible target NLL/bytes, then compute CE/PPL/BPB. Center each task accuracy against its supplied baseline, average tasks equally. `token_bytes.py`: invert ByteLevel's reversible 256-byte alphabet, skip special IDs, check raw-text round trips. GPT-2 and the pinned Super tokenizer have a verified BPE/ByteLevel decoder; other decoder types fail closed rather than inventing approximate byte counts.

`hf_eval.py`: feed a reset token plus each document-local target chunk. A logit at position i predicts target i+1. This CPU protocol differs from a packed training stream. It returns numerator/denominator counts and preserves model train/eval state. `core.py`: render explicit plain prompts, determine prefix/suffix target spans, preserve mean-token-NLL option scoring and teacher-forced reference matching. `bundle.py`: size-bound and SHA-pin the official nanochat bundle in memory; no unsafe ZIP path extraction.

The adapter is **single-process Hugging Face only**. MCore distributed reductions are not silently simulated. Do not call it validated for the hybrid export until checkpoint/logit and distributed parity tests exist.

### 2. Model and tiny training control

`dimensions.py` counts A/B parameters analytically, including embedding/head and Mamba/attention/shared/routed/latent parts. The count convention is not FLOPs. `bridge_provider.py` gives a direct pinned-source constructor; GPU SDK not installed/exercised here.

`train_cpu.py` shows real next-token AdamW updates and model export. Resume restores safe-loaded optimizer tensors, Torch RNG and a deterministic chunk cursor; input data/config hashes must match. It is an HF mechanics control, not a CPU version of the 13.8B student. Real tiny-model forward/backward and resumed/uninterrupted equality were checked.

### 3. Data and NIAH

Read [data implementation](DATA_IMPLEMENTATION.md) and [long-context implementation](LONG_CONTEXT_IMPLEMENTATION.md). The selected data pack is flat uint32 with explicit record offsets; converting to the official MCore indexed dataset format is a later integration task. Selected-pool exact dedup is not global near dedup. Curator/DataDesigner processing remains future work rather than imports pretending to do curation.

NIAH fixture generation computes actual prompt tokens and reserves output capacity. `serve_cpu.py` exposes real greedy model generation on loopback only so network -> scorer -> plot can be exercised without vLLM. It is not production serving, and tiny GPT-2's retrieval failure says nothing about NemoChat. The schedule module emits long/short plans and cursors; the actual Bridge training controller still needs optimizer/dataloader synchronization.

## Boundaries to validate with B300

1. Install pinned Bridge/Core/TE in a dedicated compatible GPU container.
2. Instantiate A/B providers, verify actual numel and tiny forward/backward/checkpoint behavior.
3. Convert the flat data artifact using the official indexed-dataset builder; validate document/reset semantics for Mamba before packed training.
4. Integrate per-target MCore loss and unique-target TP/CP/EP/DP reductions for BPB.
5. Verify hybrid export into an inference backend and CORE/NIAH parity.
6. Attach separate long/short dataloaders and update-boundary schedule to Bridge with complete training state resume.
7. Then proceed to stage-specific SFT/RLVR/MOPD implementation, not fake placeholders now.

Tests containing fake tokenizers/endpoint clients are deliberately labeled **unit fixtures**. Only actual network/model smoke outputs are recorded as measured integration results. No claim of 64K/256K capability follows from this CPU foundation.
