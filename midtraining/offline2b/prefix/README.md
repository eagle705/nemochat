# Authenticated cached-prefix native student

**Status: training stopped by user request on 2026-10-18.** The execution and handoff descriptions below are historical, not active services or authorization to restart.

The original first64 results and embedded LR/Loss figure are documented in [English](../../../docs/en/MIDTRAIN_OFFLINE_KD_FIRST64_RESULTS.md) and [한국어](../../../docs/ko/MIDTRAIN_OFFLINE_KD_FIRST64_RESULTS.md), with a verified 64-row CSV, computed summary, checkpoint evidence, and limitations. Later restarted/filtered/dynamic generations are not combined with this trajectory.

This opt-in segment uses closed shards `cp0_dp{0..7}__{0..1142}.tar` from the deliberately stopped `offline2b-production1920-01` teacher. It does not mark the full teacher cache ready or restart its producer.

- Full config `C`, cache identity, seed 43, source mixture, native GPTDataset indices and sample budget remain `[468736, 120, 8]`.
- Full native WSD remains 3662 optimizer calls, warmup 37, decay 366, LR 1e-5/min 1e-6. Segment stop is 1143, independently of `C`.
- `DatasetProvider.build_datasets` explicitly delegates to the native GPTDataset builder with the full budget rather than the shortened `TrainingConfig` context. Every training microbatch is compared to the producer's exact per-rank input digest before forward.
- The native cached-KD adapter, Adam, DDP/EP8, FP32 gradient reduction, clip 1.0, router settings and checkpoint optimizer/RNG state remain enabled. There is no teacher in the student container and no handcrafted optimization loop.
- The first unit checkpoints every 64 calls. A durable handoff waits for the all-rank synchronous checkpoint64 transaction, plots the first64 real calls, intentionally stops only that unit/container and resumes its full state in a fresh `continuation/` output. Unsaved calls after checkpoint64, if any, are discarded explicitly. The immutable initial source remains unchanged while live.
- Continuation evaluates every250, preserving the full validation budget. This avoids exhausting the unchanged 120-sample validation iterator by evaluating every64 through the whole prefix. Initial checkpoint64 is retained separately from continuation checkpoint rotation.
- Native model/optimizer payload integrity is **not_checked_explicit_user_waiver**. No payload hash audit or extra model copy is performed. Resume success must be established by actual native restore and subsequent updates; sidecars alone do not establish it.

Durable user-systemd units:

1. `nemochat-offline2b-prefixtrain-01.service`
2. `nemochat-offline2b-prefixtrain-01-handoff.service`
3. `nemochat-offline2b-prefixtrain-01-continuation.service` (retained failed attempt)
4. `nemochat-offline2b-prefixtrain-01-continuation02.service` (fresh-cache recovery)

The first continuation failed before checkpoint restoration because copied Inductor artifacts retained absolute paths to the old read-only scratch workspace; Triton autotune attempted to write there. The failed source/output/container receipt is retained. The `continuation02/` source and `continuation-02/` output use an empty local Inductor cache and a fresh HOME/Triton cache, with no generated compiler-code rewriting or SDK upgrade. Initial checkpoint64 and its source remain unchanged. This is one explicit source fix and fresh generation, not an automatic retry policy.

Receipts and real rank metrics are under `results/midtraining/offline2b-prefixtrain-01/`. The owned controller holds `gpu-all8.lock`, rejects occupied GPUs, pins the approved image, uses NVIDIA DeviceRequests with `runc`, disables networking and stops only its owned container on its 12-hour deadline. No automatic retry or teacher fallback exists.

Refresh the recorded-data plot:

```bash
.venv/bin/python midtraining/offline2b/prefix/charts/render.py \
  --output results/midtraining/offline2b-prefixtrain-01 --limit 64
```

The PNG distinguishes raw versus trailing-8 mean LM, Ghost-tail KD and primary Total. Router auxiliary loss is separate. LR is sampled before the actual native optimizer call; the first zero-LR call is counted separately. Only updates present and finite on every rank enter the chart. No future Loss is synthesized.
