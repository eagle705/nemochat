# Durable same-suite ablation controller (inert until parent review)

Scope: `midtraining/offline2b/ablation_auto/` and `results/midtraining/offline2b-ablation-auto-01/`. The new B HF directory is created only during actual future execution. Existing learner, evaluator, converters and root ledger are never modified.

## Exact launch contract

From `/raid/yoon/nemochat`:

```bash
.venv/bin/python -m unittest discover -s midtraining/offline2b/ablation_auto -v
bash midtraining/offline2b/ablation_auto/start.sh          # inert, authenticated preview
# Parent reviews every allowlisted source and source-freeze.json, then:
bash midtraining/offline2b/ablation_auto/start.sh --start
```

Unit: `nemochat-offline2b-ablation-auto-01.service`, inherited UID1001, `Restart=no`, `RuntimeMaxSec=23h`, exact absolute Python/controller argv. `run.lock` is exclusively created; failure/completion cannot be implicitly retried. The start preflight checks unchanged source, clean generation, existing B identity and external A health. It does not wait or launch work. No new user approval is needed after the parent reviews this already-authorized handoff. The builder must NOT run `--start`.

`source-freeze.json` hashes an exact complete allowlist of code/configuration/receipts only. It deliberately excludes checkpoint tensor payloads. `freeze` is fresh-only; never silently refresh the manifest after review. Every future Engine create and final acceptance rechecks the freeze. A changed source is a blocker requiring parent re-review.

## Stage order and ownership

1. Bind B `nemochat-offline2b-lmonly-1000-02.service`, supervisor PID4083833 / UID1001 / Linux starttime47312487 / full exact argv, Engine `c3779c8241de4082eb37eb0c407f574b9505e6d2c366f55c72a7c4b5b5b1eeda`. Open/recheck a real pidfd; block once on kernel process exit, at most12h. Same-starttime zombies and absent processes reconcile against immutable final proof, never successful exit alone. Different identity, intentional stop, failure, supersession or no final1000 fails durably; never retrain/restart/shrink.
2. Verify exact clean B Engine exit0/noOOM, all8 x1000 native calls, 999positive+1zeroLR, finite positive Grad Norm, no skipped/nonfinite metrics, original BASE model-only reset and `1.0 LM` objective. Require all8 actual 16,000 per-rank input receipt IDs/order/digests and accepted physical mapping equal completed C, and equal per-call LR. Require final checkpoint1000 model/optimizer/RNG metadata, native synchronous full-state save, eight transaction acknowledgements, eight nonempty `.distcp` shards and `.metadata`. Trusted project metadata pickle only; no tensor SHA/CRC/forensic reads.
3. Check existing A has not failed; start C CPU evaluation without waiting for A. Create fresh B output `checkpoints/exports/nemochat_mid_lmonly_hf_bf16_02`, run existing reviewed converter metadata/schema-only gate, real model export, then official full HF reload/bilingual finite forward/cache/generation. Flags: `--source-step 1000 --stage midtraining_lm_only --model-prefix ''`. Actual selected B native weights only, no BASE substitution. Start B CPU evaluation immediately after B acceptance.
4. Preserve external A Engine `e1404d98ba226a85000984e8a7dee2dc758a797c997fbb46ea9ec1aaa845bfa2` and `results/midtraining/offline2b-ablation-eval-01/A-full01`. If running, use one bounded Engine `/wait` request (socket timeout >6h); if complete, reuse only its full report. Never start, stop or restart A. C and B use unchanged `ablation_eval/eval.py` in the same pinned CPU image, BF16/Transformers5.12.1/torch2.13.0a0, 4CPU and128GiB each, networknone/noGPU; each evaluation bounded6h. Foreground existing launch helpers are owned through two controller threads, not detached children. Export uses the existing CPU `engine_run.py`. Controller-specific owner labels, deterministic stage names, saved exact IDs and create/cleanup lock scope cleanup to these five stages only; external A/B and the separate B completion observer are never cleanup targets.
5. Require all3 `complete` reports: same frozen suite canonical SHA `07609a0cf53e4a63d555a3e50938adfed1cb05471d02b37fb93461e7879af19e`, exact400 IDs/1,600choices,64 heldout windows/65,536 targets and correct task/source denominators. Recompute metrics, normalized predictions, conditional log probabilities, NLL/PPL; read back every per-question receipt. Run existing paired `compare.py` for A–C, A–B, B–C; render actual benchmark and overall/per-source NLL/PPL PNGs in requested display order **[1]BASE, [2]LM+KD, [3]LM-only**. Write summary and English/Korean reports. Inspect real Engine exits, validate PNG decode and clean owned work before atomically publishing `complete.json` LAST.

## Terminal artifacts and delivery

Only actual all-three acceptance produces `results/midtraining/offline2b-ablation-auto-01/complete.json`, containing absolute image paths, all3 verified score-report paths and report SHA256s, suite/coverage/native/HF evidence. Any partial/stopped/failed path writes `failure.json`, never a success marker. ExecStopPost also preserves a failed terminal outcome if killed before acceptance. Parent binds a pidfd terminal watcher to this exact controller unit and sends the real final PNGs into the existing Slack thread. This module does not post or publish to HF.

PNG names include `BASE19074_A__KD1000_C__LM1000_B`. Scores may regress; no threshold or cherry-picking. MMLU is only10subjects x20questions5shot; ARC100test0shot; HellaSwag100validation0shot. These are nonstandard diagnostic subsets, not full MMLU/leaderboard results; contamination unknown. B1.0LM versus C0.5LM+0.5KD is practical recipe comparison, not coefficient-matched causal isolation. Paired bootstrap covers question/window sampling, not training-seed uncertainty. Original checkpoint SHA/CRC waiver is preserved.

## Verification boundary

CPU tests execute real Linux pidfd normal exit, unreaped zombie (waitid WNOWAIT), stale starttime/argv/UID rejection, timeout-with-child-alive and already-absent cases. Gates use actual C receipts/schema plus declared generated CPU fixtures, missing ranks/shards, partial coverage/backend/probability mutations, exact A blocking wait controls, foreground launcher adapter with a clearly mocked Engine, durable failure/terminal ordering, and actual matplotlib/Pillow PNG creation. Fixture figures are explicitly watermarked CPU TEST FIXTURE and are NOT benchmark results or production completion. Production B export and all-three acceptance remain unexecuted until the parent starts the reviewed unit.
