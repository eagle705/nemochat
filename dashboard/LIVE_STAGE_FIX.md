# Live stage enrollment and timeline interaction correction

## Verified 2026-10-14 KST

The previous monitor enrolled only eight historical pretrain runs. Two approved one-level `run_roots` now enroll supported MID/SFT/RLVR directories automatically each collection. Prefix→stage rules are explicit; markers, IDs and paths are validated; symlinks/hidden directories/arbitrary recursion are excluded. Root/entry/run caps and incomplete-coverage warnings remain visible. Individual explicit specs win; no trainer writer/hooks or source/checkpoint changes.

Actual live service now contains13runs (8existing+5new). Parent reconciled source receipts:
- MID `bridge-mid-warmstart-02`:4updates,4096processed/loss-bearing tokens.
- SFT `bridge-sft-diagnostic-03`:4updates,16384processed tokens,3025loss-bearing tokens, final global weighted loss11.39974308013916.
- Previous MID/SFT failed runs retained as failed.
- RLVR `native-rlvr-01`: observed receipt, training_status blocked, scope rollout_only;4unique rollouts,512sampling calls,0optimizer updates,0scheduler samples,reward mean0. Exit0 does not establish policy learning.

Global SFT counters are per-update global deltas counted once, not multiplied by ranks. Global loss is producer token-weighted loss, not an unweighted average of rank losses. Small exact train-complete histories preserve coverage when JSONL tails are unsupported/truncated. RL adapter reads only safe verification/terminal files, not rollout prompts/IDs/token payloads. HOST phase projection emits only bounded allowed operation/timing fields; actualSFT98host intervals have update=null and unknown iteration identity and are NOT GPU/iteration traces. This optional backend metadata is not presented as a live kernel view.

## Interaction and scope

- Selected run defaults to newest recorded updated_at on first load and binds profile stage; explicit historical reference override remains available. Manual choices persist across refresh.
- MID/SFT/RLVR with no GPU capture cannot inherit pretrain events. `Historical capture / not live profiling` and source/run-binding caveats are conspicuous.
- Pointer tooltip, wheel zoom at pointer, drag pan, kernel search, stream filter, visible-window counts, selected kernel/API highlighting and raw labels on wide bars. All tiny events remain in chronological pages.
- Unchanged capture keeps DOM/hover/zoom/page through SSE; changed data/iteration invalidates geometry. Parent actual Chrome observed Update3→4 switch7025→6834events, AllGather filter25events, wheel window0–2541.882125→814.210390–1579.810574ms, drag883.867512–1649.467696ms, same canvas/search/customzoom after6.5s refresh. Real mouse hover/click selected raw AllGather row2003/correlation15483843 with source/timestamp detail and linked API.
- Metric/progress/hardware/table headings English; Korean explanatory/evidence prose retained.

## Execution

Parent254tests+32subtests passed; app/timeline Node syntax and diff check exit0. Focused suite only, not the whole GPU project suite. Raw `results/dashboard_live_fix/5eeb26c5dff3/{tests.log,tests.xml,live-readback.json}`. Two real SSE snapshots had distinct generated_at and13runs. Actual Chrome verified fresh defaultRL blocked display, MID/SFT metrics, no inherited timeline, desktop/mobile no document overflow and no browser errors. Automation's one6.5s awaited JS promise exceeded the CDP5s request timeout; an actual foreground wait and subsequent read verified persistence instead.

Screenshots:
- `results/dashboard-live-sft-fixed.png`
- `results/dashboard-live-rl-fixed.png`
- `results/dashboard-timeline-interactive-fixed.png`

Owned server restarted on127.0.0.1:8787 without readiness-watch notifications to avoid misleading late alerts. Current traces remain historical independent diagnostics; no new profiling/attach/GPU workload. Live metrics do not manufacture missing CUDA timing. Capture bandwidth remains a separate cost; no zero-overhead or live-all-stage-kernel-coverage claim.
