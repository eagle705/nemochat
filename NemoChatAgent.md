# NemoChat Agent — Goals, Execution Contract, and Work Log

> Read this file before making changes. It is the durable handoff for Codex and other coding agents working on NemoChat. Keep the goals, task ledger, evidence links, and append-only log current. Do not treat a plan, a file's existence, or an agent's report as proof that a feature works.

## 1. Mission and current goal

### Latest user-selected positioning — agent-assisted training systems lab

Prioritize depth in GPU training engineering and explicit YAML experiment contracts, not another agent application. Complete one dense training control with observation → hypothesis → measurement → correctness → full-update comparison. Read `docs/en/TRAINING_SYSTEMS_LAB.md`, `AGENT_EXPERIMENT_CONTRACT.md` and `TRAINING_PERFORMANCE_TOOLS.md` before changing this path. Existing hybrid, WSD/WSM, official Gym and operations invariants remain intact; standalone dense-control results do not satisfy their native integration gates.

### Operations extension (reviewed CPU/offline scope)

The user additionally selected: **bounded representative sampling for a too-large corpus**, **Hugging Face checkpoint upload/download across rentals**, and **W&B registration/monitoring for each stage**. Read `docs/en/OPERATIONS_PIPELINE.md` and its Korean companion before implementing these additions. Prior commit-readiness approval applies only to the earlier reviewed snapshot, not unreviewed additions.

Defaults: private HF repositories, disabled/explicit offline W&B, no remote side effects until destination/credentials/permissions are confirmed. Preserve raw resumable state separately from model-only exports; validate immutable revisions and artifact inventories. Stage contracts for unimplemented GPU/SFT/RLVR/MOPD paths must not be labeled executed.


**Project:** build a noncommercial, educational NVIDIA SDK reference tutorial for a randomly initialized Nemotron-inspired hybrid Latent MoE: data construction → pretraining → mid-training → long-context extension → SFT → RLVR → Super-teacher MOPD → serving.

**Current user-selected goal:** implement and verify pretraining with a **warmup–stable–decay (WSD) learning-rate schedule**, evaluate stable-phase checkpoints using **Megatron-LM PR #5114 weighted checkpoint merging / WSM**, and compare raw versus merged checkpoints using suitable **official NeMo Gym `benchmarks/` tasks**. Preserve held-out CE/PPL/BPB as diagnostic metrics and the existing nanochat-style CORE code.

Success means a runnable, reproducible implementation with real execution evidence. Actual benchmark improvement is an experimental question, not a required or promised outcome. Report unchanged scores and regressions honestly.

**Working root on the handoff host:** `/home/ubuntu/nemochat`. Paths elsewhere in this file are repository-relative unless stated otherwise. Resolve the root again on another machine.

### Exact method chosen by the user

- Use the technique implemented in https://github.com/NVIDIA/Megatron-LM/pull/5114, which describes Nemotron 3 Super's stable-phase merge evaluation.
- Continue the original WSD training trajectory. At each eligible stable-phase milestone, merge a historical sliding window ending at that milestone and evaluate it against that milestone's raw checkpoint.
- Use the official pinned merge implementation, initially its **`minus-sqrt`** coefficient style. A `linear` merge may be a separately labeled ablation, not a silent substitute.
- **Do not implement decay-branch checkpoint averaging as the selected method.** It was suggested, then superseded by the user's explicit PR selection.
- WSD LR annealing and WSM coefficient schedules are different mechanisms. Merging does not change the optimizer's learning rate or replace the training trajectory.
- Do not assume every checkpoint improves after merging. Upstream Nemotron gains are not NemoChat measurements.

### Added pretraining performance goal

Implement the Nsight Systems → optional Triton fusion performance lab in `docs/en/PRETRAIN_PERFORMANCE_LAB.md` (Korean companion available). Diagnose the actual critical path before choosing a target; compare eager/compiled reference, validate forward/backward, and measure unprofiled native full-update throughput and memory. Preserve WSD/WSM and reference defaults. Documentation is ready; implementation and GPU evidence are pending.

### Independent inference performance goal

Follow `docs/en/INFERENCE_PERFORMANCE_TRACK.md` and `docs/en/INFERENCE_PERFORMANCE_TOOLS.md`. This track complements training, beginning with CPU saved-request analysis and single-operator roofline tools; native serving, quality, load generation, profiler-backed optimization and upstream contribution remain separate gates. Use one supported pretrained control before assuming custom hybrid backend support.

### Observability and modeler-facing profiler tooling goals (G11/G12)

The user selected two separate repository folders:

- `tutorials/nemo_lens/`: a real NeMo Lens tutorial covering SDK execution, local trace UI startup/navigation, and measured overhead. Compare uninstrumented, disabled, enabled local export and enabled OTLP modes. CPU controls, exporter wiring, rendered UI and native distributed/GPU overhead are separate evidence levels. Never claim telemetry has zero overhead without a scoped measurement.
- `plugins/nemo_lens_nsight_bridge/`: an opt-in **collection/configuration plugin and Python instrumentation** focused on making modern MoE GPU operations understandable to modelers inside the existing Nsight Systems workflow. Preserve original kernel names and explain roles using evidence. Connect actual profiled Python source snapshots to operation annotations; kernel-to-source mapping requires launch/backtrace/compiler evidence. No standalone web viewer as the primary product. Native custom GUI explanation/snippet panels remain an unverified extension gate, not an implemented feature.

Reuse Lens run/step/layer context and existing backend NVTX where available. An educational CPU sparse-MoE control is not a native fused Nemotron/Megatron EP implementation; CPU span durations are not CUDA kernel timing. Do not add fake communication events or infer bottlenecks from names alone. Read the two folders' English/Korean READMEs and `docs/en/OBSERVABILITY_TRACKS.md` before continuing these packages. Current local GPU driver and `nsys` availability must be checked again on the execution host.

### RL rollout/tool bottleneck tutorial extension (G13)

The user additionally requested rollout diagnosis: model generation versus tool calls, caller/HTTP waits, verification/reward and tail trajectories. Extend the Lens tutorial under `tutorials/nemo_lens/rollout/`. Read `docs/en/RL_ROLLOUT_OBSERVABILITY.md` and Korean companion. Actual local calls and injected-delay scenarios teach tracing; they do not establish learned tool policy, native NeMo RL/Gym/vLLM execution, GPU queue timings or RL throughput. Preserve context by explicit IDs/W3C propagation, reject nonlocal/credentialed/redirected requests, do not export prompts/tool payloads by default, and never sum overlapping parent/child/concurrent spans as wall time. Native Gym has no tool_call group: HTTP client/server tool-route tracing and connection-pool metrics are distinct.

## 2. Read these sources before coding

1. `README.md`: present implementation boundary and concrete CPU entry points.
2. `docs/NEMOCHAT_PROJECT_PLAN.md`: full architecture, SDK constraints, staged budget, and seven-stage plan. Most GPU milestones are still plans.
3. `docs/IMPLEMENTATION_VERIFICATION.md`: historical real execution and independently reviewed CPU scope.
4. `docs/en/PRETRAIN_WSD_WSM_EXPERIMENT.md`: current experiment and raw/merged comparison contract.
5. `docs/en/CPU_IMPLEMENTATION.md`, `docs/en/DATA_IMPLEMENTATION.md`, `docs/en/LONG_CONTEXT_IMPLEMENTATION.md`: existing implementation behavior.
6. `docs/en/PRETRAIN_METRICS_AND_CLIMBMIX.md`: metric definitions, dataset/token accounting and licensing.
7. `docs/en/LONG_CONTEXT_AND_NIAH_PLAN.md`: long-context requirements and limitations.
8. `docs/en/EXECUTION_AND_BLOG_SERIES_PLAN.md`: evidence and publication handoffs for Parts I/II/III.
9. Corresponding `docs/ko/` companions and `THIRD_PARTY_NOTICES.md`.
10. Existing tests, `pyproject.toml`, and the actual modules/scripts you are changing.

If a new decision conflicts with an older plan, preserve the historical record, identify the superseded decision, and update affected canonical documents. Do not overwrite old verification results with new claims.

### Authoritative upstream references inspected during this handoff

These are investigation revisions, not a proven compatible GPU dependency lock. Verify them before execution and record any justified change.

| Component | Source / inspected revision | Use |
|---|---|---|
| Megatron-LM merge | PR #5114; inspected PR head `ca4826e9555032b8710dfb4a4e5693c46c1613dd` | `tools/checkpoint/weighted_merge.py`, coefficient semantics, CPU DCP merge, failure rules |
| Megatron Bridge | `NVIDIA-NeMo/Megatron-Bridge` at `8e2fdb88de2e9d842f28d00120df1af30acffa8a` | `src/megatron/bridge/training/config.py`, `docs/training/optimizer-scheduler.md` |
| NeMo Gym | **`NVIDIA-NeMo/Gym`** at `804c12a7d56a151a98ae150a352350e3c856ba47` | Official `benchmarks/`, preparation, runner, prompts and scorers |

Links:
- https://github.com/NVIDIA/Megatron-LM/pull/5114
- https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/8e2fdb88de2e9d842f28d00120df1af30acffa8a/docs/training/optimizer-scheduler.md
- https://github.com/NVIDIA-NeMo/Gym/tree/804c12a7d56a151a98ae150a352350e3c856ba47/benchmarks

Do not guess upstream command names, config fields, or paths. Inspect pinned source and CLI help. Record exact revisions, local patches, and environment/container digest. A merged PR does not imply it exists in every installed release.

## 3. Project invariants

### Architecture and hardware

- Candidate **A**: hidden 2048, latent 512, routed expert FFN 4096, 128 experts, top-k 6; analytical estimate approximately 13.773B total / 2.003B active parameters.
- Candidate **B**: latent 1024, routed expert FFN 2048; approximately 13.821B total / 2.052B active.
- Only A/B are approved pilot candidates. Non-latent C is reference-only, not an authorized extra training run.
- Nano-inspired 52 hybrid sub-blocks: 23 Mamba2 / 23 MoE / 6 Attention. Expert activation is squared ReLU, not automatically SwiGLU. Follow the detailed model plan and provider code.
- Super tokenizer identity must match at token-ID/file-hash/special-token/template level, not just vocab size.
- User's intended training hardware: standard-memory **B300, 8 GPUs, one NVLink node**, rented in week-long sessions. Inspect actual hardware/topology/storage before launching. This does not describe the current CPU host.
- Existing latent-MoE plan requires Transformer Engine and disables shared-expert overlap where the inspected Core implementation rejects that combination. Recheck version-specific support.
- Training is from random initialization; this is not a reduced pretrained Nano/Super checkpoint.

### Readability, language, and rights

- Canonical code, comments, docstrings, configs and English documents are English. Maintain separate Korean explanation under `docs/ko/`. User-facing progress may be Korean.
- Prefer small explicit functions and directly runnable scripts. Avoid generic Trainer hierarchies, registries, factories or adapter frameworks unless a concrete requirement warrants them.
- Explain scheduler boundaries, metric denominators, checkpoint layout and token accounting; do not hide them behind abstraction.
- Selected ClimbMix subsets are streamed/bounded, not full-corpus downloads. Preserve validation exclusion, dedup scope, quotas, tokenizer provenance and source rights.
- Original ClimbMix noncommercial conditions remain relevant despite community derivative MIT metadata. Audit redistribution separately.
- No unrequested commits, pushes, publications, paid judge calls, GPU rentals, large downloads, destructive cleanup or overwriting checkpoints. Ask for a budget/authorization when these become necessary.

## 4. Baseline and evidence boundary

### Historical verified baseline — do not confuse with a fresh run

`docs/IMPLEMENTATION_VERIFICATION.md` records:
- Last independently reviewed CPU foundation suite: **118 passed, 23 subtests passed**; parent rerun and `git diff --check` passed.
- Tiny HF model actual optimizer/save/resume: resumed and uninterrupted model weights, optimizer/RNG tensors and data/token counters matched exactly.
- Real tiny-model CE/PPL/BPB and a limited COPA subset; these are not NemoChat quality or full CORE results.
- Bounded real ClimbMix shard acquisition and actual Super tokenizer accounting.
- NIAH HTTP → scoring → plot pipeline exercised with six tiny-model requests; actual retrieval score 0/6, not a long-context capability result.
- Reviewed nonfinite CORE handling, authenticated client transport/error handling, quota stopping and malformed CPU server input rejection.

These numbers are historical evidence for that scope. Rerun the current suite before claiming the new workspace passes.

### Not established by that baseline

Actual NemoChat GPU forward/backward, distributed training/resume, complete model parameter census, MCore dataset conversion, TP/CP/EP metric parity, hybrid checkpoint export/serving parity, 64K training, SFT/RLVR/MOPD, and Dynamo serving were not executed. A provider/config object is not a successful GPU model run.

The current WSD/WSM/Gym extension had been delegated to parallel implementation workers before this handoff. At the handoff inspection, `scripts/merge_checkpoints.py`, `scripts/verify_wsm_merge.py`, `scripts/gym_eval.py` existed and `scripts/train_cpu.py` was present. **Their existence is not evidence of completion or approval.** Inspect their contents and tests, reconcile concurrent changes, and verify independently.

### Evidence labels required in every report

Use: `planned`, `source-verified`, `fixture-tested`, `CPU-model-executed`, `GPU-smoke-executed`, `real-benchmark-executed`, or `blocked/not-tested`.

Keep distinct:
- a scheduler unit test vs actual optimizer use;
- a synthetic DCP merge vs a real NemoChat checkpoint merge;
- a result-import fixture vs official Gym execution;
- checkpoint save/load vs correct model forward/export;
- small CPU control scores vs target-model quality;
- source/configuration support vs measured throughput or context capacity.

## 5. First actions for the incoming agent

1. Read this file and the source map above.
2. Inspect `git status --short`, current HEAD if any, active processes, available Python/venv, dependency versions, disk capacity, GPU availability and existing run directories. Record what is actually present.
3. **Resolve concurrency before edits.** The previous session dispatched WSD, WSM and Gym workers with disjoint ownership. Establish whether they are still writing. Do not modify their files concurrently; coordinate the handoff or work in an isolated worktree/branch. Preserve their artifacts even if incomplete.
4. Inventory implementation and tests, then run the current baseline once writers are quiescent. Use `.venv/bin/python -m pytest -q` if that venv is valid; otherwise build an isolated environment. Do not assume an old package list is current.
5. Update the task ledger and append a startup log entry with baseline results and gaps. Choose one concrete next task and complete its tests and execution before expanding scope.
6. Inspect new script `--help` before issuing commands; never present invented CLI flags as runnable instructions.

The handoff inspection found `.gitignore` modified and major project folders untracked. Existing work is user-owned regardless of Git tracking. Never use `git reset --hard`, `git clean`, bulk replacement or broad staging to reconcile this state.

## 6. Work packages and acceptance criteria

### G1 — WSD scheduler and resume

**Goal:** make LR progression explicit and reproducible, first in the existing tiny CPU control, then in the pinned Bridge training path.

Requirements:
- Separate invocation length (additional updates now) from the full schedule horizon. Keep the horizon/config invariant across resume/rental sessions.
- Define warmup/stable/decay boundaries and the LR used for each optimizer update; specify minimum LR and decay style. Validate finite parameters, nonnegative phase lengths and legal total duration.
- Preserve old constant-LR behavior and compatibility deliberately; never silently accept changed data/tokenizer/scheduler contracts on resume.
- Record global update, LR, schedule phase, processed/loss-bearing tokens and cursor. Save optimizer, scheduler configuration/state, RNG and data state needed for exact resume.
- Optional periodic checkpoints use new non-overwriting paths. Stable checkpoints for WSM must be captured with actual iteration/token metadata.
- Bridge source exposes `lr_decay_style="WSD"`, `lr_warmup_iters`, `lr_decay_iters`, `lr_wsd_decay_iters`, `lr_wsd_decay_style`. Inspect optimizer LR field placement in the pinned runtime.
- Explicitly configure annealing duration: the inspected Bridge code defaults an unspecified WSD decay duration to the entire decay horizon. Do not accidentally eliminate the intended stable plateau.
- Do not mix iteration-based and sample-based scheduler parameters or assume CPU control boundary semantics equal Bridge semantics without comparison.

Acceptance:
- Boundary and invalid/nonfinite input tests pass.
- Actual tiny-model continuous and split/resumed training match weights, optimizer/RNG, scheduler behavior, cursor and token metadata.
- Evidence/logs identify the control as not NemoChat training.
- GPU Bridge integration remains separately labeled until a real execution succeeds.

### G2 — Official PR #5114 WSM merge

**Goal:** a pinned, runnable upstream CPU merge path and independent output verification.

Requirements:
- Use official `tools/checkpoint/weighted_merge.py`, not a homegrown HF mean renamed WSM. Keep upstream checkout/cache separate from project source; record revision and patches.
- Confirm chronological selection and **exact upstream coefficient semantics**. `minus-sqrt` is not an arbitrary hand-written recency weighting. PR range mode weights checkpoint positions, not assumed token distances.
- Historical window ends at milestone T, uses no future/warmup/decay checkpoints unless explicitly defined as a different experiment, and contains enough compatible checkpoints.
- Official tool works on compatible same-layout `torch_dist` DCP checkpoints and uses fp32 CPU accumulation. It does not construct the target model.
- Validate key sets, global shapes, chunk layout, dtype and supported format. Transformer Engine `_extra_state` is copied from a selected source, not averaged; record the choice.
- Trusted inputs only: common and extra state can require checkpoint deserialization. Metadata-driven execution is not a security sandbox.
- Resolve optimizer/non-model state policy from the pinned tool. Never silently average or inherit optimizer/RNG state as a valid merged training trajectory.
- Preserve raw training checkpoints; merged model is evaluated separately. Fresh training from merged weights is a separate experiment requiring explicit optimizer/RNG reset.
- Output/provenance publication must not overwrite existing checkpoints. Record paths, resolved inputs, coefficients, iteration/token counters, dtype, extra-state source, revision, timings and memory where available.
- Do not assume unsupported token-window flags. Use recorded consumed-token counters for token-based selection; nominal batch×length is insufficient with variable lengths, packing or skipped work.

Acceptance:
- Run the **official** tool on trusted generated tiny DCP fixtures, then independently reload and compare merged tensors to expected weighted results.
- Test incompatible layouts/dtypes/keys, insufficient/missing inputs and existing output rejection, as supported by upstream and wrapper.
- Exercise copied extra-state behavior and inspect provenance.
- Label fixtures as mechanics evidence. Real hybrid checkpoints require a separate merge → model load → inference/export gate.
- If dependency/runtime blocks official execution, log exact errors and supported alternatives. Do not substitute fabricated results or call wrapper unit tests an upstream merge pass.

### G3 — NeMo Gym pretraining evaluation

**Goal:** official benchmark evaluation of raw/merged model arms and honest paired reporting.

Requirements:
- Discover tasks from pinned `NVIDIA-NeMo/Gym/benchmarks/`; inspect each chosen task's README/config/preparation/runner/scorer before selecting a pack.
- Freeze a small feasible quick pack and a broader milestone pack. Do not silently choose tasks requiring unsupported chat templates, tool execution, judge credits or model endpoints.
- Base-model prompt suitability matters. If adaptation is needed, identify the protocol as adapted, preserve the official reference, and apply the same adaptation to both arms.
- Connect to a real supported model endpoint/export path. An endpoint returning JSON is not proof of hybrid model or tokenizer correctness.
- Use official scoring paths. A custom result importer/comparator is useful but is not itself an official benchmark run.
- Raw and merged arms share task/data revision, exact sample IDs/order, prompt/template, tokenizer, decoding settings, stop rules, generation budget, scorer and sampling seeds/repeats where applicable.
- Preserve predictions, per-example results, errors, exclusions, task counts, denominator and units. Transport/auth/timeouts/unsupported results must not silently become incorrect answers or zero scores.
- Report per-task raw score, merged score and paired delta, with uncertainty where supported. Do not aggregate incompatible scales or report a single unexplained average. Use percentage points only for percentage metrics.
- Retain CE/PPL/BPB with unchanged held-out data/context/token-byte protocol; keep existing CORE but do not label a partial subset as full CORE.
- Include data contamination/split review and explicit budget for expensive benchmarks.

Acceptance:
- Tests cover comparison contract mismatches, missing/failed tasks and score units.
- CLI/import path is actually exercised; fixtures clearly labeled.
- At least one actual official task must run against a real supported model before claiming Gym evaluation works end-to-end. Tiny-control run, if suitable, proves wiring only.
- A real NemoChat raw-versus-merged quality claim requires both target-model arms on the same frozen pack. If hardware/serving is unavailable, deliver the runnable validated preparation plus a precise blocker, not an invented score table.

### G4 — Integration, review and documentation

**Goal:** join the pieces without confusing isolated tests with a complete pretraining system.

- Establish the target chain: data/tokenizer contract → pinned Bridge training → resumable raw stable checkpoints → official WSM → verified model load/export → Gym evaluation → paired report.
- Add a readable milestone orchestration script only after each component contract is clear; no stub that prints fabricated success.
- Preserve environment and input/output manifests. Run directory names are not sufficient provenance.
- Rerun targeted tests and the full suite after all writers finish. Run `git diff --check`; include new/untracked files in review and syntax checks because Git diff alone omits them.
- Independent review should cover correctness, security, concurrency, checkpoint trust, reproducibility and evidence claims. Reproduce/fix blockers, then rerun and request scoped re-review.
- Update `README.md`, English/Korean implementation docs, `docs/IMPLEMENTATION_VERIFICATION.md` and this ledger/log with exact execution and remaining blockers.

### G8 — Pretraining Nsight/Triton performance lab

Read `docs/en/NSIGHT_CODEX_GUIDE.md` for read-only CSV/SQLite analysis, schema and correlation pitfalls, the Codex task and GUI confirmation. No analyzer is implemented by that document.

Acceptance is P0–P3 in `docs/en/PRETRAIN_PERFORMANCE_LAB.md`: reproducible native GPU baseline, trace-based candidate selection, optional isolated forward/backward fusion, and correctness plus unprofiled full-update/memory A/B with checkpoint/resume. CPU reference tests and GPU skips are not CUDA evidence. Preserve negative results; no default adoption before review. Paid GPU access remains separately authorized.

### G9 — Independent inference performance track

Acceptance follows I1–I7 in the inference track. First validate strict saved-request/roofline tooling and synthetic opt-in. Actual SLO benchmark, engine support, quality, profiler diagnosis and code-change A/B require native GPU evidence. No invented token ITL from streaming chunks; missing timings are unavailable. Roofline headroom is not achieved speedup. Keep workload/resource/quality contracts explicit; no commit, paid run or publication without authorization.

## 7. Resource and publication gates

- Never start a B300 rental or production-scale pretraining without confirmed access/budget and a pinned compatible environment.
- Before rental training: validate provider construction and actual parameter census, finite forward/backward, A/B pilot, checkpoint/resume, dataset format, hybrid inference/export and small benchmark path.
- Freeze scheduler horizon, checkpoint frequency, WSM windows and evaluation cadence after measured throughput/storage/cost. Do not copy Nemotron's absolute token windows into the smaller tutorial blindly.
- Retain checkpoints needed by incomplete merge windows and recovery. Do not prune inputs before their merges and provenance are validated.
- Separate accepted unique, processed, consumed and loss-bearing tokens. RL rollout/teacher/update accounting stays separate in later stages.
- Part I targets data/pre/mid/LC, 64K target and 256K stretch; Part II SFT/RLVR; Part III MOPD with same-node vLLM teacher. Current task does not authorize implementing all later stages.
- Blog/release packages remain **[1–4], [5–6], [7]**. Publish only after evidence gates and rights review. No local artifact implies public upload or publication.

## 8. Task ledger — maintain this section

Statuses: `TODO`, `IN_PROGRESS`, `BLOCKED`, `VERIFIED_CPU`, `VERIFIED_GPU`, `VERIFIED_BENCHMARK`. Mark the exact scope, not the entire project, verified.

| ID | Goal | Handoff status | Evidence / next action |
|---|---|---|---|
| B0 | Existing CPU foundation | Historical `VERIFIED_CPU` | Parent extension baseline rerun is logged below; each incoming agent still reruns its current workspace baseline |
| G1 | WSD scheduler/control/resume | `VERIFIED_CPU` reviewed scope | New inventory, explicit legacy opt-in and exact CPU resume passed independent re-review and parent rerun |
| G2 | PR #5114 WSM integration | `VERIFIED_CPU` synthetic DCP scope | Tracker guard/optimized verifier reviewed; isolated 25 tests and official -O fixture passed; real hybrid merge untested |
| G3 | Gym benchmark integration | `VERIFIED_CPU` importer fixture scope; official run `BLOCKED` | Parent suite passed; Gym runtime/target endpoint absent, official scores unmeasured |
| G4 | Integrated review/docs | `VERIFIED_CPU` commit-ready scoped artifact | All three re-reviews passed; final parent suite 222 passed, 1 skipped, 25 subtests; no staging/commit/push |
| GPU1 | Real NemoChat pretraining/merge/Gym comparison | `BLOCKED` pending environment/resources | GPU stack/export/evaluation gates; no benchmark gain measured |
| G5 | Large-corpus sampling and budget planning | `VERIFIED_CPU` selected-population scope | Actual pinned Super tokenizer CLI and planner regression tests passed; source-wide survey pending |
| G6 | HF checkpoint persistence/recovery | `VERIFIED_CPU` local/mock scope | Local bundle inventory/restore and mock transport gates passed; remote round trip pending destination/authorization |
| G7 | Stage-wide W&B monitoring | `VERIFIED_CPU` offline scope | Actual offline SDK tests and exact CPU trajectory passed; online/distributed stages pending |
| G8 | Nsight Systems / Triton pretraining optimization | `PLANNED_GPU` | Protocol added; no kernel implemented; driver unavailable and nsys absent |
| G9 | Independent inference performance | `VERIFIED_CPU` fixture/import scope | Saved-request and single-operator roofline CLIs exercised; independent review passed; parent 520-test suite passed; native serving/quality/GPU gates pending |
| G10 | Agent-assisted training systems lab | `VERIFIED_CPU` strict offline/source control scope | Whole-source review and shared-validator regressions passed; actual dense YAML A/B executed as experimental-only; native hybrid/GPU/distributed/quality gates remain unverified; see docs/en/WHOLE_REPOSITORY_REVIEW.md |
| G11 | NeMo Lens execution/UI/overhead tutorial | `VERIFIED_CPU` SDK/transport + real Jaeger/browser reviewed scope | `tutorials/nemo_lens/`; actual SDK/Jaeger/UI observations and CPU overhead; 15 tests + independent re-review passed; GPU overhead unverified |
| G12 | Modeler-facing MoE Nsight Bridge | `VERIFIED_CPU` source-tree control reviewed scope; native capture `BLOCKED` | `plugins/nemo_lens_nsight_bridge/`; 40 tests + independent re-review passed; explicit meaning/source and Lens identities verified; native plugin/NVTX/kernel mapping/GUI unverified |

When handing off, update this table with actual status and link the specific log entry/artifact. A worker timeout means unresolved status, not automatic failure or success.

## 9. Append-only work log

### Logging rules

Append an entry after each meaningful implementation or experiment, including failures and decisions. Do not erase earlier entries; append corrections with a reference to the superseded entry. Avoid secrets, API tokens, private signed URLs or full dataset dumps. Record real timestamps from the system clock.

Use this template:

```markdown
### YYYY-MM-DDTHH:MM:SSZ — <agent> — <task ID>: <short outcome>
- Goal:
- Starting state: git revision/dirty state; concurrent writers; environment.
- Changes: explicit paths and behavior, not just file counts.
- Commands executed: exact runnable commands, workdir and relevant safe settings.
- Results: exit codes; tests; metrics with denominator/units; errors/retries.
- Artifacts: paths, manifest/provenance and hashes where actually verified.
- Evidence level: planned/source-verified/fixture-tested/CPU-model-executed/GPU-smoke-executed/real-benchmark-executed.
- Review: reviewer scope, findings, fixes and re-review status.
- Limitations/blockers: precise missing prerequisites; no speculative success.
- Decisions: rationale and superseded assumptions.
- Next action: concrete task and acceptance gate; owner if concurrent.
```

### Historical baseline — timestamp not reconstructed

- Goal: CPU-first foundations for Part I, readable code and genuine execution.
- Result recorded in `docs/IMPLEMENTATION_VERIFICATION.md`: 118 passed, 23 subtests passed; independent scoped re-review passed; parent rerun passed.
- Scope: bounded data acquisition, actual tokenizer checks, tiny-model evaluation/update/save/resume, NIAH network/scoring/figures and defensive failure handling.
- Limitations: no B300 training, actual hybrid export/inference, long-context training or SFT/RLVR/MOPD; no commit/push/publication claimed.
- Evidence: historical report, not re-executed while writing this handoff.

### 2026-10-07T05:08:09Z — Hermes — HANDOFF: Codex goals and log document

- Goal: create `NemoChatAgent.md` from the existing implementation, project documents and user's WSD/WSM/Gym decisions.
- Inspected: README, project plan, verification report, execution/blog plan, WSD/WSM experiment spec, script inventory and live Git status.
- Git status: `.gitignore` modified; README, notices, configs, docs, pyproject, scripts, src and tests untracked. Preserve all existing work.
- Decision: user's PR #5114 stable-phase WSM supersedes the earlier decay-branch suggestion. Official Gym benchmarks are the requested benchmark source; CE/BPB remain diagnostics.
- Observation: merge/verify/Gym scripts already appeared during parallel work. Their verification reports were not available to this handoff author. This file does not mark them complete.
- Changes: this agent brief, goals, acceptance gates, task ledger and append-only log.
- Evidence level: document/source inspection only. No training or benchmark score was produced by this document-writing step.
- Next action: incoming Codex establishes writer quiescence, inventories code and runs the fresh baseline before editing.

### 2026-10-07T05:14:37Z — Hermes — G1/G2/G3: post-worker verification update

- Parallel batch returned: WSD and Gym workers completed; WSM worker timed out without a final summary. Timeout was not treated as proof that no artifacts existed.
- Parent executed `.venv/bin/python -m pytest -q` in the project: **182 passed, 1 skipped, 23 subtests passed**. `git diff --check` passed. The skip is the opt-in upstream merge integration in the default environment.
- Parent inspected `docs/en/WSM_IMPLEMENTATION.md` and `/home/ubuntu/runs/wsm-fixture-5/verification.json`, then independently executed `NEMOCHAT_WSM_UPSTREAM=/home/ubuntu/runs/wsm-upstream/Megatron-LM /home/ubuntu/runs/wsm-upstream/venv312/bin/python -m pytest tests/test_wsm.py -q`: **10 passed**, including actual official CPU DCP merge integration. This resolves execution uncertainty despite the worker timeout; it is not a full code review or real-model benchmark.
- Upstream runtime requires isolated Python 3.12 at this pin (`typing.override` blocked 3.11); its import graph required Triton even though merging itself was CPU-only. Preserve the existing tested isolated environment; do not modify the main project dependency environment blindly.
- WSD worker reported actual tiny-model 8-update W/S/D=2/4/2 training and exact continuous/split/periodic resume agreement. Parent verified its tests in the full suite, not all persistent model artifacts yet.
- Gym scope is an official-result importer/comparator with 21 worker-reported tests, included in the parent suite. It is **not** an executed official benchmark runner. Official benchmark scores remain unmeasured; no NemoChat endpoint/export was validated.
- New reference documents: `docs/en/WSD_IMPLEMENTATION.md`, `docs/ko/WSD_IMPLEMENTATION.md`, `docs/en/WSM_IMPLEMENTATION.md`, `docs/en/GYM_PRETRAIN_EVALUATION.md`, `docs/ko/GYM_PRETRAIN_EVALUATION.md`. Check the Korean WSM companion: it was not present at the parent's preceding inventory and may still need creation.
- Remaining: independent code review; WSD model-artifact verification; documentation reconciliation; GPU Bridge/export and actual Gym raw/merged benchmark execution. No performance improvement claim, commit or push.
- Next Codex action: start from these verified mechanics rather than rebuilding them; inspect partial WSM worker deliverables and complete the remaining evidence gates.

### 2026-10-07T06:38:16Z — Hermes — G4: pre-commit repairs and parent rerun

- Initial independent reviews identified four blockers: parent tracker overwrite through `iter_*` WSM output, optimized verifier assertions, missing CPU checkpoint-file identity inventory, authenticated loopback proxy credential escape.
- Separate fix workers implemented regression-tested repairs; also hardened response-byte limits and accepted-socket inactivity timeout. SHA inventory is accidental-change detection, not authentication of a rewritten manifest.
- Parent rerun: **222 passed, 1 skipped, 25 subtests passed** in project environment; isolated upstream WSM **25 passed**. Complete official fixture under Python `-O` printed **PASS**, report `/tmp/nemochat-parent-opt-bnob2vzo/fixture/verification.json`.
- Parent new-format WSD resume comparison `/tmp/nemochat-cpu-identity-hX2aU1zi/{resumed,continuous}` printed exact-equality **PASS**. Compileall/pip check/diff checks passed.
- Canonical plans corrected to selected WSD+stable WSM+official Gym comparison; missing Korean companions added, README current scope clarified, wheel includes README/notices, ignore rules protect raw outputs/credentials. Authored fixtures remain included.
- Independent repaired-snapshot review dispatched and **still pending**. No commit-readiness approval yet. See `docs/PRECOMMIT_REVIEW.md` and `docs/IMPLEMENTATION_VERIFICATION.md` for scope/history.
- No staging, commit, push, GPU rental or actual NemoChat benchmark execution. Next: evaluate independent re-review verdicts, fix any remaining blockers and rerun final checks before approval.

### 2026-10-07T06:40:45Z — Hermes — G4: final pre-commit approval

- All three independent re-reviews passed without blocking security, logic or documentation findings.
- Parent reran the full suite after receiving verdicts: **222 passed, 1 skipped, 25 subtests passed**; tracked diff check passed. External official WSM integration separately passed **25 tests**, and complete official optimized verifier passed.
- README/doc/config/packaging review passed. Minor random-depth config-precedence wording clarified in both language companions; final wheel should match frozen README.
- Local reviewed CPU foundation, WSD control, WSM tooling and Gym importer are ready for a scoped commit. No GPU/target-model benchmark approval, staging, commit or push.
- Remaining nonblocking limitations include malformed checkpoint-root JSON diagnostics, trusted immutable single-writer assumptions and nontransactional save.
- Next: user authorizes commit; then stage intentional code/docs/config/test/fixture files, inspect staged diff and keep ignored runtime artifacts excluded. Actual GPU/export/Gym benchmark work remains pending.

### Operations extension recovery — parent evidence, review pending

- Batch `deleg_19d13270` timed out after writing substantial implementations. Timeout is not a failure/success verdict for the files; parent inspected traces and reran the resulting quiescent suite: **323 passed, 1 skipped, 25 subtests passed**.
- Parent independently ran the actual cached pinned Super tokenizer corpus CLI: `/tmp/nemochat-ops-parent-OmooAiTU/corpus.json`. This is a two-document sample of a three-document handwritten fixture, not a corpus-wide survey.
- Parent ran the illustrative measured-input planner: `/tmp/nemochat-ops-parent-OmooAiTU/budget.json`; no B300 measurement inferred.
- Parent verified the synthetic local training bundle `/tmp/nemochat-hf-offline-ywpcsmw2/packed`; native model validity/resume and remote transfer remain separate gates.
- Parent actually executed W&B 0.30.0 offline tiny-HF training at `/tmp/nemochat-ops-parent-OmooAiTU/trained`, tracking under the sibling `tracking/` directory. Earlier offline split/continuous artifacts were independently compared and exactly matched model/optimizer/RNG/cursor/token counters.
- Added optional `hub`/`tracking` extras and ignored `/outputs/`, W&B runtime directories and DCP shards. Dependency check passed. No online calls/staging/commit/push performed.
- Independent review dispatched as `deleg_522e13bb`; all new extension readiness remains pending review. W&B companion docs must be completed; unimplemented GPU/later stages remain contract-only.

### Operations repairs — final local regression evidence

- Parent read the repaired tracking module and reran all tests: **358 passed, 1 skipped, 25 subtests passed**. Optional-dependency check, compileall, wheel build, ignored outputs, JSON and English/Korean local link checks passed.
- Budget total-limit/slack, storage initial footprint, integer ceiling and HF reserved namespace repairs have regression coverage. W&B schema/exception preservation/local receipt continuity repairs landed with actual offline SDK tests.
- G5–G7 local CPU/offline gates are passed; remote HF transfer/online W&B/native GPU resume remain pending. See docs/en/OPERATIONS_REVIEW.md. No staging/commit/push or paid remote calls occurred.

### 2026-10-07T14:54:33Z — Hermes — G8: performance lab scope accepted

- Added English/Korean performance protocol and linked README, project plan, WSD/WSM docs and this brief.
- Current host: `nvidia-smi -L` failed to communicate with driver; `command -v nsys` found no executable. No GPU profiling, Triton implementation or speedup measurement occurred.
- Root AGENTS.md update was blocked by its protected-file approval guard and was not retried. Read this brief explicitly for G8.
- Parent reran `.venv/bin/python -m pytest -q`: 358 passed, 1 skipped, 25 subtests passed (18.31s); `git diff --check` passed. Seven changed docs passed local-link, code-fence and whitespace checks. These are existing CPU/offline regressions and document validation, not Triton tests. CUDA/native-training gates remain pending.

### 2026-10-07T15:20:40Z — Hermes — G8: Nsight/Codex analysis guide

- Added English/Korean NSIGHT_CODEX_GUIDE with version-checked CLI examples, schema-aware read-only SQLite analysis, process/context-aware correlation, GUI confirmation and a ready-to-use Codex task. Linked README, performance lab and G8.
- Documentation-only change; no actual export, trace diagnosis, analyzer implementation or GPU speedup was executed.

### Inference track — final local verification

- Added independent I1–I7 design, English/Korean tool docs, saved-request analyzer and single-operator roofline calculator; linked README/project plan/AGENTS and G9.
- Parent full suite after writers completed: 520 passed, 1 skipped, 25 subtests passed (19.70s); pip check/compileall/diff checks passed. Independent reviewer reported no blocking security or logic issues.
- Parent synthetic CLI artifacts: `/tmp/nemochat-inference-parent-qR1ewG/`; no real engine/quality/GPU profile/speedup or public PR occurred.
- See `docs/en/INFERENCE_IMPLEMENTATION_VERIFICATION.md` for evidence and nonblocking provenance/error-diagnostic follow-ups. No commit/push.

### 2026-10-13T00:07:42Z — Hermes — G11/G12: executed scope and review blockers

- Goal: add separate NeMo Lens tutorial and modeler-facing Nsight Bridge folders while preserving existing training/operations defaults.
- Changes: `tutorials/nemo_lens/`, `plugins/nemo_lens_nsight_bridge/`, `scripts/verify_lens_ui.py`, browser CLI tests and English/Korean observability scope/verification docs. Optional dependencies remain isolated.
- Starting baseline: main suite 765 passed, 2 skipped, 25 subtests; post-browser-tests main suite 772 passed, 2 skipped, 25 subtests. Tutorial parent tests 9 passed; Bridge parent tests 32 passed. Main dependency/compile/diff checks passed.
- Parent actual CPU tutorial: `outputs/lens-parent-smoke-01/` has a real 17-span OTLP trace with exact hierarchy verified. Chromium rendered four steps with forward/backward/optimizer, screenshot `outputs/lens-ui-parent-03/trace.png`; repository browser verifier artifacts `outputs/lens-ui-parent-04/`.
- Parent overhead: `outputs/lens-parent-overhead-01/benchmark.json`, 4 repeats/mode, 200 updates, warmup 40; median baseline/disabled/file/OTLP training 87.397029/93.068929/185.613084/153.911699 ms. Exact loss/parameter parity passed. This is tiny CPU MLP instrumentation cost, not native GPU MoE overhead.
- Parent actual Bridge: `outputs/lens-bridge-parent-01/`, 21 operations and 21 Lens spans with matching bridge IDs; sparse-MoE forward/gradient reference parity and real source snippet verification passed. No NVTX emission or CUDA attribution claimed.
- Initial Bridge worker timed out after writing code/tests/docs; parent executed the artifacts but discovered missing WORK_LOG.md. Timeout was not treated as no artifacts or a completion verdict. Recovery worker owns that folder.
- Independent review: FAIL for tutorial OTLP exporter ambient netrc authentication and redirected POST escape; both reproduced with synthetic/mocked transport, not evidence of observed real credential leakage. Privacy fix worker owns tutorial. Bridge had no blocking review findings; launcher/NVTX failure tests were suggested and assigned to recovery worker.
- Current status: implementation and post-review fixes IN_PROGRESS, not release/commit approved. Native GPU/Nsight capture, plugin loading and GUI panel remain BLOCKED/unverified (driver unavailable, nsys absent). No staging/commit/push/publication/paid GPU run.
- Evidence: `docs/en/OBSERVABILITY_VERIFICATION.md` and Korean companion. Next: verify repaired artifacts/tests/actual export, independent re-review, final documentation and ledger update.

### 2026-10-13T00:29:33.471799+00:00 — Hermes — G11/G12: final scoped review passed

- Tutorial parent 15 tests and Bridge parent 40 tests passed; independent post-fix folder re-review passed and reviewed hashes matched current files.
- Final browser/CLI parent 38 tests passed; independent browser re-review also passed 38 tests. Full trace identity/service/all exact operation labels verified in actual Jaeger at `outputs/lens-ui-parent-hardened-02/`. A tight WebSocket regression exposed synchronous pre-handshake close deadlock; fixed without connecting to a destination server.
- Final main suite 783 passed, 22 skipped, 25 subtests: 20 opt-in browser cases executed separately plus two existing skips. Main dependency/diff checks passed. See English/Korean OBSERVABILITY_VERIFICATION documents and ignored review verdicts under outputs/.
- Local CPU tutorial/transport/Jaeger/browser helper and Bridge source-tree controls are reviewed. Nonblocking extra browser regression coverage remains. Native Nsight plugin discovery/loading, CUDA NVTX/kernel-source mapping, fused MoE/EP, native GUI panel and GPU overhead remain BLOCKED/unverified.
- No commit/push/publication, paid GPU access or cloud telemetry. Dedicated local Jaeger service remains loopback-only; its tutorial documents startup, SSH tunnel and project-only shutdown.

## 10. Ready-to-use instruction for Codex

> Read `NemoChatAgent.md` and its referenced documents before editing. Reconcile the current G1–G12 ledger with the latest user task. For G11/G12 read the two dedicated folder READMEs and `docs/en/OBSERVABILITY_VERIFICATION.md`; preserve the failed-review record, complete transport privacy repairs and re-review before approval. Pretraining performance work follows G8/P0–P3 without replacing WSD/WSM. First establish that previous workers are no longer writing, preserve the dirty/untracked workspace, inspect the actual implementation and run a fresh baseline. Then complete WSD resume correctness, official Megatron PR #5114 stable-phase WSM merge validation, and official NeMo Gym raw/merged evaluation contracts with readable English code and separate Korean explanations. Use real commands and artifacts, record failures and evidence levels, and keep this file's ledger and append-only log updated. Do not substitute custom averaging for the official PR, invent benchmark results, overwrite raw training checkpoints, start paid GPU/judge work, or commit/push without authorization. Continue through verification of the executable CPU scope; explicitly document any GPU/serving blockers and do not call the entire project complete.

**Discovery note:** `NemoChatAgent.md` is an explicitly supplied brief, not automatically discovered `AGENTS.md`. Tell Codex to read it in the task prompt. Adding an `AGENTS.md` forwarding file is a separate optional change, not performed by this handoff.

### Whole-repository final review — 2026-10-13T02:23:34Z

User explicitly authorized whole-source review, narrow repairs/comments and a local commit (no push). Cycle-2 NIAH/Bridge/rollout+API reviews PASS; 17 current hashes match. Final parent main 828 passed/22 skipped/25 subtests; Lens+rollout38/57subtests, Bridge52, Chromium39, officialWSM25 plus optimized verifier PASS. Actual cached WSD resume equality, dense YAML A/B, live NIAH HTTP scoring/plots, CPU MoE and final three25-span rollout+rendered UI verified in exact CPU scope. See docs/en/WHOLE_REPOSITORY_REVIEW.md and Korean companion. Preserve all historical timings; latest raw receipts remain ignored. Native GPU/distributed/target-quality/Nsight gates stay blocked/unverified. Final staged scan/wheel read-back precede the authorized feature commit; use Git for its actual ID.
