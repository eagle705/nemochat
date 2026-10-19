# Agent Guide — NemoChat

## Purpose and reading order

NemoChat is a noncommercial educational NVIDIA SDK tutorial for a small Nemotron-inspired hybrid Latent MoE. The seven-stage GPU system is a goal, not an implemented capability. Read `README.md` first for actual entry points and scope.

- Current goals, task ledger and append-only evidence: `NemoChatAgent.md`. Historical log entries are not fresh requests; check the latest ledger, current code and user's task.
- Architecture and stage plans: `docs/NEMOCHAT_PROJECT_PLAN.md`.
- Implementation evidence: `docs/IMPLEMENTATION_VERIFICATION.md`, `docs/PRECOMMIT_REVIEW.md`, and `docs/en/OPERATIONS_REVIEW.md`. Verify current state; do not reuse historical test counts as new execution evidence.
- WSD/WSM: `docs/en/PRETRAIN_WSD_WSM_EXPERIMENT.md` and related implementation docs.
- Pretraining performance lab (G8): `docs/en/PRETRAIN_PERFORMANCE_LAB.md` and its Korean companion under `docs/ko/`. Follow Nsight Systems diagnosis → optional Triton forward/backward fusion → correctness → unprofiled native full-update throughput/memory A/B and checkpoint/resume gates. Preserve WSD/WSM and reference defaults. Implementation and GPU measurements remain pending; CPU tests, skipped CUDA tests and isolated kernel gains are not evidence of whole-training speedup.
- Inference performance (G9): `docs/en/INFERENCE_PERFORMANCE_TRACK.md` and `docs/en/INFERENCE_PERFORMANCE_TOOLS.md`, with Korean companions. Saved-request and single-operator roofline analysis are CPU scaffolding; native serving, load generation, quality and profiler-backed engine improvements require separate real evidence. Never infer token ITL from streaming chunks or claim model headroom as achieved speedup.
- Operations: `docs/en/OPERATIONS_PIPELINE.md`, `CORPUS_AND_BUDGET.md`, `HF_CHECKPOINT_PIPELINE.md`, `WANDB_STAGES.md` (all under `docs/en/`).
- English technical docs have Korean companions under `docs/ko/`. Load relevant docs, not the entire document tree for every task.

## Repository map

- `pretraining/`, `midtraining/`, `sft/`, `rl/`: stage-owned entrypoints/configs/docs. Midtraining is KD/continued-pretraining only; SFT must never be nested inside it.
- `scripts/`: cross-stage runnable utilities; inspect `--help` before constructing commands.
- `src/nemochat/`: data preparation, diagnostics/evaluation, models, training controls, long-context probes, checkpoint tooling, planning, artifact transport and tracking.
- `tests/`: regression tests and clearly labeled synthetic/handwritten fixtures.
- `configs/`: versionable experiment/tool configurations; no credentials.
- `.env.example`: environment template; actual `.env` files are ignored. Python entry points do not automatically load them.

## Runtime directory convention

Resolve relative CLI/config paths from the repository-root CWD: `datasets/` for downloaded/prepared inputs, `checkpoints/` for downloaded/exported/resumable weights and state, `results/` for reports/traces/plots, and `logs/` for text/W&B logs. See [runtime layout](docs/en/RUNTIME_LAYOUT.md) ([한국어](docs/ko/RUNTIME_LAYOUT.md)). Pass supported CLI paths explicitly; do not invent experiment-schema fields or implicitly relocate HF caches. Tutorials launched from their own folder must use explicit repository-root paths. Keep versioned configs/fixtures in place. Preserve historical evidence and receipt paths; use the migration manifest for old → new local mappings, never overwrite existing artifacts.

## Project skills

Reusable agent-neutral workflows are in `skills/`. Before a matching task, read the relevant `SKILL.md` and referenced source/docs, loading only what is needed:
- `skills/nemochat-experiment-validation/SKILL.md`: YAML execution and performance claims.
- `skills/nemochat-checkpoint-resume/SKILL.md`: checkpoint integrity, transfer and exact resume.
- `skills/nemochat-observability-validation/SKILL.md`: Lens/Jaeger/source evidence and limits.

Use the current agent's equivalent file/terminal tools; unavailable capabilities must be reported, never fabricated. Skills do not grant authorization or override the user's current request. Root `skills/` is an explicitly read project library, not a claim of automatic skill discovery by every harness. See `skills/README.md` for sharing and Korean guidance.

## Working conventions

- Use explicit small functions and readable scripts. Avoid generic trainers, registries, factories and speculative abstraction.
- Code/comments and canonical docs are English; maintain Korean companion explanations for changed technical docs.
- Inspect Git status and actual files first. Existing untracked files can be intentional work: never delete, replace or ignore them simply because `git diff` omits them.
- Parallel writers must have disjoint ownership. Re-read a file changed by another writer before editing it.
- Preserve default behavior and resume compatibility; keep optional tracking/dependencies explicit.

## Scientific and operational invariants

- Selected pretraining policy: one WSD trajectory plus stable-window official Megatron PR #5114 WSM, initially `minus-sqrt`. No silent custom averaging or decay-branch replacement. Raw optimizer-bearing checkpoints remain the resume source.
- CPU controls, synthetic DCP merges, imported scores and local/mock transport tests are separate from real NemoChat GPU training, remote round trips and model-quality evidence. Report negative results and blockers honestly.
- Prepared corpus tokens, processed tokens and loss-bearing tokens differ. A selected shard/sample is not a whole-corpus survey. Budget projections require measured or explicitly illustrative stage-specific throughput; never invent B300 performance.
- HF model-only exports and native resumable training bundles differ. Inventory verification does not prove native optimizer resumption or `from_pretrained` compatibility. Persist immutable revision handles and verify read-back.
- W&B is disabled unless explicitly configured. Prefer offline validation; existing later-stage contracts are not executed training. CPU RNG preservation is not CUDA/distributed correctness.

## Safety and side effects

- No staging/commit/push, public publication, paid GPU/judge jobs or real remote upload/online run without user authorization for that operation and destination.
- HF private repositories are the default. Do not bypass rights review to publish artifacts.
- Never read/print actual `.env` credentials, include them in logs/config/artifacts, or commit secrets. Use environment variables. Key-based tracking sanitization is not universal secret detection.
- Do not source untrusted environment files: shell sourcing executes code. Only source a locally trusted file when explicitly needed.
- Never overwrite existing checkpoint/output paths or replace raw state with merged weights. Treat pickle-containing training artifacts as trusted inputs only.

## Verification and handoff

CPU setup on a fresh host (GPU SDK setup is separate):

```bash
python3 -m venv .venv
.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -e '.[cpu,test,hub,tracking]'
```

Run focused tests while iterating, then after writers stop:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m pip check
.venv/bin/python -m compileall -q src scripts tests
git diff --check
```

Inspect all changed/untracked source and docs; `git diff --check` alone does not cover untracked files. Verify relevant CLI examples and local links, and exercise optional upstream integration in its compatible runtime instead of calling a skipped test validated.

Before handing off, update `NemoChatAgent.md`'s ledger and append execution evidence, exact commands, scope, artifact paths and remaining gates. Do not erase historical records. A subagent report or timeout is not a verdict: inspect artifacts and independently verify important results.

## Canonical stage-owned source (2026-10-19)

`pretraining/`, `midtraining/` (KD/continued pretraining only), `sft/`, `rl/` own stage source/configs/entrypoints. `datasets/` and `checkpoints/` remain payload-only. See [stage-layout mapping](docs/en/STAGE_LAYOUT.md) and its original-blob manifest. Prior run paths in append-only journals are historical, not current source entrypoints. No operational RL trainer is claimed; native payloads and evidence remain in their original locations.
