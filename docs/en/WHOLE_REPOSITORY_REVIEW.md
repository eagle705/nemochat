# Whole-repository review and commit gates

[Korean companion](../ko/WHOLE_REPOSITORY_REVIEW.md).

**Current status: independent review PASS for the reviewed CPU/source-tree scope. Local commit is authorized; staged-content and final packaging gates are required before execution.** The user authorized a local commit after whole-source review and needed repairs. Push/publication, cloud telemetry, uploads and paid GPU work remain unauthorized. This report will record the final scoped verdict before staging.

## Scope and baseline

The initial repository HEAD is `b5594a5debc6c090b16a33d7a9ebb4fb647a2d4e`, with only `.gitignore` tracked. Real implementation/config/docs/tests were pre-existing untracked user work. Review therefore includes commit-visible untracked contents, not only Git diff. Preserve the work; no stash/reset/clean is needed.

Initial inventory in this pass: 207 commit-visible candidates, including 103 Python files and 70 Markdown files. The active rollout extension may add more; the final staged inventory supersedes this count. Root modules cover data preparation, evaluation, model dimensions/provider declarations, WSD CPU update/resume, official WSM launcher, Gym artifact comparison, long-context probes, corpus/budget planning, checkpoint transport, tracking, strict experiment runner and performance analysis. Optional Lens tutorial/Bridge/browser and rollout controls are included in their compatible isolated runtimes.

## Parent-executed checks so far

- Main suite: **783 passed, 22 skipped, 25 subtests** (30.40 s). Twenty browser skips were previously exercised explicitly; two existing hardware/external-runtime gates remain distinct.
- Main `pip check` passed.
- Ruff E9/F63/F7/F82 passed. Full F scan found three unused imports; removed them and added a short comment explaining why optional Megatron imports follow upstream validation. Final E9/F scan passed after this cleanup.
- Focused post-cleanup planning/browser CLI regressions: **54 passed** (1.78 s).
- All **21 root script `--help` commands** exited 0.
- Wheel build succeeded; read-back found 34 Python modules, exact current README metadata and bundled third-party notices. Rebuild after final documentation freeze.
- Separate official WSM runtime: **25 tests passed**, including an actual CPU synthetic DCP merge. Full official fixture verifier under Python `-O` returned **PASS** for eight checks, stored at `outputs/precommit-whole/wsm-optimized-final/verification.json`.
- Actual offline tiny-HF WSD 4-update control, split 2+2 versus uninterrupted: exact model, optimizer/RNG and cursor/token-count equality **PASS**. Outputs at `outputs/precommit-whole/wsd-{split-2,resumed-4,continuous-4}/`. No native/distributed parity or model quality claim.
- Strict YAML dense baseline and activation-checkpoint candidate both executed successfully. Artifact comparison succeeded as `experimental-only`; small CPU timing is not GPU speedup evidence. Outputs at `outputs/precommit-whole/dense-{baseline,checkpointing}/` and `dense-comparison.json`.
- Commit candidate AST/credential-pattern/size/symlink scan found no initial security flags or large candidates. No actual credentials were read/displayed. Scan again on the frozen staged contents.
- Initial local Markdown link issues were only links to the in-progress rollout READMEs. Those must exist and validate before staging.

## Independent review ownership

- Core reviewer: data/eval/evaluation/long_context/models/training/checkpoints, their scripts/configs/tests.
- Operations/performance reviewer: experiments/training_perf/planning/tracking/artifacts/inference_perf, associated scripts/configs/tests and existing observability tooling.
- Rollout extension is still being finalized by its implementation worker and needs a separate frozen-source review. Prior observability approval is historical, not approval of this addition.

Verdicts, concrete repairs and reviewed-source hashes will be recorded after they return. Passing tests do not by themselves imply independent review approval. Read-only review recommendations must be distinguished from executed fixes.

## Commit acceptance

1. Finish all writers; identify native GPU/remote-work blockers explicitly.
2. Repair real blocking issues and add focused failure-path regressions. Prefer explanatory comments over broad speculative refactors.
3. Run the final main and isolated suites, actual relevant CPU/SDK controls, static scans, CLI/docs/JSON/YAML checks and wheel read-back.
4. Review all staged files and diff, including source fixtures and curated result provenance. Exclude credentials, raw source snapshots/traces, runtime logs, checkpoints, model/data binaries, caches, virtual environments and build products.
5. Use an appropriate feature commit message covering CPU training systems and observability labs; verify commit contents and post-commit Git state. No push.

## Evidence boundaries

This repository is an educational noncommercial systems lab. Native hybrid training, distributed resume/export, target-model WSM/Gym improvements, native inference/EP, actual Nsight plugin loading/kernel-source mapping/panels and GPU overhead remain separate unverified gates. A clean local commit is not validation of those capabilities. Raw outputs stay ignored; Markdown evidence records the exact executed scope.

## First independent verdicts — repairs required

Both broad source reviews returned FAIL with no confirmed security issue, but three reproduced logic blockers: NIAH evidence overwrite, NIAH interleaved fixture validation/request loss, and Bridge cleanup replacing an existing body exception. Dedicated fix contexts own these narrow repairs and failure-path tests. A separate rollout review is in progress. This is not commit approval.

Nonblocking improvements selected: removed three unused imports; explained CPU initialization reseeding before `.to(device)`, why device diagnostics must not force per-update CUDA synchronization, and why remote upload read-back failure is not remote rollback. No cloud upload was executed. Deferred suggestions include hosted-model immutable revision enforcement, stricter data-config key validation and storage validation on measurement-required plans.

Parent reran all three rollout scenarios: exact 25-span identity/count/hierarchy ingestion each. Pre-repair parent observations (single batch each): baseline 91.265 ms batch / generation tail; tool-slow 341.431 ms / tool.client tail; tool-straggler 852.563 ms / tool.client tail. Actual Chromium rendered the straggler trace with 25 spans and seven operation labels; screenshot at `outputs/precommit-whole/rollout-ui/trace.png`. Eight focused rollout tests passed. These are local scripted controls, not native RL or benchmark findings, and do not approve export/cleanup failure paths currently under review.

## Final reviewed snapshot — 2026-10-13T02:23:34Z

All three cycle-2 independent re-reviews passed with no remaining blocking security/logic findings in their scopes. Parent verified all 17 reviewed-file hashes against current contents. Original whole-source reviews cover unchanged remaining source families; repaired NIAH, Bridge, rollout and API/browser paths were separately re-reviewed.

- Final parent main suite: **828 passed, 22 skipped, 25 subtests** (28.88 s). Twenty skips are opt-in browser cases exercised separately; existing CUDA/external-WSM gates are distinct.
- Scoped Lens/rollout: **38 passed, 57 subtests**; plugin **52 passed**; explicit Chromium/CLI **39 passed**. Official separate-runtime WSM **25 passed**, optimized verifier PASS.
- Main pip check and scoped Ruff E9/F passed. Style-only baseline suggestions outside this fatal/undefined/unused gate are not blanket formatting approval.
- Actual WSD split/continuous state equality, two YAML dense controls/comparison, live tiny-HF HTTP NIAH generate/run/score/PNG/SVG pipeline, and actual Lens CPU MoE control were exercised. Tiny controls establish mechanics, not model quality, 64K support or native/distributed integration.
- Fresh final rollout scenarios at `outputs/precommit-whole/final-cycle2-{baseline,tool-slow,tool-straggler}/` each verified 25 spans with complete identity, resource, per-trajectory hierarchy and timestamp correspondence. Actual final Chromium straggler rendering verified 25 spans/seven operation labels with no JS errors; private screenshot `outputs/precommit-whole/final-cycle2-ui/trace.png`.
- The legitimate JSON double representation (`float64`, value 250) is compared by semantic tag type, not Python decoded-number type. Bool/int IDs remain strict. Timestamp policy is pinned-SDK quantization only: <=1 us start, <=2 us duration/end and <=1 us remote-parent computed-end allowance; no relative epsilon or invented timestamps.
- NIAH retains published partial artifacts and permanent reservation attempt tombstones; retries use new paths. Private staging has explicit trusted-writer ownership. All cleanup attempts preserve the primary error. Multi-artifact visibility is not transactional and interruption recovery is not implemented. Bridge transfers only its own stage/type cleanup diagnostics, never arbitrary backend notes.

Source/config/docs/fixtures remain commit candidates; raw outputs, credentials, model/data binaries, environments/caches and build products stay excluded. Post-document-freeze wheel read-back and staged scans are mandatory before the local feature commit. The actual commit ID is obtained from Git after execution, not embedded recursively in this snapshot. No push, cloud upload, online W&B, paid GPU run or native GPU/Nsight/RL/vLLM/Gym approval.

Nonblocking follow-ups retained: hosted mutable-revision policy, stricter data-config keys/storage validation on incomplete plans, finite injected-delay metadata checks and additional add_note-override permanent regression. Current normal injected delays are fixed finite constants; reviewed correctness gates passed.

## Runtime-path migration reference

The historical paths, timestamps and results above remain original evidence. For new runs follow the [repository-root runtime layout](RUNTIME_LAYOUT.md); locate relocated local artifacts through the runtime migration manifest's old → new mappings. Preserve receipt remote paths/revisions/hashes and inventory-relative payload paths; never overwrite existing artifacts.
