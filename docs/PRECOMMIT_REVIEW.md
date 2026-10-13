# Pre-commit Review — 2026-10-07

Historical scoped evidence. For the latest whole-repository pass, repairs and commit gates, see [current review](en/WHOLE_REPOSITORY_REVIEW.md) ([Korean](ko/WHOLE_REPOSITORY_REVIEW.md)).

**Status: approved for a local CPU-foundation/WSM-tooling commit within the reviewed scope.** Initial blockers were repaired, all three independent re-reviews passed, and the parent reran the full suite after receiving verdicts. No staging, commit or push has been performed. This is not GPU/target-model benchmark approval.

## Scope

All commit-visible tracked/untracked code, tests, configs, README, English/Korean documents, agent handoff, notices and packaging. At first inventory the repository contained only a tracked `.gitignore`; therefore tracked `git diff` alone was not sufficient review coverage.

## Initial independent findings

| Priority | Issue | Required resolution |
|---|---|---|
| P1 | WSM output named `iter_<digits>` permits upstream writes to the parent tracker | Enforce new output-root naming; preserve a sentinel tracker regression |
| P2 | CPU resume contract lacks checkpoint-file integrity checks | Hash complete checkpoint file inventory; verify before load; explicit unverified legacy opt-in |
| P2 | WSM verification assertions disappear under Python `-O` | Runtime explicit errors and observed extra-state report; optimized execution regression |
| P2 | Authenticated loopback HTTP may route through environment proxy | Direct connection for authenticated HTTP loopback, no redirects; proxy regression with dummy credentials |

Nonblocking hardening selected for this repair: bounded HTTP response bytes and accepted-socket timeout for the loopback CPU smoke server. Nontransactional checkpoint saves remain a documented limitation unless explicitly changed. SHA inventories detect accidental changes, not a maliciously rewritten manifest or trusted-input deserialization risk.

## Documentation and packaging corrections

- README includes WSD/WSM and explicitly labels Gym as result importer/comparator.
- Canonical plans now select WSD + stable-phase PR #5114 WSM + suitable pinned official Gym comparisons, retaining CE/PPL/BPB/CORE as separate diagnostics.
- Historical plans acknowledge actual CPU inference/NIAH while preserving target-model GPU and distributed gates.
- Added missing Korean WSM implementation and WSD/WSM experiment companions.
- Expanded external Megatron/Gym source notices without claiming blanket project/data licensing.
- Packaging now embeds README metadata and third-party notices; local wheel build/read-back passed.
- Ignore rules exclude raw `runs/results/checkpoints/exports/figures`, weights/optimizer binaries and environment-specific credential files. Authored text fixtures remain commit-visible. Curated publication figures should be reviewed and kept in a separately intentional location such as documentation assets, not silently force-added from raw outputs.

## Pre-fix execution evidence

- Project suite: 182 passed, 1 skipped, 23 subtests passed.
- Official isolated WSM suite: 10 passed (the project suite's external integration skip separately exercised).
- WSD actual split/periodic resume artifact comparisons: exact weights, optimizer/RNG, cursor/token metadata.
- README WSD eight-update CPU command and tiny-HF base evaluation command actually ran.
- Eleven script help commands, compileall, pip check, local links/JSON/fences/whitespace checks passed.

These results are not approval of later repairs. Latest verified snapshot and independent re-review outcomes must be appended below and in `IMPLEMENTATION_VERIFICATION.md` before marking commit-ready.

## Known scope boundaries

No actual NemoChat B300 pretraining, hybrid target checkpoint export/serving, official Gym inference or raw/merged quality gain was established. The merge fixtures and tiny-HF control validate mechanics only. Existing historical worker/intermediate test counts are retained as history, not current baseline claims.

## Repaired snapshot — 2026-10-07T06:38:16Z

All three repair workers completed. Parent inspected the changed checkpoint code and independently reran:

- Project suite: **222 passed, 1 skipped, 25 subtests passed**.
- Isolated upstream WSM suite: **25 passed**, including actual official synthetic DCP merge.
- Official WSM verifier under Python `-O`: **PASS**, all eight fixture checks; observed-state report at `/tmp/nemochat-parent-opt-bnob2vzo/fixture/verification.json`.
- New inventory-format tiny-HF WSD split/continuous artifacts at `/tmp/nemochat-cpu-identity-hX2aU1zi`: exact model, optimizer/RNG, cursor/token-count comparison **PASS**.
- Compileall, `pip check`, and tracked diff whitespace checks passed.

Regression tests now cover tracker preservation, optimized verifier guards, checkpoint identity tampering/legacy opt-in, proxy credential routing, response cap, socket timeouts/disconnects and recovery. This is **CPU mechanics validation**, not actual target-model training or benchmark inference.

Independent repaired-snapshot re-review is still pending. Do not infer approval from these passing tests. Failed serialization may leave a partial checkpoint directory; the documented single-writer/trusted-input limitations remain.

## Final verdict — 2026-10-07T06:40:45Z

All three independent re-reviews passed with **no remaining blocking security, logic or documentation findings** in their scopes. Parent then reran the full suite: **222 passed, 1 skipped, 25 subtests passed**; `git diff --check` passed. The external WSM integration skip was separately exercised with **25 passed**, and the complete optimized verifier passed as recorded above.

Minor documentation clarification: supplied NIAH config values override CLI random-depth flags; both language companions now explain how to select random depths. Malformed CPU checkpoint JSON roots fail closed but may produce a less helpful AttributeError; diagnostic hardening remains a nonblocking suggestion.

The reviewed local CPU foundation, WSD controls, official WSM tooling and Gym importer are **ready to commit**. No claim of actual NemoChat GPU training, target hybrid export compatibility, official Gym inference or WSM quality gain is approved. Trusted immutable single-writer checkpoints, nontransactional save behavior and caller-controlled SDK transports remain documented limitations. No staging/commit/push performed.
