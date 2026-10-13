# Implementation Verification — CPU Foundation

Historical scoped evidence. For the latest whole-repository pass, repairs and commit gates, see [current review](en/WHOLE_REPOSITORY_REVIEW.md) ([Korean](ko/WHOLE_REPOSITORY_REVIEW.md)).

This report describes **actual local execution**, not B300 training or final model quality. The original repository contained documentation but no application implementation/test suite, so no earlier application-test baseline existed.

## Executed environment

Python 3.11 virtual environment under `.venv`, CPU-only PyTorch; key installed versions verified with pip:

| Package | Version |
|---|---|
| torch | 2.14.1+cpu |
| transformers | 4.57.6 |
| tokenizers | 0.22.2 |
| pytest | 9.1.1 |
| pyarrow | 25.0.1 |
| matplotlib | 3.11.2 |
| numpy | 2.4.6 |
| requests | 2.34.2 |
| PyYAML | 6.0.3 |

These are observed local versions, not a compatible pinned Bridge/TE GPU stack.

## Unit suite before independent review fixes

Command: `.venv/bin/python -m pytest -q`.

Observed: **65 passed, 16 subtests passed**. Unit fixture tokenizers and mocked endpoints are explicitly not model capability results. Compileall and `git diff --check` also passed.

## Real model inference

Command: `scripts/base_eval.py`, `sshleifer/tiny-gpt2` revision `5f91d94bd9cd7190a9f3216ff93cd1dd95f2c7be`, local hand-written `tests/fixtures/texts.jsonl`, context 128, COPA capped to 3 examples.

Artifact: `runs/cpu-base-smoke/metrics.json` (local, ignored by Git).

- Eligible NLL: 822.1864862442017 nats.
- Eligible target tokens: 76; represented bytes: 242.
- BPB: 4.9015056462414535.
- COPA correct: 2/3; centered score: 0.33333333333333326.
- Score scope: **partial task subset**, not full CORE.

These numbers belong to the tiny HF control and a hand-written corpus. They are not NemoChat training results, useful model-quality claims, or a nanochat leaderboard reproduction. BPB context protocol is explicitly document-local BOS-prefixed chunks.

The initially attempted `hf-internal-testing/tiny-random-gpt2` tokenizer failed raw-byte conservation on the Unicode fixture (81 text bytes vs 77 represented bytes). The error was retained as fail-closed behavior; the check was not bypassed. A complete-byte GPT-2 tokenizer/model was used instead.

## Actual Super tokenizer validation

Loaded only the tokenizer, not the 120B teacher weights, at revision `2dc98e2afe4face0e4ce40972a915c45368bd34a`. Verified raw-byte and text round trips for ASCII whitespace, Hangul/café/emoji, and code. Observed vocabulary 131072. Its BPE/ByteLevel backend supports the implemented exact accounting.

## Real optimizer/save/resume

Executed `scripts/train_cpu.py` on the tiny HF control for:

1. 2 updates -> `runs/cpu-train-2`.
2. Resume for 2 more updates -> `runs/cpu-train-resumed`.
3. Independent uninterrupted 4 updates -> `runs/cpu-train-uninterrupted`.

The maximum final safetensor difference across weights was **0.0**. After review, `scripts/verify_cpu_resume.py runs/cpu-train-resumed runs/cpu-train-uninterrupted` checked exact equality of the saved optimizer/RNG tensors in `optimizer_rng.pt` **and** the contract, step, cursor and consumed-token metadata in `run.json`, and printed PASS. Its seven positive/negative regression fixtures passed; comparison checks remain active under Python `-O`. Loss/token sequences matched:

| Step | Cumulative target tokens | LM CE |
|---|---:|---:|
| 1 | 22 | 10.815328598022461 |
| 2 | 45 | 10.814241409301758 |
| 3 | 76 | 10.826849937438965 |
| 4 | 98 | 10.81783390045166 |

This validates CPU update and resume mechanics only. Model has tiny dimensions; three authored fixture documents are intentionally reused. No generalization claim.

## Real bounded ClimbMix acquisition

Command: `scripts/prepare_data.py --config configs/data/climbmix_smoke.json --output runs/climbmix-remote-smoke`.

Pinned source shard `shard_00000.parquet` downloaded **92291918 bytes**, inside explicit 100000000-byte bound. Parent read the resulting manifest and inspected outputs after execution. Quota counters:

- Seen rows: 2.
- Accepted text tokens: 254; accepted EOS: 2.
- Flat token total: **256**, quota reached.
- Training-consumed tokens: **0**.
- Truncated records: 1, explicitly recorded.

Temporary shard/cache was removed. This run demonstrates selected-shard acquisition, not byte-perfect network minimization, full-corpus/global dedup, near dedup, GPU curation or MCore indexed format. Final source/input/output SHA256 hashes and rights are in the manifest.

Separate local fixture acquisition with the **actual Super tokenizer** produced 30 text + 2 EOS tokens, exactly 32, at `runs/local-super-smoke`. Config: `configs/data/local_super_smoke.json`.

## Actual NIAH network -> score -> figure

Started the loopback-only `scripts/serve_cpu.py`, verified `/health`, then ran `scripts/niah.py generate/run/score/plot` using the same pinned tiny-GPT-2 tokenizer/model.

- Context budgets: 128, 256 (prompt plus generation reservation).
- Needle depths: 0%, 50%, 100%; one sample/cell.
- Requests: **6**; usage-validated responses: **6**.
- Exact matches: **0/6**.
- Files: `runs/niah-cpu/{fixtures.jsonl,predictions.jsonl,cells.jsonl,cells.csv,heatmap.png,heatmap.svg,heatmap.metadata.json}`.

The actual PNG was visually inspected: model ID, context budgets, depth, fixed 0..1 scale, cell sample counts and Wilson intervals are visible. Retrieval failure is expected for this tiny control; no 64K/256K capability claim. The low sample count is pipeline smoke only.

The LC CLI emitted six planned steps and a cursor, exercising schedule generation. It is **not** an actual training controller or checkpoint of completed GPU updates.

## Independent review status

The first independent reviewer found no concrete security issue but identified a blocking bug: NaN model logits/NLL could become a valid-looking CORE correctness decision. The implemented fix now rejects nonfinite logits and token NLL before argmax/scoring. Parent read the actual changed code and ran real-torch regression tests covering NaN, both signs of Inf, finite logits that overflow NLL, and normal inference. Schema/few-shot/token-boundary golden fixtures were added. Hosted models now require an immutable 40-character revision SHA, with CLI regression tests.

After these changes, parent ran `.venv/bin/python -m pytest -q`: **86 passed, 16 subtests passed**. Compileall and `git diff --check` passed. The normal tiny-model evaluation was re-run at `runs/cpu-base-reviewed/metrics.json`; BPB and COPA results were unchanged. An initial new CLI test fixture omitted required `--output` and failed before the intended revision guard; the test command was corrected and the complete suite re-run. This was a test-fixture error, not a bypass of the revision guard.

The corrected CORE scope **passed independent re-review**: 24 targeted tests passed; no security or logic concerns remained in that scope. The data/LC review then identified two independently reproduced issues: authenticated remote HTTP could expose a bearer credential, and falsy non-null server error fields could be treated as valid completions. These were reproduced using dummy credentials/mocked responses, not real credential disclosure.

Fix workers implemented HTTPS enforcement for authenticated non-loopback urllib calls and rejected every non-null error field. They also addressed the review suggestions: data preparation stops before advancing the iterator when fewer than two quota slots remain, and the CPU server returns structured 400 responses for non-object JSON. Parent read the changed source and ran the complete suite: **118 passed, 23 subtests passed**; compileall and `git diff --check` passed.

Parent restarted the actual tiny-model server, checked health, reran six NIAH requests (all tokenization-validated; actual matches still 0/6), then exercised JSON list/int/string requests: all returned structured HTTP 400 and the server remained ready. Files are under `runs/niah-reviewed`. The actual Super tokenizer local pack was rerun at `runs/local-super-reviewed`, again exactly 30 text + 2 EOS tokens. The temporary CPU server was stopped after verification.

Final independent re-review of the **scoped data/LC/server fixes and checkpoint verifier passed**, with no remaining security concerns, logic errors or suggestions in that scope. The reviewer ran 85 targeted tests plus 23 subtests and the full suite (118 tests plus 23 subtests). Parent independently reran the full suite after receiving the verdict: **118 passed, 23 subtests passed**, and `git diff --check` passed. This approval is limited to the reviewed CPU code, not unexecuted GPU integrations. Hosted client transport remains the caller's responsibility.

The data and long-context implementation workers initially timed out after writing their files. Their completion status was not used as proof of correctness; parent execution and the subsequent independent reviews are the evidence. One worker's intermediate full-suite run observed three data tests failing while the separate quota-fix worker was still modifying its owned files. Those transient results do not describe the final workspace; both the final reviewer and parent verified the complete passing suite.

## Pre-commit extension audit — 2026-10-07 (blockers under repair)

This section supplements, rather than replaces, the historical 118-test CPU foundation results above. Independent initial review found four blocking issues: WSM `iter_*` output could overwrite a parent tracker; WSM verifier assertions disappeared under optimized Python; CPU resume lacked checkpoint-file integrity inventory; authenticated loopback HTTP could leak credentials through an environment proxy. Fixes and independent re-review are **pending**. The execution results below describe the **pre-fix** workspace, not approval of the repaired snapshot.

Parent executed from the current workspace:

- `.venv/bin/python -m pytest -q`: **182 passed, 1 skipped, 23 subtests passed**. The skip is the external-upstream WSM integration.
- `NEMOCHAT_WSM_UPSTREAM=/home/ubuntu/runs/wsm-upstream/Megatron-LM /home/ubuntu/runs/wsm-upstream/venv312/bin/python -m pytest tests/test_wsm.py -q`: **10 passed**, including actual official CPU synthetic DCP merge and independent output checks. This is an isolated Python 3.12 runtime, not the tutorial's Python 3.11 environment.
- `scripts/verify_cpu_resume.py` against `/tmp/nemochat-wsd-20261007/{resumed,periodic-resumed}` versus `continuous`: both printed **PASS** for exact model, optimizer/RNG and cursor/token counters.
- README WSD eight-update command, with fresh temporary output paths: exit 0, 197 loss-bearing tokens, four periodic checkpoints. Artifact: `/tmp/nemochat-precommit-1mvpfbyi`. This is actual tiny-HF CPU execution, not NemoChat training.
- All 11 `scripts/*.py --help` commands exited 0; compileall passed; `pip check` reported no broken requirements; a local wheel build completed.
- AST-based scan of reviewable Python files found no `eval`/`exec`, `os.system`, `shell=True`, direct pickle-load calls or assignment-shaped secret candidates. This narrow automated scan is not proof of absence of vulnerabilities.
- Markdown local-link and JSON parse checks passed at initial inventory. No binary or >1MB file appeared in commit candidates. New/untracked files are included in the audit; tracked-only `git diff --check` passed but does not cover them.

Documentation changes include README WSD/WSM/Gym scope and commands, a Korean WSM companion, and WSM/Gym upstream notices. Gym remains a **result importer/comparator**, not an executed official benchmark runner. No actual NemoChat benchmark score, GPU result or WSM quality gain has been measured. Absolute `/tmp` and external-runtime paths describe this audit host, not portable included artifacts.

No commit, staging, push, paid API call or GPU rental was performed. Review findings, fixes and final rerun must be recorded before claiming commit readiness.

## Repaired extension snapshot — 2026-10-07T06:38:16Z

The four initial review blockers have implementations and regression tests in place. Parent reran the quiescent project suite: **222 passed, 1 skipped, 25 subtests passed**; the isolated official WSM suite passed **25 tests**. Parent also ran the complete official fixture verifier under Python `-O`: **PASS**, with all eight checks and an observed extra-state report at `/tmp/nemochat-parent-opt-bnob2vzo/fixture/verification.json`. Parent compared the new SHA-inventory WSD checkpoints `/tmp/nemochat-cpu-identity-hX2aU1zi/{resumed,continuous}`: exact model/optimizer/RNG/cursor/token equality **PASS**. Compileall, pip check and tracked diff check passed.

New checkpoints require file inventories before resume loading; unverified historical checkpoints require explicit legacy opt-in. WSM wrapper disallows `iter_<digits>` output roots, verifier uses explicit runtime guards, and authenticated loopback HTTP bypasses environment proxies. NIAH HTTP response reads are bounded and the loopback smoke server has socket inactivity timeout handling. Detailed outcomes and remaining single-writer/nontransactional/trust boundaries are in [pre-commit review](PRECOMMIT_REVIEW.md).

Independent repaired-snapshot re-review subsequently **passed in all three scopes** (WSM/NIAH/server; CPU checkpoint integrity; documentation). After receiving the verdicts parent reran the full suite at 2026-10-07T06:40:45Z: **222 passed, 1 skipped, 25 subtests passed**; tracked diff check passed. No blocking findings remain in the reviewed CPU/tooling scope. Minor NIAH config-precedence wording was clarified. These runs do not establish actual NemoChat GPU or official Gym benchmark execution. No staging, commit or push was performed. See the final verdict in [pre-commit review](PRECOMMIT_REVIEW.md).

## GPU and publication boundaries

Not executed: pinned Bridge/TE provider construction, actual 13.8B parameter census, distributed forward/backward, MCore dataset conversion, TP/CP/EP BPB parity, hybrid export/inference parity, Curator GPU operations, long-context updates, SFT/RLVR/MOPD, Dynamo serving.

No Git commit/push, public model/dataset release or external blog publication was performed. Existing project-plan milestones remain plans except for the CPU implementation scope above.
