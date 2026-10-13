# Nsight Bridge execution work log

Scope: this plugin source tree only. No root project edits, staging, or commit.
[English README](README.md) · [한국어 README](README.ko.md).
This log was created during timeout reconciliation because the README-linked
`WORK_LOG.md` was absent. Earlier parent/reviewer counts are historical context,
not new evidence below. Native Nsight capture and panel acceptance remain blocked.

## Reconciliation and independent review hardening

Read current bridge, launch CLI, source-tree configuration, existing tests and
English/Korean documentation before modifying them. Reviewed the independent
review's concrete Bridge suggestions. The tutorial exporter concerns in that
review are outside this task's ownership; no tutorial files were edited here.

### Changes

- Added `tests/test_review_hardening.py`: exact mocked launch argv/cwd, complete
  inherited environment (including plugin search dirs), unchanged parent env,
  return-code propagation, and refusal of existing output/prefix/report/SQLite
  sidecar paths. The fake subprocess creates no capture artifacts.
- Added fake runtime NVTX push/pop failures, including simultaneous wrapped-call
  and pop failures. Found push failure skipped record creation, pop failure
  skipped records and replaced the wrapped function exception.
- Changed only `Bridge.call` cleanup: attempt the operation diagnostic record
  after push/pop failure; pop only after successful push; retain primary callable
  exception. Successful-call pop failures still propagate and record error status.
  `nvtx_error` contains stage/type only, no potentially sensitive exception text.
  Record-write failure does not replace an already propagating exception.
- Preserved original source snapshot/integrity tests unchanged; no weakened
  source checks, magic code, global monkeypatch, or framework refactor.
- English/Korean READMEs explicitly state source-tree-only usage; standalone wheel
  manifest/resource packaging is pending. No new packaging feature implemented.

### Actual executed commands and results

All commands below ran from
`/home/ubuntu/nemochat/plugins/nemo_lens_nsight_bridge` in its existing `.venv`.

1. Before fixing cleanup:
   `.venv/bin/python -m pytest -q tests/test_review_hardening.py`
   → **3 failed, 5 passed**. The three NVTX regression cases reproduced missing
   records and masked primary failure; mocked launch cases already passed.
2. After fixing cleanup: `.venv/bin/python -m pytest -q`
   → **40 passed in 7.85s**; repeated fresh focused suite →
   **40 passed in 7.52s**, zero skips. Persisted repeat:
   [`artifacts/reconciliation-01/focused-tests.log`](artifacts/reconciliation-01/focused-tests.log).
3. `.venv/bin/python -m lens_nsight_bridge doctor`
   → exit 0; `nsys_path: null`, `cuda_available: false`, Torch `2.14.1+cpu`,
   Lens `0.2.0+4d045c0`, OTel SDK `1.45.1`. Persisted
   [`doctor.log`](artifacts/reconciliation-01/doctor.log).
4. `.venv/bin/python -m lens_nsight_bridge run --lens --device cpu --output /home/ubuntu/nemochat/plugins/nemo_lens_nsight_bridge/artifacts/reconciliation-01/lens-cpu`
   → exit 0, forward/reference and gradient/reference **passed**, finite gradients
   true, 8 tokens / 4 experts / top-2 / 16 assignments, loss
   `0.0007786298519931734`; NVTX false, Lens true. Persisted
   [`lens-cpu-run.log`](artifacts/reconciliation-01/lens-cpu-run.log) and actual
   [`result.json`](artifacts/reconciliation-01/lens-cpu/result.json).
5. `.venv/bin/python artifacts/reconciliation-01/verify_capture.py`
   → exit 0; independently read **21 actual Lens spans + 21 operation records**,
   asserted all bridge/source IDs and file hashes match, all records status ok,
   NVTX false, and no NVTX errors. This read-only verification script invokes the
   actual snippet CLI with captured bridge ID
   `10efc7dd312943d2a0b0131d2bd43d87:6efe9d5e5bbf468f` and `--current-root`.
   Snippet exit 0, real `control.py:30–32`, `F.silu(gate) * value`, SHA256
   `aada59cdeece1938dd1699a3d3e870d2bdc3205879af33fc3b291191989174f8`.
   Command/verified snippet persisted in
   [`snippet-verification.log`](artifacts/reconciliation-01/snippet-verification.log).

Evidence inventory: actual capture
[`operations.jsonl`](artifacts/reconciliation-01/lens-cpu/operations.jsonl),
[`lens-spans.jsonl`](artifacts/reconciliation-01/lens-cpu/lens-spans.jsonl),
[source snapshot](artifacts/reconciliation-01/lens-cpu/sources/aada59cdeece1938dd1699a3d3e870d2bdc3205879af33fc3b291191989174f8.py),
[repeatable read-only verifier](artifacts/reconciliation-01/verify_capture.py).

An initial tool-wrapper parsing attempt after the successful CPU run failed
while reading JSONL (line-prefix assumption, then wrapper key mismatch). No CPU
run output was fabricated or rerun over an existing path. The persisted read-only
verifier above resolved that harness issue and exercised the real snippet CLI.

### Final artifact checks

A read-only local Markdown link check over both READMEs, this log and both role
cards checked **23 local links, zero missing**. `compileall -q` over bridge source,
tests and the read-only verifier exited 0. These are artifact checks, not GPU
acceptance. The repository largely contains pre-existing untracked work, so a
tracked-only `git diff --check` cannot validate all these new files.

### Remaining acceptance gates

No installed `nsys` or working GPU: native capture, actual NVTX emission,
manifest discovery/target-environment propagation, CUDA launch/backtrace/kernel
attribution and native explanation/code panel remain **blocked/unverified**.
Mock success checks command construction only. CPU spans and source snapshots
are not GPU performance or kernel mapping evidence. Distributed/native adapters
and standalone wheel manifest deployment remain pending.
