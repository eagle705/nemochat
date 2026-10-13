# Parent verification: Lens tutorial and Nsight Bridge

[Korean summary](../ko/OBSERVABILITY_VERIFICATION.md). This is real local execution evidence, not GPU or production approval. No commit/push/publication occurred.

## Environment and scope

- Repo HEAD at inspection: `b5594a5debc6c090b16a33d7a9ebb4fb647a2d4e`; existing dirty/untracked files preserved.
- Python 3.11.15, project torch 2.14.1+cpu. Docker/Compose works. `nvidia-smi` cannot communicate with the driver; `nsys` absent.
- Main dependency environment remains unchanged; optional Lens runtimes and browser runtime are isolated.
- Baseline parent suite before new tests: **765 passed, 2 skipped, 25 subtests passed** (30.89 s).

## Actual tutorial execution

From the repository root:

```bash
tutorials/nemo_lens/.venv/bin/python tutorials/nemo_lens/train.py \
  --mode otlp --steps 4 --warmup 2 --output outputs/lens-parent-smoke-01
tutorials/nemo_lens/.venv/bin/python tutorials/nemo_lens/verify_jaeger.py \
  --result outputs/lens-parent-smoke-01/result.json
```

Actual trace ID: `9537b50eb6b2842441aa93901e7f4c84`. Jaeger returned exactly **17 spans**: one root, four steps, four each forward/backward/optimizer. Service/run identity and parent-child hierarchy passed. Raw response and receipt reside alongside result.json.

### Real browser, not only API

Chromium launched via a separate Playwright runtime at `/home/ubuntu/runs/nemo-lens-browser/.venv`. Parent ran:

```bash
/home/ubuntu/runs/nemo-lens-browser/.venv/bin/python scripts/verify_lens_ui.py \
  --trace-id 9537b50eb6b2842441aa93901e7f4c84 --output outputs/lens-ui-parent-04
```

The actual Jaeger search page and trace page rendered; all five operation names were visible, no page JavaScript errors. Screenshot of the same trace was visually inspected at `outputs/lens-ui-parent-03/trace.png`, showing four steps and their nested phases. The repository script's initial runtime screenshots/receipt are in `outputs/lens-ui-parent-04/`. Later independent review identified that the initial verifier itself did not enforce browser-origin boundaries and accepted partial operation text; these genuine screenshots remain observation evidence, not approval of the verifier implementation. A dedicated repair must constrain browser requests and verify trace-specific identity/labels before final approval.

An initial external browser probe used an incorrect exact `Service` locator; the UI label is `Service (2)`. It timed out, was diagnosed by reading the real DOM, and corrected. No fabricated screenshot or API-only render claim was substituted.

## Independent repeated CPU overhead run

```bash
tutorials/nemo_lens/.venv/bin/python tutorials/nemo_lens/benchmark.py \
  --steps 200 --warmup 40 --repetitions 4 --output outputs/lens-parent-overhead-01
```

Four modes × four rotated-order repetitions, fresh subprocess each, exact losses and final parameter hashes equal. Every OTLP trace and file export passed complete-count validation. These timed updates are a small CPU MLP on generated data, not native NemoChat or MoE GPU training.

| Mode | Median training ms / 200 updates | Overhead relative to baseline |
|---|---:|---:|
| Baseline, no Lens wrapper | 87.397029 | Reference |
| Disabled wrapper | 93.068929 | +6.489809% |
| Enabled file export | 185.613084 | +112.379169% |
| Enabled local OTLP | 153.911699 | +76.106329% |

Setup, shutdown and full-process durations are separately recorded in `benchmark.json`. Concurrent host activity and tiny operations limit generalization; no statistical significance or zero-overhead claim. The worker's separate eight-repeat/400-update result is a different run, not to be mixed with this table. GPU/rank/span-granularity/backpressure/failure measurements remain pending.

## Actual Bridge execution

```bash
plugins/nemo_lens_nsight_bridge/.venv/bin/python \
  plugins/nemo_lens_nsight_bridge/launch.py doctor
plugins/nemo_lens_nsight_bridge/.venv/bin/python \
  plugins/nemo_lens_nsight_bridge/launch.py run --lens --output outputs/lens-bridge-parent-01
```

Actual CPU sparse-MoE control: 8 tokens, 4 experts, top-2, finite gradients; independent token-major forward and gradient comparisons passed. **21 operation records and 21 real Lens spans** have exactly matching bridge IDs. Nine roles were observed: routing, topk, permutation, dispatch, expert_up, activation, expert_down, combine, backward.

The parent used the actual activation bridge ID to execute `snippet --current-root plugins/nemo_lens_nsight_bridge`. It returned verified captured code from `lens_nsight_bridge/control.py:30-32` (SwiGLU implementation), after source file/snippet identity checks. `outputs/lens-bridge-parent-01/parent_verification.json` retains the output.

The parent also requested `run --nvtx` and `launch`: both rejected with exit 2 before creating success output directories, respectively because CUDA/NVTX and `nsys` are unavailable. No actual NVTX event, plugin discovery/load, CUDA event correlation, native fused EP backend or Nsight GUI panel was verified.

## Test and quality gates

Parent reruns:

- Tutorial isolated tests: **9 passed** (7.73 s).
- Bridge isolated tests: **32 passed** (7.21 s).
- Added optional-browser CLI tests: **7 passed** (0.39 s); unsafe URLs rejected without creating output.
- Final main suite at this checkpoint: **772 passed, 2 skipped, 25 subtests passed** (28.47 s).
- Main `pip check`, Python compileall and `git diff --check` passed. Untracked files require explicit source/document checks in addition to Git diff.

Independent review arrived with **FAIL** for two tutorial exporter privacy defects: ambient `.netrc` authentication through Requests `trust_env=True`, and redirected OTLP POST bodies escaping the initial loopback URL. The verifier API's redirect refusal did not protect the exporter transport. Separate fix workers were dispatched; this report does not mark the corrected snapshot approved until post-fix tests and re-review complete. Bridge review found no blocking issues in the inspected snapshot, with nonblocking requests for launcher/NVTX failure tests. The initial Bridge implementation worker timed out; existing CPU artifacts were verified by the parent, but its missing work-log link and final documentation require reconciliation.

## Parent post-repair checks

- Tutorial transport now uses an explicit local-only Requests Session with `trust_env=False`, redirects disabled and 3xx rejection. Parent reran original and privacy tests: **15 passed** (7.97 s). Actual fresh post-fix export `outputs/lens-parent-privacy-fixed-01/` ingested **17 spans**, identity and hierarchy verified. These are functional/transport tests, not a new overhead measurement.
- Bridge parent focused suite: **40 passed** (7.33 s), including mocked successful launcher and runtime NVTX failures. Parent actual post-fix CPU control `outputs/lens-bridge-parent-fixed-01/` passed forward/gradient parity. Recovery WORK_LOG.md now exists. Host-local Bridge artifacts are explicitly ignored.
- Independent post-fix re-review **PASSED** for both folders: no security/logic blockers; fresh reviewer tests 15 tutorial / 40 Bridge. Parent verified all reviewed SHA256s match current files. Verdict: `outputs/observability-reviewed-verdict.json`. The separate browser repair is now parent-executed: **38 Chromium/CLI tests passed** (33.70 s), including redirect/external resource/WebSocket/identity/missing-label/JS-error failure gates. A synchronous routed-WebSocket close deadlocked in a tight test; the parent removed that pre-handshake close, retained no-server-connect routing, and verified the destination received no request. Actual final Jaeger rendering at `outputs/lens-ui-parent-hardened-02/` verified full trace ID, service, all five exact operation labels and no page errors for 17 spans. Final main suite: **783 passed, 22 skipped, 25 subtests passed** (28.51 s): 20 opt-in browser cases were exercised separately, plus two existing skips. Main dependency/diff checks passed. Browser independent re-review **PASSED**: fresh reviewer Chromium/CLI suite 38 passed (33.81 s), no security/logic blockers; parent verified all three reviewed file hashes match. Nonblocking coverage suggestions remain for hostile proxy variables, service-worker registration, same-origin WebSockets and popups. Native GPU/Nsight gates remain unverified.

Local backend remains loopback-only on port 16687; access via SSH tunnel as documented. It stores traces in memory and is a learning service, not durable production telemetry.
