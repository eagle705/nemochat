# NeMo Lens: actual CPU training → local Jaeger traces

[Korean walkthrough](README.ko.md)

This is an **offline generated-data CPU MLP control**, not NemoChat's hybrid model,
GPU training, a CUDA profiler, or evidence of model quality. It performs real PyTorch
forward, backward and SGD updates. Lens adds semantic spans; it does **not** reveal
CUDA kernel timings, Nsight launch correlations, or kernel-to-source attribution.
No cloud exporter, W&B account, API key, model download or dataset download is used.
Package/image installation needs network access; training itself stays local.

## Provenance and isolation

- Official SDK: [NVIDIA-NeMo/Lens](https://github.com/NVIDIA-NeMo/Lens/tree/4d045c03186b90afdc815db46b671f6a21bc7e98),
  commit `4d045c03186b90afdc815db46b671f6a21bc7e98`, installed as
  `nemo-lens==0.2.0+4d045c0` from pinned Git, not copied source. Upstream is Apache-2.0.
- OpenTelemetry SDK/API/exporters `1.43.0`; `requirements.lock` fixes the resolved
  tutorial dependencies (not a lock of the parent project's torch dependencies).
- Jaeger all-in-one `1.76.0` is pinned by image digest in `compose.yaml`.
  It is a minimal in-memory learning service, **not a production deployment**.
- Exercised Python `3.11.15`, project torch `2.14.1+cpu`. The tutorial's own ignored
  `.venv` borrows the project's site-packages **read-only**, through a `.pth` file.
  Tutorial-installed Lens/OTel take precedence. No root dependency or venv is changed.
- `train.py` clears inherited OTEL/Lens/W&B/scheduler/proxy settings before SDK
  import, explicitly disables metrics/logs, passes a credential-free loopback HTTP
  exporter, and rejects nonlocal endpoints. It does not load `.env`. The explicit
  requests Session has `trust_env=False`: ambient netrc authentication and environment
  proxies are not consulted. It forces `allow_redirects=False`; every 3xx response
  fails export without a second request, including 307/308 trace-body redirects.

## Setup and start

From the repository root, with its CPU `.venv` already installed and `uv` available:

```bash
cd tutorials/nemo_lens
python3 setup_env.py                  # refuses to replace an existing tutorial .venv
# Confirm ports are free; choose other host ports in compose.yaml if needed.
ss -ltn
# Dedicated project name: nemochat-lens-tutorial. No unrelated containers touched.
docker compose up -d
.venv/bin/python verify_jaeger.py      # bounded readiness polling, not a blind sleep
.venv/bin/python train.py --mode otlp --steps 20 --output runs/my-smoke
.venv/bin/python verify_jaeger.py --result runs/my-smoke/result.json
```

Ports were checked free before launch: UI `127.0.0.1:16687`, OTLP gRPC
`127.0.0.1:14317`, OTLP HTTP `127.0.0.1:14318`. This example exports HTTP
`/v1/traces`; gRPC is exposed only for optional future experiments.
If you change ports, pass `--endpoint http://127.0.0.1:PORT/v1/traces` and
`--api http://127.0.0.1:UI_PORT` to the relevant scripts. Outputs must be fresh.

## Read the code and the UI

1. `setup_env.py`: create isolated dependencies without reinstalling torch.
2. `train.py`: deterministic seed, one CPU thread, fixed random tensors, tiny MLP;
   register consumer-owned `run`, `step`, `phase` groups **before** `setup_telemetry`.
   `managed_span` nests `train.run → train.step → train.forward/backward/optimizer`.
   `nemo.run.id`, rank and CPU metadata identify the run. All enabled spans use the
   imported SDK and real batch exporter; `handle.shutdown()` drains it.
3. `verify_jaeger.py`: query the exact trace ID, validate every operation count,
   service/run identity and parent-child link, and save full Jaeger response + receipt.
4. `benchmark.py`: isolated subprocess comparisons and full-export acceptance gates.

Open **http://127.0.0.1:16687**. On Jaeger's Search page, select service
`nemochat-lens-cpu`, operation `train.run`, choose Last Hour and Find Traces.
Open a trace; expand `train.run`, then a `train.step`. Inspect the three phase spans,
`train.step` tag and process tags (`nemo.run.id`, `nv.dl.rank`, `tutorial.device`).
The horizontal bars measure CPU code-block wall time, including instrumentation,
not individual GPU kernels. The direct URL is `/trace/<trace_id>` from `result.json`.

For a remote host, on **your laptop**:

```bash
ssh -N -L 16687:127.0.0.1:16687 ubuntu@YOUR_HOST
```

Then use your laptop's `http://127.0.0.1:16687`. Do not expose the UI publicly.
API ingestion and actual Chromium rendering were exercised by the parent.
Screenshots remain private ignored host-local artifacts rather than bundled outputs;
see [parent verification](../../docs/en/OBSERVABILITY_VERIFICATION.md). Traces are
in-memory and disappear on restart. Cleanup only this project's service:

```bash
docker compose down
```

## RL rollout / tool bottleneck lesson

The [rollout lab](rollout/README.md) ([Korean](rollout/README.ko.md)) extends this tutorial to concurrent trajectories, local model generation, actual HTTP tools and verification. Inspect caller admission, generation, tool client/server and last-finisher latency without adding overlapping spans twice. Explicitly injected delay is an educational fixture, not native RL/vLLM/Gym performance evidence. Read the [pinned native RL/Gym integration guide](../../docs/en/RL_ROLLOUT_OBSERVABILITY.md) ([Korean](../../docs/ko/RL_ROLLOUT_OBSERVABILITY.md)) for existing span groups, HTTP propagation, connection-pool waits and coverage limits.

## Measure overhead, not “zero overhead”

```bash
.venv/bin/python benchmark.py --output runs/my-benchmark --steps 400 --warmup 40 --repetitions 8
# Without Docker, run baseline / disabled / file only (OTLP explicitly omitted):
.venv/bin/python benchmark.py --output runs/my-offline --skip-otlp
.venv/bin/python -m pytest -q test_tutorial.py
uv pip check --python .venv/bin/python
```

Each arm uses a **fresh interpreter**. Baseline neither imports Lens nor enters
its wrapper. Disabled installs no-op providers but enters the same wrappers.
File mode uses OTel's `ConsoleSpanExporter` redirected to newline-delimited JSON
on disk; OTLP mode sends actual spans to local Jaeger. All arms perform identical
untraced warmup updates before the timed updates. Initial imports/model construction
are excluded from `training_ns`; SDK setup and process wall time are retained
separately. Export runs asynchronously and **can overlap training**; shutdown is
reported separately, not silently removed from end-to-end cost.

Eight repetitions rotate the arm order, balancing each of four positions twice.
Raw durations, order, medians and median-relative overhead are saved. Every loss
and final parameter SHA256 must match exactly; file export must contain all spans;
OTLP must pass API validation for **every** repetition. A missing/partial export
fails the benchmark rather than counting as fast success. Queue size is 65,536;
large workloads beyond this small control may need separate queue/drop analysis.
This is a single-host noisy measurement, not a universal SDK overhead estimate.
It does not isolate pure wrapper cost from exporter scheduling/serialization and
has no GPU or distributed applicability.

## Historical executed evidence (before exporter privacy fix)

The measurements below and `executed_results.json` are unchanged **pre-fix**
results. They do not measure the hardened Session; no new overhead benchmark
was run for this privacy-only change.

Actual commands (this folder, not the repository-wide suite):

```bash
.venv/bin/python train.py --mode otlp --steps 20 --output runs/smoke
.venv/bin/python verify_jaeger.py --result runs/smoke/result.json
.venv/bin/python benchmark.py --output runs/benchmark-8x400 --steps 400 --warmup 40 --repetitions 8
.venv/bin/python -m pytest -q test_tutorial.py
uv pip check --python .venv/bin/python
.venv/bin/python -m compileall -q train.py benchmark.py verify_jaeger.py setup_env.py test_tutorial.py
docker compose config --quiet
```

Result: **9 tests passed**, compatible tutorial dependencies. Smoke ingested all
**81 spans**; each of 8 OTLP benchmark traces ingested all **1,601 spans** with
verified identity and hierarchy. Exact loss sequences and parameter hashes match
across all 32 training subprocesses.

| Mode | Median timed training (ms / 400 updates) | Median shutdown (ms) | Training overhead vs baseline |
|---|---:|---:|---:|
| baseline, no Lens | 176.995 | 0.000315 | reference |
| disabled wrappers | 187.138 | 0.012530 | +5.73% |
| enabled file | 426.042 | 6.384910 | +140.71% |
| enabled local OTLP | 323.105 | 3.955455 | +82.55% |

Overhead is **nonzero and substantial** for this tiny workload and four spans per
step. Larger operations and coarser span groups require new measurements, not
extrapolation. JSON file formatting can cost more than local OTLP here.

- [Versionable measured summary + raw durations](executed_results.json).
- Host-local full outputs (ignored): `runs/smoke/{result,jaeger_trace,jaeger_receipt}.json`
  and `runs/benchmark-8x400/benchmark.json`.
- Each benchmark child directory keeps losses, stdout/stderr and exported spans or
  complete API response/receipt. Re-run with a **different output directory**.

## Exporter privacy-fix verification

Fresh local validation (not a timing comparison):

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider test_tutorial.py test_exporter_privacy.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python verify_jaeger.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python train.py --mode otlp --steps 20 --output runs/privacy-fix-smoke
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python verify_jaeger.py --result runs/privacy-fix-smoke/result.json
```

**15 tests passed** (9 existing + 6 privacy regressions). The privacy tests mock
netrc lookup, environment proxies and HTTP transport, use only synthetic credentials
and bodies, and perform no network I/O or real credential reads. They verify 307/308
export failure with exactly one loopback POST, even when redirects are explicitly
requested. A separate real local smoke ingested **81 spans**, with verified identity
and hierarchy, trace `a9e445e371045b8abdcfae78165c65c3`.
Fresh artifacts: `runs/privacy-fix-smoke/{result,jaeger_trace,jaeger_receipt}.json`.
No new performance claim is made.

Acceptance: SDK import/install, actual optimizer updates, correctness parity,
real complete local export, loopback readiness and focused tests **exercised**.
Rendered GUI, GPU/Nsight, CUDA/source attribution, native NemoChat integration and
production observability **not established** by this tutorial.
