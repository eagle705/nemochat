# NeMo Lens ↔ Nsight Systems bridge MVP

An **explicit-import, configuration-plugin educational control**, not another web
application, a GPU profiler, or a native Nemotron/Megatron adapter. NeMo Lens emits
OpenTelemetry spans; Nsight Systems collects the GPU timeline. Their roles stay
separate. [한국어 설명](README.ko.md).

## Delivered and pending

| Capability | Evidence / status |
|---|---|
| Sparse MoE forward and backward, token-major reference parity | Actual CPU exercised; 8 tokens, width 16, hidden 24, 4 experts, top-2 |
| Local Lens spans and common bridge/source identities | Actual pinned Lens exercised; 21 spans and 21 operation records |
| Real Python source snapshots, strict snippet/hash verification | Actual tests + CLI; no generated/pseudocode snippets |
| Optional torch CUDA NVTX push/pop path | Implemented; unavailable requests fail; real CUDA emission **not exercised** |
| Official `TargetEnvironment` YAML + discovery-aware launcher | Official docs source-checked; installed Nsight validation **blocked** |
| Kernel → source attribution / CUDA backtrace | **Unverified**; requires actual correlated capture, not timestamp proximity |
| Native Nsight explanation/code panel | **Pending**; public configuration/collection plugin APIs are not a custom GUI-widget API |
| Native fused Nemotron / Megatron / EP adapter | **Pending**; this demo has no distributed communication |

## CPU quick start (from this directory)

**Source-tree-only installation/usage:** run from this checkout (or use an
editable install retaining it). The current wheel does not package the Nsight
manifest, and `cli.BASE/nsight` assumes this source layout. A standalone wheel
installation is not a supported Nsight plugin deployment; packaging/resources
remain pending rather than being silently claimed to work.

```bash
cd /home/ubuntu/nemochat/plugins/nemo_lens_nsight_bridge
# Keep all optional installs in this plugin's environment, not the project .venv.
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r requirements.txt
# Install torch in this environment if it is not shared, or use a shared interpreter.
# This host's local .pth shares /home/ubuntu/nemochat/.venv/lib/python3.11/site-packages;
# it is machine-local, ignored, and never changes the project's environment.
.venv/bin/python -m lens_nsight_bridge doctor
.venv/bin/python -m lens_nsight_bridge run --lens \
  --output /home/ubuntu/nemochat/results/my-lens-moe-cpu
.venv/bin/python -m pytest -q
```

Use a **fresh output path**. No overwrite/resume option exists. Omit `--lens` for
source/operation-only validation with no Lens import. Without `--nvtx` or the
explicit plugin enable environment, CPU mode performs **no NVTX calls**. `--nvtx`
on this CPU-only host fails instead of claiming capture. CUDA is explicit via
`--device cuda`. Torch and pytest are shared local prerequisites; Lens is pinned
at `4d045c03186b90afdc815db46b671f6a21bc7e98` (installed version
`0.2.0+4d045c0`). Lens runs a custom **local file exporter**, with metrics/logs off;
there is no OTLP exporter, collector, browser, server, or external telemetry.
Network is used only for installation, not by bridge runtime export.

Lens/OTel setup is process-global: create one bridge before other telemetry setup
in a fresh process. The MVP **rejects existing OTel providers**, disabled SDK, and
non-recording spans (e.g. `OTEL_TRACES_SAMPLER=always_off`), rather than silently
writing local spans while another exporter sends them elsewhere. It does not
attach itself to an already-running Lens application. Existing native run/step/
layer context reuse is a later explicit adapter gate.

## Existing Nsight workflow, not a GUI extension

`nsight/nemo_lens_bridge.yaml` is an official **configuration** manifest with
`PluginName`, `Description`, and `TargetEnvironment`; it deliberately has no
`LibraryPath`, `ExecutablePath`, `sitecustomize`, or injection machinery.
`TargetEnvironment` sets two flags. Python user code still must **explicitly
import** `Bridge` and wrap its own operations. Merely enabling this manifest
cannot instrument an arbitrary Python training script.

```bash
# Requires actual installed compatible Nsight + CUDA. This host is blocked.
.venv/bin/python launch.py launch \
  --output /home/ubuntu/nemochat/results/my-lens-moe-cuda
# Equivalent discovery/configuration shape (verify installed CLI help first):
export NSYS_PLUGIN_SEARCH_DIRS="$PWD/nsight"
nsys plugins list
nsys profile --enable=nemo_lens_nsight_bridge --trace=cuda,nvtx \
  --output=/home/ubuntu/nemochat/results/my-lens-capture \
  "$PWD/.venv/bin/python" -m lens_nsight_bridge run --device cuda --lens \
  --output /home/ubuntu/nemochat/results/my-lens-moe-user-code
```

`launch` probes the installed `--version`, `profile --help`, `plugins --help`, and
`plugins list`; refuses missing `--enable`, plugin listing errors/validation
errors, absent GPU, or existing output prefix; preserves inherited search dirs.
The manifest flags are read by the **explicitly imported** CLI/example, not an
automatic injector. `doctor` reports full probe stdout/stderr/return codes and
backtrace/sampling help mentions. Do **not** assume generic Python backtrace
flags: CUDA backtrace and sampling configuration need actual version-specific
help and a verified collected launch/backtrace chain. No such options are claimed
tested here. `examples/explicit_import.py` shows the application opt-in pattern.

Open the resulting `.nsys-rep` in the existing Nsight Systems GUI. Select NVTX
ranges like `MoE/activation: SwiGLU | bridge=... src=...`; the range title explains
the host operation and carries exact lookup IDs. Keep raw CUDA kernel names
unchanged. On a GPU host, inspect CUDA API launch correlation, streams, and NVTX
containment before attributing any kernel; **CPU evidence cannot prove that map**.
The detailed role cards and snippet utility below are local companions, **not a
panel installed into Nsight**. Copy an exact bridge ID from the range label:

```bash
.venv/bin/python -m lens_nsight_bridge snippet \
  --output /home/ubuntu/nemochat/results/my-lens-moe-cpu \
  --bridge-id '<exact bridge ID from operations.jsonl or captured NVTX>' \
  --current-root "$PWD"
```

This prints actual snapshot code only after verification. Without `--current-root`
the immutable captured snapshot is verified and can be read after checkout drift;
with it, changed current code is rejected. No editor/deep-link integration is
claimed. A native UI panel/source-opening workflow remains a separate SDK/version
and customer-workflow acceptance gate.

## Evidence files and source trust boundary

- `operations.jsonl`: every explicit host operation, bridge ID, role card, status,
  host monotonic timestamps, NVTX emission flag, complete source reference.
  Runtime NVTX push/pop failures attempt an error record with `nvtx_error` stage
  and exception type (no message); `nvtx_emitted` means push returned successfully,
  not proof of capture. Pop failure propagates unless preserving a wrapped-call
  exception. Record I/O is best-effort when another error is already propagating.
- `lens-spans.jsonl`: **real NeMo Lens** OTel spans with the same bridge ID, source
  path/start/end, source and snippet SHA256s, explanation, and NVTX label.
- `sources/<whole-file-sha256>.py`: exact captured source bytes, content-addressed.
- `result.json`: successful control/reference outcome only; absent on failure.

Before executing a wrapped callable, the bridge validates source root containment,
compiles (never executes) the file to compare the **loaded Python code object**,
and checks inspect's extracted lines agree with captured bytes. It refuses stale
loaded code vs modified source. This small MVP targets ordinary undecorated Python
functions; wrappers, dynamically generated/C-extension code, or transformed code
must use a future evidence-specific adapter, not weakened checks.

Extraction verifies snapshot location, file hash, source-reference identity, exact
1-indexed inclusive line bounds, snippet hash, and optionally live checkout hash.
Absolute paths, `..`, backslashes, and symlink escapes are rejected. Hashes detect
accidental drift/tampering; they do **not authenticate** a maliciously rewritten
manifest and snapshot together. Source/records are local sensitive customer data:
review storage permissions/retention before capturing, and do not upload them by
default. The implementation is single-process/synchronous, not a distributed or
concurrent-write-safe exporter. Snapshot compilation/extraction and spans add
host overhead: **diagnostic control only**, not a throughput measurement.

## Role cards and attribution boundaries

See [operation glossary/source cards](docs/ROLES.md),
[한국어 역할 카드](docs/ROLES.ko.md), and [executed work log](WORK_LOG.md).
The readable control uses routing scores → selected-weight softmax/top-k → token
copy sorting → local gather → expert up GEMM → eager SwiGLU → down GEMM → weighted
scatter-add. It never pretends local sorting/gather is EP all-to-all.
Forward and gradient parity compare the dispatch path with a token-major reference;
they share routing/top-k weights. Unannotated reference checks and autograd.grad
validation also execute in the diagnostic run and may produce GPU kernels on a
future CUDA run; **not every captured kernel is claimed annotated**.

An explicit scope may launch many eager kernels; fused compilation may merge many
source ops into one kernel; a scope can launch no kernel. Autograd entry is not a
unique forward source for gradient kernels. CUDA Graph replay may launch previously
captured work with a different host stack. Never infer roles by neighboring time,
NVTX nesting alone, duration, or a guessed kernel-name glossary. Preserve unknowns
until launch correlation/backtrace/compiler metadata establishes a supported map.

## Official references and existing native annotations

- [Nsight developing plugins](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#developing-an-nsight-systems-plugin),
  [manifest](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#plugin-manifest-file),
  [discovery](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#plugin-discovery).
  Official docs state configuration plugins amend target environment;
  `NSYS_PLUGIN_SEARCH_DIRS` locates manifests; `nsys plugins list` validates them.
  Any discovered invalid manifest can block enablement. Plugin dirs are trusted
  executable configuration; only use controlled directories.
- [Pinned Lens file exporter guide](https://github.com/NVIDIA-NeMo/Lens/blob/4d045c03186b90afdc815db46b671f6a21bc7e98/docs/observability/backends.mdx)
  and [setup API](https://github.com/NVIDIA-NeMo/Lens/blob/4d045c03186b90afdc815db46b671f6a21bc7e98/src/nemo/lens/handle.py).
- Inspected local research consumer snapshot:
  `/home/ubuntu/research/nemo_lens/consumers/NVIDIA/Megatron-LM/megatron/core/transformer/moe/moe_layer.py`
  imports `nvtx_range_push/pop` at line 39 and wraps `delayed_expert_wgrad` at
  lines 873–875. This is a **partial snapshot inventory**, not a full pinned
  runtime inventory or permission to duplicate Megatron annotations. Before a
  native adapter: resolve actual installed Megatron commit, inventory router /
  dispatcher / grouped experts / combine / backward annotations and Lens groups,
  reconcile native scopes with explicit IDs, then exercise distributed/CUDA
  correctness and instrumentation overhead. No broad global monkeypatch is used.

## Runtime layout / 실행 자료 경로

Use repository-root `datasets/`, `checkpoints/`, `results/`, and `logs/`; see [runtime layout](../../docs/en/RUNTIME_LAYOUT.md) and [한국어](../../docs/ko/RUNTIME_LAYOUT.md). Relative CLI output paths use the caller working directory. Historical recorded paths/measurements below remain unchanged; prior host artifacts physically migrated, with relative compatibility links and SHA-256 inventory in root `results/runtime-layout-migration.json`. Tutorial benchmark stdout/stderr now go to root `logs/nemo_lens/<output-name>/` (override `--log-output`); use a fresh name for both results and logs.
