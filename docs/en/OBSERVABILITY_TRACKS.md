# NeMo Lens tutorial and Nsight Bridge

For new CLI runs, use repository-root paths and fresh artifacts; see [runtime layout](RUNTIME_LAYOUT.md). Inputs belong in `datasets/`, checkpoints in `checkpoints/`, reports/traces/plots in `results/`, and text/W&B logs in `logs/`. Create required parent directories, not final no-overwrite output directories. Checked-in fixtures/configs and historical evidence paths remain unchanged.

## Two distinct deliverables

- [NeMo Lens tutorial](../../tutorials/nemo_lens/README.md): learn application telemetry, run an offline training control, export actual spans, open the local trace UI, and measure instrumentation overhead.
- [NeMo Lens–Nsight Bridge](../../plugins/nemo_lens_nsight_bridge/README.md): attach modeler-readable MoE meaning and source identity to profiling annotations in the existing Nsight Systems workflow. It is not a new web dashboard.
- [Korean companion](../ko/OBSERVABILITY_TRACKS.md).

These are independent, opt-in lab extensions. They must not change the default training loop, dependency environment, WSD/WSM policy or checkpoint/resume behavior. No GPU rental, external telemetry, commit or publication is authorized by this addition.

## Shared source investigation

NeMo Lens source inspected at `NVIDIA-NeMo/Lens@4d045c03186b90afdc815db46b671f6a21bc7e98`:

- [README](https://github.com/NVIDIA-NeMo/Lens/blob/4d045c03186b90afdc815db46b671f6a21bc7e98/README.md): OTel instrumentation API, consumer-declared groups and early-development status.
- [Architecture](https://github.com/NVIDIA-NeMo/Lens/blob/4d045c03186b90afdc815db46b671f6a21bc7e98/docs/design/architecture.mdx): GPU profiling belongs to PyTorch profiler or `nsys`, not Lens.
- [Demo stack](https://github.com/NVIDIA-NeMo/Lens/blob/4d045c03186b90afdc815db46b671f6a21bc7e98/docs/observability/stack.mdx): Jaeger, Grafana and Kibana are example backends, not a supported production architecture. The upstream stack defaults to Weave routing; it must not be copied blindly for a local tutorial.
- [Megatron span groups](https://github.com/NVIDIA/Megatron-LM/blob/d7157825675a65c2503a061ffd1fffa7886df9b7/docs/user-guide/observability/span-groups.md): existing step, microbatch, layer, MLP and communication spans should be reused rather than duplicated.
- [Nsight Systems plugin documentation](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#developing-an-product-name-plugin): collection/configuration/in-process plugins are not evidence of a public custom GUI-widget API. Check the installed version and its actual help before using a manifest.

An investigation pin is not a complete compatible GPU lock. Each runtime records package versions and executed scope separately.

## Tutorial acceptance gates

1. **SDK execution:** a real Lens exporter receives spans from actual offline optimizer updates. A handwritten fixture is labeled; no downloaded corpus or target-model quality claim.
2. **Backend ingestion:** export to the dedicated loopback-only local backend and query actual stored spans. Backend errors must not count as success.
3. **UI:** load the actual UI, select the service, search and open a trace. HTTP health and trace API responses are not rendered-UI verification. Record screenshots or an explicit browser blocker.
4. **Overhead:** same workload/input/seed/thread count, fresh subprocesses, warmup, interleaved repeated cases, raw durations, median comparisons, correctness checks and separate flush/shutdown timings. Enabled telemetry is not presumed free. Small CPU results are not native MoE GPU overhead.
5. **Native extension:** repeat on real MoE training, selected rank subsets, coarse/fine groups, exporter load and failure conditions. Full-update throughput, memory and synchronization costs are required before production advice.

## Bridge acceptance gates

1. **Semantic instrumentation:** explicit MoE scopes (routing, expert selection, token organization, expert compute, combine, optional shared expert) with compact descriptions and stable operation identifiers. Separate host work from GPU execution.
2. **Source evidence:** extract snippets from a verified profiled source snapshot, with file hash and line bounds; refuse changed files, traversal and ambiguous claims. Never generate code snippets with an LLM.
3. **Nsight plugin:** exercise actual discovery and enablement in a compatible installed Nsight version. Manifest syntax checks alone are not successful plugin loading.
4. **CUDA trace:** preserve raw kernel names and use CUDA correlation, actual captured backtraces or explicit annotation evidence. No association by timestamp proximity. One-to-many eager and many-to-one fused mappings remain explicit.
5. **Modern MoE backend:** inventory existing annotations; cover selected dispatch/combine backend, grouped GEMM and fused communication/compute. A CPU educational sparse-MoE control does not establish EP/NVLink/RDMA support.
6. **Native GUI:** custom explanation and snippet panels require verified APIs or Nsight team collaboration. Do not claim current annotation support is that panel.
7. **User utility:** have modelers identify an operation and the next investigation step with/without the annotations; assess correctness as well as time saved.

## Integration hypothesis, not implemented upstream support

One explicit scope may emit both a Lens span and an NVTX annotation carrying a shared bridge identity. This could connect run/step/layer context to a short Nsight diagnostic capture. A correlation ID does not prove a kernel's algorithm, shape, critical-path contribution or originating forward line. Backward, compiled fusion, worker threads and CUDA Graph replay need separate evidence. Do not automatically convert every distributed OTel span into a GPU range.

## Optional real-browser UI verification

Backend API success is not proof that the UI renders. `scripts/verify_lens_ui.py` uses a separate Playwright runtime to open the actual Jaeger search page and a stored training trace, verify visible operation names and save screenshots. It does not create a new web viewer. From the repository root:

```bash
python3 -m venv .venv-lens-browser
.venv-lens-browser/bin/python -m pip install playwright
.venv-lens-browser/bin/python -m playwright install chromium
.venv-lens-browser/bin/python scripts/verify_lens_ui.py \
  --url http://127.0.0.1:16687 --output results/observability/lens-ui-check
```

Start the tutorial backend and export a training trace first. Use `--trace-id` from an actual `result.json` to inspect a specific short trace. Output paths must be fresh. Browser installation requires network access and disk space; missing OS libraries or blocked downloads are explicit blockers, not simulated screenshots. The browser verifier is optional and not part of the base package dependencies. It verifies Jaeger, **not the native Nsight GUI**.

Screenshots and captured page text contain trace content; retain them in private ignored output directories and do not upload customer data by default. The browser verifier enforces the approved origin, rejects redirects/external requests, blocks service workers/WebSockets, and checks full trace identity, service and every expected operation in trace-specific DOM elements. Failures produce no success receipt.

The explicit browser regression suite is opt-in. On this host it was run with the browser interpreter and a read-only pytest import path from the project environment (no changes to main dependencies):

```bash
PROJECT_SITE=$(.venv/bin/python -c 'import site; print(site.getsitepackages()[0])')
PYTHONPATH="$PROJECT_SITE" NEMOCHAT_LENS_BROWSER_TESTS=1 \
  .venv-lens-browser/bin/python -m pytest -q \
  tests/test_lens_ui_browser.py tests/test_lens_ui_cli.py
```

Use your actual browser environment path; the parent used `/home/ubuntu/runs/nemo-lens-browser/.venv`. Default main tests skip Chromium cases deliberately; explicit runs are required before claiming those gates passed.

## Privacy and operation boundaries

Use dedicated local services bound to loopback; access through an SSH tunnel rather than opening public ports. Do not use ambient cloud exporter credentials. Customer code/paths may be sensitive even if tensor values are excluded. Keep source snippets local and do not send them to an external model by default. New artifacts must use fresh ignored output directories. Stop only services created by this tutorial, never unrelated containers.
