# Rollout latency: generation versus real HTTP tools

[한국어](README.ko.md)

A small **executed CPU observability lesson**, not a learned agent, NeMo RL/vLLM/NeMo Gym training run, optimizer, or benchmark. Planning is scripted: each trajectory runs real greedy LM generation, then calls an actual loopback HTTP calculation server, then checks its deterministic result. Generated text does **not** select the tool or its arguments. There are no policy updates or quality claims.

## Environment and scope

Use the existing parent tutorial `.venv` read-only. Exercised: Lens `0.2.0+4d045c0`, OTel `1.43.0`, torch `2.14.1+cpu`, borrowed transformers `4.57.6`. Nothing is installed or changed by these scripts. The model/tokenizer must already be cached:

- `sshleifer/tiny-gpt2`, immutable revision `5f91d94bd9cd7190a9f3216ff93cd1dd95f2c7be`;
- every load uses `local_files_only=True`; missing cache is a failure, never a network download;
- four concurrent trajectories, eight actual new tokens each, one untraced warmup;
- a caller-owned semaphore admits **one CPU generation at a time**, while tools run concurrently;
- queue spans measure this caller semaphore, **not** GPU scheduling, vLLM queueing, prefill or decode.

The existing [parent tutorial](../README.md) explains how to start its loopback Jaeger service. Default OTLP is `127.0.0.1:14318/v1/traces`, UI/API is `127.0.0.1:16687`. This lesson does not launch or modify containers.

## Run from the repository root

Use **fresh, unique output directories**, preferably under ignored root `outputs/`. Run names are output directory basenames; trace IDs and trajectory span IDs are authoritative join keys, not filenames alone.

```bash
P=tutorials/nemo_lens/.venv/bin/python
LAB=tutorials/nemo_lens/rollout
$P tutorials/nemo_lens/verify_jaeger.py   # check the existing service
for scenario in baseline tool-slow tool-straggler; do
  out="results/lens-rollout/my-${scenario}"
  $P "$LAB/run.py" --scenario "$scenario" --mode otlp --output "$out"
  $P "$LAB/diagnose.py" --spans "$out/spans.jsonl" --output "$out/diagnosis.json"
  $P "$LAB/verify.py" --result "$out/result.json"
done
$P "$LAB/test_rollout.py" -v
$P -m pytest --import-mode=importlib -q "$LAB"
```

Offline alternative (no collector required):

```bash
$P "$LAB/run.py" --mode file --scenario tool-slow --output results/lens-rollout/my-offline
$P "$LAB/diagnose.py" --spans results/lens-rollout/my-offline/spans.jsonl \
  --output results/lens-rollout/my-offline/diagnosis.json
```

Both modes preserve actual OTel JSON spans. OTLP mode additionally sends those same spans through the real Lens exporter pipeline. `result.json` means execution/export succeeded; only `jaeger_receipt.json` proves ingestion, complete counts, resource identity and hierarchy. The verifier compares **every local span ID** with the Jaeger response. Failures retain `failure.json` and raw spans, return nonzero, and create no successful result/receipt. Diagnosis rejects error or incomplete stage coverage. All artifact writers refuse existing paths.

## Read the Jaeger waterfall

Select service **`nemochat-lens-rollout`**, operation **`demo.rl.batch`**, Last Hour, Find Traces. Open the direct `ui_url` in `result.json`, or `/trace/<trace_id>`. Expand:

```text
demo.rl.batch
  demo.rl.trajectory  [demo.trajectory.id = 0..3]
    demo.rl.caller_queue
    demo.rl.generation
    demo.rl.tool.client  [span.kind = client]
      demo.rl.tool.server  [span.kind = server, remote W3C parent]
    demo.rl.reward
```

The service resource carries `nemo.run.id` and `tutorial.device=cpu`. HTTP injection/extraction sends only W3C trace context; `asyncio.to_thread` preserves the caller context, and the server explicitly attaches the extracted remote context. Concurrent trajectory IDs must stay matched along trajectory → client → server. Server and client **share the same trace**; server is a child of the exact client span, not merely a similarly named span. This is actual HTTP, despite both endpoints being in one process.

- **baseline:** no artificial server delay; inspect CPU admission and generation.
- **tool-slow:** every tool handler sleeps **250 ms**; overlapping tool calls increase batch wall time far less than their sum.
- **tool-straggler:** only the final trajectory's tool sleeps **750 ms**; inspect the long last-finisher tail.
- Server `demo.injected_delay_ms` records the deliberate intervention. It is not discovered production latency or fabricated engine timing.

If the host is remote, forward UI on your laptop with `ssh -N -L 16687:127.0.0.1:16687 ubuntu@YOUR_HOST`. Keep the service private. Jaeger is in-memory; stable **saved** trace/span IDs remain in the artifacts, but the live trace disappears after service restart. Browser screenshots are a parent-task responsibility; this lab verifies the API, not rendered UI.

## Diagnosis without double counting

`diagnose.py` reads actual span timestamps (nanoseconds preserved) and emits:

1. Sample size, each trajectory's four **sequential direct-child stage** durations and unknown/uninstrumented gaps. It rejects stage overlap rather than pretending stages are exclusive.
2. Nested server duration and injected delay for inspection, **excluded from stage totals**: client already contains server time. Never add batch + trajectory + stages, or client + server.
3. Concurrent **batch wall** and **sum of trajectory walls** separately. The sum is work/latency accounting, not elapsed time or utilization.
4. Last-finisher ID, its largest stage, and interval overlap by stage **after the penultimate trajectory finished**. A large queue total can coexist with generation on the final tail.

These nominate observed bottleneck **candidates**, not guaranteed causal production bottlenecks. The tail is a temporal attribution, not a full dependency-DAG critical-path proof. OS scheduling, Python executor dispatch, HTTP handling and instrumentation contribute to timings/gaps. No engine-level queue, GPU kernel, network-only latency, or prefill/decode decomposition is established.

## Executed evidence

Raw runs: `outputs/lens-rollout/{baseline-01,tool-slow-01,tool-straggler-01}/`. Each run has four trajectories and **25 spans**; all three ingested with exact trace/span identity, operation counts and client/server parent hierarchy verified. All generated eight tokens and passed the scripted tool correctness check.

| Scenario | Batch wall ms | Sum trajectory wall ms | Tail candidate | Tail after penultimate ms |
|---|---:|---:|---|---:|
| baseline | 89.718 | 231.254 | generation | 19.370 |
| tool-slow | 339.804 | 1228.233 | tool.client | 20.290 |
| tool-straggler | 838.455 | 975.902 | tool.client | 769.496 |

These are **one observed batch per scenario**, not repeated performance comparisons; some runs overlapped regression activity. No confidence intervals or production speedup are claimed. Baseline's last-finisher *total* dominant stage was caller queue, but its final tail was mostly generation. Read both outputs before proposing a change.

Trace IDs:

- baseline: `a6ad22ec0e2b47e559e4f3df03604a3b`
- tool-slow: `d10ed445dd3ad207e75410ab28c5ef3b`
- tool-straggler: `78beb8b3b7a046a7d9c6659740b2258f`

See [machine-readable evidence](executed_results.json). Eight focused tests exercise real generation, concurrent HTTP intervals, context identity/kinds, HTTP errors, socket timeout, overwrite refusal, exclusive accounting/overlap rejection, privacy, netrc/proxy isolation and redirect refusal for both tools and Jaeger API. Test subprocess outputs remain under `outputs/lens-rollout-tests/<unique-id>/`.

## Safety and limitations

- Reuses parent's `clean_environment`, hardened `local_otlp_session`, `loopback_endpoint`, and `get_json` explicitly via source-tree import. Does not load `.env`; metrics/logs are disabled.
- Tool socket binds only numeric `127.0.0.1` on an ephemeral port. Requests use `trust_env=False`, refuse redirects, reject nonlocal/credential-bearing URLs, and cap response bytes. Jaeger API bypasses environment proxies and refuses redirects too. No ambient netrc auth is sent.
- No prompts, generated text, tool arguments or tool results appear in span attributes. Failures use value-free messages. Raw JSON spans still contain source locations and run identity; keep artifacts private.
- `--timeout` is requests' connection/read **socket inactivity timeout**, not a strict total generation/request deadline. Failed requests remain failed even if the server completes later; shutdown waits for handlers before draining telemetry.
- The tiny pretrained model is an inference control only; reward checks the **scripted tool**, not generated reasoning. This is not a learned tool-use policy, multi-turn agent, native RL training integration or benchmark.

## Native RL provenance, not an implementation claim

The inspected [NeMo RL span groups](https://github.com/NVIDIA-NeMo/RL/blob/67fc59b2ccc6db48dc873eefa86d64506ca38126/docs/observability/span-groups.md) distinguish rollout/generation/reward and umbrella spans. [vLLM tracing](https://github.com/NVIDIA-NeMo/RL/blob/67fc59b2ccc6db48dc873eefa86d64506ca38126/docs/observability/vllm-tracing.md) distinguishes driver spans from native engine spans, coverage gaps and run-level versus parent-child correlation. This lab uses its own `demo.rl.*` names deliberately: native instrumentation, hardware timing and production causal attribution require separate runs and coverage validation.

## Final cycle-2 acceptance

The independent re-review passed for this scripted CPU/source-tree scope. Run the complete pytest command above for **23 tests and 57 subtests** (the original standalone unittest command runs eight normal-path cases). Local JSON/remote receipt/schema, exporter failure, resource cleanup, boolean identities and malformed timestamps are covered.

The pinned OTel 1.43.0 JSON formatter emits six rounded fractional digits, not original nanosecond precision. Diagnosis preserves the precision actually stored; Jaeger floors original start/duration to microseconds. Receipt correspondence allows only the validated 1us start / 2us duration-end / 1us computed parent-end quantization; it never rewrites recorded timestamps. Jaeger float64 tags may appear as integral JSON numbers; semantic type checks retain strict bool/int identities.

Parent's final three runs each verified 25 spans and exact hierarchy/timestamps at `outputs/precommit-whole/final-cycle2-{baseline,tool-slow,tool-straggler}/`. Fresh Chromium rendered straggler trace `631328f77ad238252a254beb63c04620`, all seven operation labels and 25 spans with no JS errors; private artifact `outputs/precommit-whole/final-cycle2-ui/trace.png`. The evidence JSON preserves earlier observations separately. This acceptance is not native RL/Gym/vLLM/Nsight/GPU validation.

## Runtime layout / 실행 자료 경로

Use repository-root `datasets/`, `checkpoints/`, `results/`, and `logs/`; see [runtime layout](../../../docs/en/RUNTIME_LAYOUT.md) and [한국어](../../../docs/ko/RUNTIME_LAYOUT.md). Relative CLI output paths use the caller working directory. Historical recorded paths/measurements below remain unchanged; prior host artifacts physically migrated, with relative compatibility links and SHA-256 inventory in root `results/runtime-layout-migration.json`. Tutorial benchmark stdout/stderr now go to root `logs/nemo_lens/<output-name>/` (override `--log-output`); use a fresh name for both results and logs.
