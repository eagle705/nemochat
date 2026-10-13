# Independent Inference Performance Track

For new CLI runs, use repository-root paths and fresh artifacts; see [runtime layout](RUNTIME_LAYOUT.md). Inputs belong in `datasets/`, checkpoints in `checkpoints/`, reports/traces/plots in `results/`, and text/W&B logs in `logs/`. Create required parent directories, not final no-overwrite output directories. Checked-in fixtures/configs and historical evidence paths remain unchanged.

## Purpose and evidence boundary

This track targets the ability to turn performance models and profiler evidence into reproducible engine improvements. It complements, rather than replaces, the [pretraining performance lab](PRETRAIN_PERFORMANCE_LAB.md). Preserve the existing WSD/WSM method and training resume contracts.

Two CPU-first building blocks are implemented and fixture-tested with this track: strict saved-request analysis and a single-operator roofline calculator. They are not a load generator, serving engine, profiler parser, quality evaluator or performance improvement. Synthetic fixtures demonstrate contracts only. Actual GPU export/serving, quality checks, streaming instrumentation, Nsight traces and upstream changes have separate gates. Never describe an imported timestamp as independently measured by this project.

## Model and engine strategy

Start with one pinned inference engine and a pretrained model explicitly supported by that engine/version. Choose based on native model support and available hardware, not a claim that one engine is universally faster. Do not require full NemoChat pretraining before beginning serving performance work.

- Supported pretrained control: benchmark methodology and engine optimization.
- NemoChat hybrid: separate export, native inference and cache-state compatibility gate. A successful export does not prove backend support.
- VLM control, later: independently attribute input preprocessing, encoder and decoder costs; do not redesign NemoChat into a VLM merely for coverage.

Record model/tokenizer immutable revision, source/container hashes, driver/runtime, precision/quantization, hardware/topology, parallel layout, cache policy, graph settings, batch policy, request corpus/generation policy and engine feature flags. Store references, not credentials or private prompts. All GPU/remote/publishing work requires appropriate authorization.

## I1 — Workloads, service contract and saved-request analysis

Use workload classes with declared distributions, not a single convenient shape:

- Long input / short output: prefill and TTFT.
- Short input / long output: decode and ITL.
- Mixed lengths: scheduling and head-of-line effects.
- Shared prefixes versus unique inputs: explicitly controlled cache reuse.
- Sustained arrivals and bursts: queueing, tail latency and failure behavior.
- Long contexts / concurrency: cache capacity and admission pressure.

Freeze request IDs, actual token counts and arrival schedule for A/B. Distinguish closed-loop concurrency tests from open-loop arrival tests. Report target arrivals versus actual sends, client saturation, retry/backpressure policy, warmup and drain windows. A slow client or closed-loop self-throttling is not demonstrated server capacity.

Client monotonic timing must distinguish scheduled arrival, actual submission, first output and terminal completion where available. Queueing/server/network attribution needs additional consistent instrumentation; the first analyzer only reports metrics justified by its actual schema. Standard streaming chunks can contain multiple tokens: chunk arrival gaps are not true token ITL. Mark token timing unavailable unless actual per-token events are collected. Missing instrumentation is not zero latency.

Required reports: successful/errors/timeouts/cancellations, request counts, TTFT/E2E distributions with sample counts, ITL only with token-level coverage, throughput and explicit-window SLO goodput. Define percentile convention and whether ITL SLO applies per request or per gap. Declare no-output requests and one-token requests separately. P99 from a small sample is descriptive, not a reliable production-tail guarantee. Failures remain in denominators; quality-gated goodput additionally requires real quality evidence.

CPU acceptance: strict local schema, finite/monotonic timing, consistent counts/events, duplicate refusal, synthetic opt-in, no-overwrite output, actual CLI tests. GPU acceptance: instrument a real engine, replay production-representative workloads and validate timing against independent observations.

## I2 — Operator roofline and speed-of-light estimates

Use explicit operator FLOPs and memory traffic with user-supplied effective compute and bandwidth ceilings. Record where every quantity came from. The simple bound is `max(FLOPs / compute_rate, DRAM_bytes / bandwidth_rate)`; arithmetic intensity is `FLOPs / DRAM_bytes`. This model does not sum independent kernels into a full-service prediction.

Separate HBM traffic from L2 traffic; do not combine bytes at one memory level with bandwidth at another. Precision/Tensor Core eligibility, shape efficiency, cache reuse, launch cost, synchronization and communication affect achievable performance. Occupancy is diagnostic, not an objective to maximize indiscriminately. Peak datasheet ceilings and calibrated microbenchmark ceilings are distinct inputs.

Missing ceilings mean measurement required. If observed operator time is below the bound, report inconsistent assumptions rather than impossible utilization or guaranteed speedup. Model headroom is not achieved speedup and does not establish an end-to-end opportunity. No hardcoded B300 bandwidth/compute claims.

First research question: compare actual prefill/decode expert token distributions and GEMM shapes, routing/launch overhead and memory traffic in a supported MoE path; then assess whether the custom latent hybrid exhibits the same behavior.

CPU acceptance: arithmetic/validation and provenance tests. GPU acceptance: measured operator workload and Nsight Compute counters consistent with the model; explain discrepancies.

## I3 — Trace-to-code optimization

Follow the [Nsight/Codex guide](NSIGHT_CODEX_GUIDE.md): Systems for timeline/host/communication, Compute for shortlisted kernels, PyTorch Profiler where helpful and custom NVTX/counters.

Find one real engine critical-path bottleneck, reproduce it in isolation, and implement one readable optional change. Candidate categories are pointwise fusion, routing transforms, avoidable copies/casts, host tensor preparation or graph-capture blockers. Check existing fused implementations first; the educational pretraining SwiGLU example is not automatically an inference bottleneck.

Validate output/logits, real shapes/layouts, prefill/decode, cache continuation, graph replay and supported quantized paths. Preserve unsupported-input fallback/rejection. Inference requires no gradient implementation unless sharing the kernel with training, but its serving/cache contracts are broader than an isolated forward pass. Profile before/after with equivalent capture scope, then measure final service performance with profilers disabled.

Acceptance: source mapping, hypothesis and alternatives, working patch, correctness, isolated result, full-serving TTFT/ITL/P99/goodput result and adoption decision. Operator-only gains or service regressions must be reported.

## I4 — Scheduling, cache and CUDA Graph trade-offs

Hold workload, quality and resources fixed; change one dimension at a time. Inspect supported batch token budget/chunked prefill, admission, cache memory allocation and graph capture scope. Report a latency/throughput/goodput Pareto frontier, not one best tokens/s number. Include queueing and burst behavior.

NemoChat hybrid cache capacity includes attention KV and Mamba recurrent state, plus engine buffers and graph memory. Validate per-request isolation, reset, reuse, eviction and cached versus stateless behavior. General attention KV formulas do not fully model hybrid state. Graph memory can compete with cache capacity; validate this rather than assuming graphs improve every workload.

Acceptance: workload/SLO-specific configuration recommendations and reproducible cache correctness tests.

## I5 — Precision and quality

Treat weight, activation and KV quantization as separate controls. Use engine-supported formats/kernels and calibration where required. Fix evaluation inputs/protocol and specify task-level quality thresholds before viewing results; logits tolerances alone do not certify task quality. Compare memory, latency, goodput and quality. Lower bit width is not necessarily faster for a particular shape/backend.

Speculative decoding follows baseline stabilization: include draft/verification cost, acceptance distribution, output correctness policy and workload dependence. Do not present acceptance rate alone as a gain.

Acceptance: measured accuracy/performance Pareto results with explicit unsupported configurations.

## I6 — Topology, distributed execution and VLM extension

Where model fit permits, compare TP versus replicas under equal total resources and the same SLO. Inspect collectives, topology and rank imbalance. Single-node NVLink is scale-up evidence, not inter-node RDMA evidence. Multi-node and disaggregated prefill/decode require real request-level state/KV transfer and compatible backend support before claims. They are later optional gates, not assumed capabilities.

A later supported VLM control must preserve image resolution/count and prompt/output distributions, and separate preprocessing, encoder and decoder costs. Record any preprocessing caches and quality effects.

## I7 — Canonical records, regression gates and upstream contribution

Keep immutable source/config/input identities, complete raw request records (or privacy-safe equivalents), measurement-window definitions, resource allocation, warmup, repeats, failed requests, quality artifacts and trace handles. Store new run directories under ignored `results/inference_perf/`; never overwrite existing reports.

Compare only compatible workloads/resources/quality policies. Predeclare tolerated noise and regression thresholds using stable runners and repeated baseline measurements; shared-GPU variability is not proof of code regression. Do not approve an optimization solely from a fixture, profiler summary or a single best run.

Choose one upstream engine for the first contribution. Package a minimal reproduction, narrow patch, correctness tests, benchmark method and raw evidence. Respect company IP/publication policy and ask before commit/push/public PR or disclosure. An accepted contribution is a later outcome, not promised by this track.

## Initial implementation and priority

1. CPU request-record analyzer and synthetic contract tests.
2. CPU single-operator roofline calculator with explicit provenance.
3. Real engine instrumentation/load generation and frozen benchmark corpus (pending).
4. One measured critical-path optimization with service-level A/B (pending).
5. Quality/regression gate and upstream-ready change (pending).

See [tool schemas and runnable local examples](INFERENCE_PERFORMANCE_TOOLS.md) once implementations are verified. Saved-log analysis and the roofline model are scaffolding for this sequence, not substitutes for production-representative inference experience.
