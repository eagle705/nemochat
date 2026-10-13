# RL rollout bottleneck observability with NeMo Lens

[Korean companion](../ko/RL_ROLLOUT_OBSERVABILITY.md). Runnable local exercise: [rollout lab](../../tutorials/nemo_lens/rollout/README.md). This document is a source-verified integration guide; it is not an executed native NeMo RL/Gym/vLLM training recipe.

## Questions and measurement boundaries

A long rollout can be dominated by model generation, caller admission, tool execution, HTTP connection acquisition, retries, verification/reward, or one slow trajectory. Record these separately where instrumentation exists. Do not label every long HTTP client span as slow tool compute: it includes transport, server work and other client overhead. Client semaphore wait is not vLLM scheduler queue time. Missing telemetry means unknown, not zero.

Concurrent rollout durations must not be summed and called batch wall time. Inspect the last-finishing trajectory and its stages, compare cohort start/completion and overlap, and retain errors/timeouts/cancellations. Parent spans contain children; server tool spans are nested within client tool-call duration. Avoid adding both. A longest stage is a diagnostic candidate, not proof that an optimization will improve RL throughput.

## Native NeMo RL phase view

Inspected `NVIDIA-NeMo/RL@67fc59b2ccc6db48dc873eefa86d64506ca38126`:

- [Configuration](https://github.com/NVIDIA-NeMo/RL/blob/67fc59b2ccc6db48dc873eefa86d64506ca38126/docs/observability/configuration.md)
- [Span groups and coverage](https://github.com/NVIDIA-NeMo/RL/blob/67fc59b2ccc6db48dc873eefa86d64506ca38126/docs/observability/span-groups.md)
- [vLLM tracing layers](https://github.com/NVIDIA-NeMo/RL/blob/67fc59b2ccc6db48dc873eefa86d64506ca38126/docs/observability/vllm-tracing.md)

Relevant groups include rollout, generation, logprob, reward, advantage, policy_update, data_plane and efficiency. Short diagnostic captures may opt into `per_prompt` in addition to `per_step`. This is a **partial telemetry configuration**, not a complete run command:

```yaml
telemetry:
  enabled: true
  span_groups: per_step,per_prompt
```

Read the full installed-version configuration and exporter requirements before use. Per-prompt span volume scales with prompts; do not enable the finest tracing for every production step by default. The `job` root can produce an ever-growing trace; removing it can also change propagation to async rollout collectors. Treat this as an explicit scope trade-off, not a universally safe tuning switch.

Inspect native spans such as `rl.<algo>.generation`, `rl.gym.run_rollouts`, `rl.<algo>.reward_calculation`, `rl.<algo>.policy_training`, `rl.idle.buffer_starvation` and `rl.idle.refit_bubble` only where that execution path actually emits them. The native docs identify coverage gaps: async `generate_async` is not the same as traced synchronous `generate`. Confirm coverage on the chosen algorithm/backend/version.

vLLM native engine/request tracing is independent from the RL-side span around the generation call. Native vLLM traces are opt-in and potentially high volume. Offline `LLM.generate()` cannot pass per-request trace context in the inspected integration; native spans correlate through shared run/resource identity, not a guaranteed parent-child edge to the RL rollout. Do not fabricate queue/prefill/decode splits when only driver-level duration is available.

## NeMo Gym: cross-process model, tool and verify view

Inspected `NVIDIA-NeMo/Gym@8fddfb519e9473c6e51b8e8bf89e02aa7e5fa0c1`:

- [Span groups](https://github.com/NVIDIA-NeMo/Gym/blob/8fddfb519e9473c6e51b8e8bf89e02aa7e5fa0c1/fern/versions/latest/pages/observability/span-groups.mdx)
- [Distributed tracing](https://github.com/NVIDIA-NeMo/Gym/blob/8fddfb519e9473c6e51b8e8bf89e02aa7e5fa0c1/fern/versions/latest/pages/observability/distributed-tracing.mdx)
- [Metrics](https://github.com/NVIDIA-NeMo/Gym/blob/8fddfb519e9473c6e51b8e8bf89e02aa7e5fa0c1/fern/versions/latest/pages/observability/metrics.mdx)
- [Connection-pool instrumentation](https://github.com/NVIDIA-NeMo/Gym/blob/8fddfb519e9473c6e51b8e8bf89e02aa7e5fa0c1/nemo_gym/telemetry/connection_pool.py)

A partial Gym diagnostic selection:

```yaml
telemetry:
  enabled: true
  span_groups: per_rollout
```

This selects server/http_client/rollout/verify/agent/model_call and deliberately omits job/startup. There is **no `tool_call` group** in the inspected Gym. Resources tools appear as server routes, e.g. `POST /get_weather`, nested beneath the outbound HTTP client context. Keep both `server` and `http_client` active: one reads and the other writes `traceparent`; disabling either breaks cross-process joining. Model calls appear through `gym.model.*`, rollouts through `gym.rollout`, and verification through `gym.verify`.

When Gym and RL share a process, only one Lens telemetry setup is allowed. If RL initializes it first, Gym reuses RL's provider/configuration. Remote Gym processes still require their own configuration, exporter and identity. Shared group names/presets may activate spans from both libraries; do not initialize a second independent provider in the same process.

### Distinguish tool service from client connection wait

The native metric `gym.http.connection_pool.queue_duration_ms` measures queued aiohttp connection acquisition, not model inference or tool compute. Its outcome is `ok` or `abandoned`, and pool constraint can be `total`, `per_host` or `unknown`. Immediate acquisition has no queue histogram sample. Pair it with `gym.http.connection_pool.connect_total` over the same destination/window to estimate how often attempts queued. Retries, redirects and reconnects may create separate connection attempts; those counts are not the number of semantic tool calls.

Also inspect `http.server.active_requests`, route duration/status, `gym.rollout.duration_ms` and `gym.verify.duration_ms`. `gym.verify.success_rate` measures successful completion of the verification call, **not answer correctness**. Reward and benchmark quality remain experiment telemetry, not a replacement for these operational metrics.

## Optional trajectory/tool evidence, separate from OTel spans

[Rollout evidence guide](https://github.com/NVIDIA-NeMo/Gym/blob/8fddfb519e9473c6e51b8e8bf89e02aa7e5fa0c1/fern/versions/latest/pages/observability/rollout-evidence.mdx) and [simple agent execution](https://github.com/NVIDIA-NeMo/Gym/blob/8fddfb519e9473c6e51b8e8bf89e02aa7e5fa0c1/responses_api_agents/simple_agent/app.py) document structured model-call/agent/tool evidence. Capture is an independent, privacy-sensitive choice; OTel spans alone do not imply these artifacts exist.

The inspected simple agent records tool `duration_ms` with a monotonic timer, `timing_source="executor"`, invocation/tool-call identity and completion/error status when trajectory collection is enabled. Join by explicit IDs, not timestamp proximity. Native evidence may contain prompts, tool arguments/results and conversations: obtain authorization and use private storage before enabling it. Keep gaps for unsupported/unmatched/ambiguous evidence; do not invent semantic turns or missing timings.

## Local tutorial versus native acceptance

The local rollout exercise intentionally demonstrates a controlled workflow: actual local generation/HTTP tools and real Lens spans, with any injected delay explicitly marked. It is not learned tool selection, native distributed RL, vLLM GPU latency or a production bottleneck discovery. Use it to learn how traces change, then reproduce on an approved native workload.

Native acceptance requires: chosen RL/Gym/vLLM version and algorithm, service/rank/run identity, preserved context across actual HTTP/Ray/process boundaries, exporter coverage and sampling, cohort/window policy, token counts from real usage, error/timeout/retry handling, throughput/quality before-and-after and instrumentation overhead. If GPU compute is suspected, use a short Nsight diagnostic capture rather than claiming OTel explains individual kernels.
