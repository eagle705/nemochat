# CPU-first long-context implementation

## Scope and non-claims

`src/nemochat/long_context/` contains small Python functions returning ordinary dictionaries/lists. The schedule CLI **does not train a model**, change attention/RoPE, provision GPUs, or integrate NVIDIA NeMo/Megatron Bridge. Neither generated fixtures nor mocked test responses establish 64K capability or retrieval accuracy. A real endpoint run is required for measurements. Even successful NIAH is not evidence of broad long-context competence.

## Fixed-token schedule

```bash
.venv/bin/python scripts/lc_schedule.py \
  --long-only-steps 2 --alternating-steps 4 \
  --effective-tokens 131072 --micro-batch-size 1 --world-size 1 \
  --cursor /tmp/lc-cursor.json --output /tmp/lc-steps.jsonl --count 3
# Repeat the identical command to emit the next three steps.
```

64K means **65,536** tokens; 4K means **4,096**. The plan starts with the requested number of long-only optimizer-step records. The alternating phase begins with long, then short, then long, etc. Every step satisfies:

```
effective_tokens = sequence_length * micro_batch_size * world_size * gradient_accumulation_steps
```

The default six lengths are 65536, 65536, 65536, 4096, 65536, 4096. Accumulation is 2 for long and 32 for short at the default budget. The short phase therefore has more sequences, not more effective tokens. Noninteger, zero/negative, empty-plan, indivisible, or too-small batch/token configurations fail rather than being rounded. `effective_sequences` is the global number of complete padded sequences per step; this does not count only nonpadding/loss-masked tokens.

APIs: `make_plan(...)`, `step_at(plan, step)`, `load_cursor(path, plan)`, `advance_cursor(cursor, count)` -> `(records, new_cursor)`, and `save_cursor(path, cursor)`. JSON cursor stores the complete versioned plan and `next_step`. Resumption requires identical parameters and rejects corrupt/out-of-range state. The CLI writes its **current batch only**, replacing output, then checkpoints the cursor. Both files use atomic replacement individually, **not a multi-file transaction**. One writer per cursor is required; concurrent processes are not locked. A crash between output and cursor writes can replay that batch. A consuming trainer must checkpoint its own acknowledged step and handle replay; the CLI advancing is not proof that training completed.

## NIAH evidence publication

NIAH generate/run/score/plot destinations must be **fresh and nonoverlapping** (including CSV, PNG, SVG and metadata). Existing files and broken symlinks are rejected before tokenizer loading or completion requests; use a new prefix for a rerun. Exclusive reservations and no-clobber publication preserve competing evidence. Complete fixtures and unique IDs are validated before the first completion request. Schedule/cursor replacement above is intentionally different and unchanged. Reservation sidecars persist on success and failure as one-attempt tombstones: every retry uses new destinations. Published partial artifacts are retained after a later failure, never rollback-unlinked. Private staging names must not be modified by other writers; cleanup attempts all private unlinks and FD closes independently. Process death or cleanup failure may leave private staging files. Bundles are not atomically visible as one transaction; check exit status and completeness, and do not infer success from one file. Incremental completion recovery is not implemented.

## Tokenizer-measured synthetic fixtures

Use the exact HF tokenizer served by the model, preferably a local snapshot or pinned revision:

```bash
.venv/bin/python scripts/niah.py generate \
  --tokenizer /path/to/tokenizer --local-files-only \
  --config configs/eval/niah_smoke.json --output /tmp/niah-fixtures.jsonl
# Optional: --haystack /path/to/corpus.txt
# Random depths: config values override CLI flags. Edit random_depths in the
# supplied config (currently 0), or omit --config and use --random-depths 3.
```

Config values override corresponding CLI values. The smoke config generates 18 synthetic prompts (two budgets, three depths, three samples), **not predictions**. The API `generate_fixtures(tokenizer, context_lengths, depths, ...)` takes the actual tokenizer object. No tokenizer/model weights are silently substituted. Remote tokenizer loading disables `trust_remote_code`; offline runs can force `--local-files-only`.

Assembly is deliberately defined, rather than estimating tokens from words/characters:

1. Encode the haystack with `add_special_tokens=False`.
2. Repeat/slice those tokens, split at `round(filler_count * depth)`, and decode each slice with cleanup disabled.
3. Insert a complete textual needle containing a seeded random 128-bit hexadecimal secret between the decoded slices, surrounded by fixed retrieval instructions.
4. Encode the **whole final prompt**, including instructions and special tokens, with `add_special_tokens=True`. Shrink filler until `prompt_tokens + generation_tokens <= context_length`. Exact budget filling is not promised; check the recorded actual length.
5. Require the complete needle to remain in the assembled prompt and the decoded final token sequence. Tokenizers whose normalization destroys it are rejected. This is token-slice-based haystack assembly, not a guarantee that decoding/encoding every filler slice preserves its original IDs. If byte-token cuts cross characters, final encode measurement still governs the budget.

Depth is a requested proportion of **haystack token slices**, not a proportion of the entire prompt, bytes, or characters. `needle_offset_haystack_tokens` and `actual_haystack_depth` expose rounding. Zero haystack has null actual depth. Explicit depths and random depths (via `random_depths`) are frozen by seed; each sample gets a different reproducible secret. Changing grid/order/random-depth count changes the random sequence. Repeated stock prose is intentionally a minimal smoke corpus, not a realistic benchmark. Save generated fixtures to freeze the entire experiment.

Rows include prompt, expected secret, needle, full token count, SHA-256 of final token IDs, corpus hash, seed, depth, generation reservation, tokenizer name, and CLI revision. `validate_fixture` re-encodes and checks hash/length, full budget, and intact needle before any request. Name/revision alone is not sufficient tokenizer validation.

## Real prediction runner

```bash
# Export OPENAI_API_KEY using your normal secret manager if the server needs it.
.venv/bin/python scripts/niah.py run \
  --tokenizer /path/to/tokenizer --local-files-only \
  --input /tmp/niah-fixtures.jsonl --output /tmp/niah-predictions.jsonl \
  --endpoint http://localhost:8000/v1/completions --model actual-served-model \
  --timeout 60 --api-key-env OPENAI_API_KEY
```

The URL must be the complete **plain-text `/v1/completions` endpoint**, not a chat endpoint. This avoids unaccounted chat-template overhead. The runner uses `urllib.request`, sends the real prompt with `max_tokens`, temperature 0, and stream false, and makes an actual request for each fixture. Keys come only from the named environment variable; embedded URL credentials are rejected. Temperature 0 does not guarantee deterministic server execution. Authenticated non-loopback urllib requests **require HTTPS before any network call**. Plain HTTP authentication is permitted only for literal loopback IP addresses or `localhost`, preserving the local smoke server. URL credentials and invalid ports are rejected; redirects cannot forward the bearer key. Authenticated loopback HTTP requests disable environment proxies (including when `HTTP_PROXY` is set and `NO_PROXY` is empty), so plaintext bearer keys cannot escape via a proxy. Other urllib requests retain normal proxy configuration. Without a key, HTTP is allowed, but remote services should still use HTTPS because prompts may be confidential. The optional hosted SDK client owns its own transport configuration; this URL guard does not inspect that client. Do not send confidential haystacks to untrusted providers.

`run_predictions(..., client=client)` optionally accepts a hosted OpenAI-compatible SDK client implementing `client.completions.create(..., timeout=...)`; the caller installs/configures that SDK. Chat-only hosted models are intentionally unsupported by this protocol. There is no dummy fallback, echoed answer, or fabricated prediction. HTTP failures, malformed JSON/choices/text, server error objects, network failures, and timeouts produce explicit non-`ok` records. HTTP 400/404/413/422/501 are labeled request `unsupported` (not proof the model lacks capacity); other HTTP failures are errors. Error records do not include potentially sensitive server bodies. Successful HTTP JSON bodies are capped at 1,048,576 bytes (`MAX_RESPONSE_BYTES`): read at most cap + 1 bytes, reject overflow before parsing, and record `status=error` with no prediction or scored sample. HTTP error bodies are not read and retain their HTTP status classification. This cap applies to urllib, not caller-configured SDK clients.

Only string completion responses whose server `usage.prompt_tokens` exactly matches the local final prompt length become `status=ok`. Missing usage is `tokenization_unverified`; mismatching usage is `tokenization_mismatch`. Both preserve returned text for inspection but are excluded from accuracy. **Matching counts do not prove matching server token IDs**: operators must verify the tokenizer/BOS policy. A stricter server may intentionally use different BOS behavior, and then measurements remain excluded until the tokenizer/protocol is aligned. Budget is locally validated; the server's true context capacity is not assumed. Requests are sequential and use a per-request timeout; there is no retry/resume engine or end-to-end experiment deadline. The runner collects results then atomically writes output, so an interruption can lose that run's in-memory results.

## Honest scoring and plots

```bash
.venv/bin/python scripts/niah.py score --input /tmp/niah-predictions.jsonl \
  --output /tmp/niah-scores.jsonl --csv /tmp/niah-scores.csv
.venv/bin/python scripts/niah.py plot --input /tmp/niah-predictions.jsonl \
  --output-prefix /tmp/niah-measured
# Plotting fixture JSONL instead visibly says SYNTHETIC FIXTURE — NOT MODEL PERFORMANCE.
```

`exact_match` compares `" ".join(text.split())` on both sides: **only whitespace normalization**. It does not lowercase, remove punctuation, accept substring matches, or strip explanatory prose. `"The code is <secret>"` is wrong even when the secret appears inside it. A valid but truncated completion is scored as returned, normally wrong if incomplete; no recovery is invented.

`score_predictions` groups context/depth cells, rejecting duplicate IDs and mixed models/endpoints or mixed `data_kind`. Every cell reports total requests, valid count, successes, excluded count, status histogram, accuracy and **95% Wilson binomial interval**. Non-`ok` requests never enter the denominator; no valid response means null accuracy/interval, **not zero**. Raw fixture inputs contain no predictions and score to null. Synthetic predictions can be plotted if explicitly marked `data_kind=fixture`, but must not be labeled measurements. Small samples make intervals wide; exclusion-heavy cells can be severely selection-biased. Inspect coverage as well as accuracy.

PNG/SVG heatmaps read the supplied JSONL prediction records, use a fixed **0..1** scale, annotate valid/total sample counts and intervals, and visibly label timeout versus unsupported versus tokenization/error cells with distinct colors/text. Mixed exclusions are enumerated; unobserved cells say not sampled. Measured plots name the model, while fixture plots have a synthetic title. The accompanying `.metadata.json` stores `data_kind`, measured/synthetic flags, model/endpoint, scoring definition, scale, and cell counts/uncertainty. JSONL/CSV/PNG/SVG/metadata writers use temporary files in the destination directory and `os.replace`; multi-artifact outputs are not transactional. These records/metadata are user-provided provenance, not a cryptographic attestation that a server was called.

## Loopback CPU smoke server

`scripts/serve_cpu.py` remains a serial, loopback-only control server, not a production service. Each accepted socket has a 5-second inactivity timeout (`REQUEST_TIMEOUT_SECONDS`) for headers, body reads and response writes. Stalled body reads return JSON HTTP 408 when the peer remains writable; incomplete headers are closed by the standard HTTP handler. Timeouts and disconnected peers are handled without request tracebacks, allowing subsequent `/health` and completion requests. This is not an overall request deadline or a model-generation timeout; a trickling peer or slow generation can still occupy the serial server.

## Verification

```bash
.venv/bin/python -m pytest tests/test_long_context.py tests/test_cpu_server.py -q
```

Tests use a tiny character tokenizer, a locally built HF byte-level tokenizer, and mocked network replies. They exercise fixture budgets/round-trips/frozen seeds, exact token budget/resume, actual HTTP request construction, timeout/unsupported/error exclusion, strict matching, Wilson intervals, CLI generation/scoring, and PNG/SVG labels. **They are correctness tests, not performance measurements.** No external model run or training result is claimed.
