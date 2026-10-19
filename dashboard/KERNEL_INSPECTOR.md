# Kernel, operation and source inspector

[한국어](KERNEL_INSPECTOR.ko.md). The dashboard's primary surface remains a live monitor; this adds a source/evidence inspector, not a replacement Nsight profiler or a native Nsight GUI widget.

## What becomes actionable

Choose a training stage, capture, measured update and host phase. The inspector ranks the **selected update/phase** kernels by recorded CUDA duration and count, shows a human-readable operator family, and connects available saved source evidence. Whole-capture statistics remain a separate view: startup and checkpoint loading can otherwise dominate Fill/AllGather and mislead training optimization.

Investigation priorities must include observed metrics, scope, a candidate explanation and one bounded next experiment. A large cumulative kernel duration is not proof of critical-path contribution, whole-node imbalance or an achieved speedup. Correctness, memory and unprofiled full-update comparison are required before adopting a change.

## Evidence levels

1. **Name-based operation reference:** a symbol can suggest a family (collective, reduction, copy/cast, TopK, permutation, matrix operation, sequence scan). It does not establish tensor shape, exact Python caller, model layer or a causal bottleneck. Unknown/vendor-opaque symbols stay unclassified.
2. **Captured launch association:** runtime correlationId plus recorded process identity links CUDA kernels to CUDA API calls. A same-process API contained in an NVTX phase gives scoped association; it is not exact-thread nesting, a Python stack frame or proof of the forward source for an autograd worker.
3. **Verified host-scope source:** a saved, hash-checked Python file can show which explicit NVTX phase wraps `model(...)`, `loss.backward()/finalize_model_grads(...)`, or `opt.step()`. This verifies the host wrapper's recorded code, not the inner kernel's Python frame.
4. **Verified Bridge host operation:** the existing plugin's immutable file hash, snippet hash, source identity and line bounds can verify the actual explicitly annotated function's saved snippet. These educational sparse-MoE records are not native Nemotron/MCore kernel attribution.
5. **Captured Python call stack / compiler mapping:** exact native source attribution requires actual captured frames or corresponding launch/compiler evidence and source-version identity. Do not substitute a likely library filename, current checkout or generated code.

Source cards show kind, status, path/function, line range, exact saved text and hashes. The API cannot accept arbitrary source paths from a browser. Bounds, relative containment, symlink refusal and hash/identity mismatch rejection apply. Hashes detect drift/tampering but do not authenticate a malicious producer that rewrites both source and manifest.

## Existing native capture boundary

The configured native software trace is an **independent native MCore diagnostic**, not proof that the currently selected Bridge pilot is the same captured execution. The analyzed scope is rank0 measured updates3/4, while all eight ranks have separate execution receipts. This is not whole-node trace coverage.

Parent read-only SQLite inspection found **3,569,543 CUDA runtime records, zero non-null callchainId records, and no exported stack/symbol tables**. Therefore this capture cannot reveal exact native Python caller frames that were never collected. Its observed host phases and saved harness source can still support a useful, correctly labeled investigation path.

The old capture used CPU sampling disabled. Installed Nsight Systems **2026.5.1** help lists Python/CUDA backtrace controls and states their tracing/sampling prerequisites. A future short diagnostic capture needs installed-runtime verification and authorization; it must not automatically attach to or alter a live trainer. Documentation/help support does not establish that a backtrace-enabled capture works on this host, nor that software CUDA tracing will provide equivalent stack coverage. Profiler overhead and permissions must be measured/checked separately.

Relevant read-only readiness files:
- `results/e2e/20261013-1539/profile_retry/version.txt`
- `results/e2e/20261013-1539/profile_retry/profile-help.txt`
- `results/e2e/20261013-1539/profile_retry/source-provenance.json`

Official reference: `https://docs.nvidia.com/nsight-systems/UserGuide/index.html`. The saved installed help takes precedence over generic command examples; no new profiling command is claimed executed by this dashboard feature.

## Data flow and low interference

```
Existing SQLite + verified selected NVTX names
       -> offline read-only correlation/aggregation
       -> bounded inspector.json
       -> cached live dashboard import

Existing Bridge operation source references
       -> saved snapshot/hash/identity verification
       -> bounded exact source cards
```

Heavy analysis is not run on every browser connection or each live collection. Source snapshots are never executed. No new trainer callback, monkeypatch, mandatory writer, GPU workload, capture or collector is part of this extension. The existing plugin remains responsible for opt-in collection/source annotation; the dashboard consumes its evidence.

CLI/schema ownership and precise fields are in `INSPECTOR_CONTRACT.md`; supported arguments are in `capture_analysis.py --help`. Executed analysis/source/UI evidence and current limits are in [inspector verification](INSPECTOR_VERIFICATION.md). The current native adapter deliberately handles the saved rank0 measured updates3/4 and their explicit phase names; other naming/selection schemes need their own adapter. Use fresh outputs and preserve negative evidence; unknown mappings must not be upgraded merely to make an optimization recommendation look decisive.
