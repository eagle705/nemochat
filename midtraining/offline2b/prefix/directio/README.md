# Native-codec O_DIRECT read transport — parent-only integration plan

**No GPU launch is authorized by this package or CPU gate. A BLOCKED CPU gate prohibits integration release.**

## Smallest new generation (not executed)

1. Parent reviews `expectations.json` SHA, the final exact Engine receipt, `shards.jsonl`, counterproof, blocker list, TDD logs, and this source. Require a COMPLETE/clean all-9144 gate and exact Engine exit 0. Do not adjust SHA expectations to make a failed direct read pass.
2. Create a fresh scratch SDK/staged source generation and fresh output/HOME/compiler caches. Preserve continuation03, its old outputs, installed SDK, originals, teacher, BASE weights and all original cache files. The new staged native `utils_logits.py` must remain byte-identical to the reviewed source; the version guard must match its pinned full-source SHA and original opener AST before process-local injection.
3. Copy the reviewed continuation03 **launcher source only** into the new scratch generation, pin/record its original SHA, and patch only that new copy. Change `CACHE` from `/output/cache-overlay` to the original audited local cache directory `/raid/yoon/nemochat/results/midtraining/offline2b-production1920-01/cache`; do not use symlinks or replacement shards. Parent creates a new prefix-manifest from the immutable original-file expectations (not the rejected overlay's timestamps). Preserve original metadata/cache identity and all source-budget values.
4. Before constructing `CachedKDForwardStep`/any native cache reader in each rank, import the actual staged `megatron.bridge.training.cached_kd.utils_logits` module, then call:

```python
from midtraining.offline2b.prefix.directio.integration import activate
receipt = activate(
    native_utils_logits,
    '/raid/yoon/nemochat/results/midtraining/offline1920-parent-01/directio-acceptance-01/expectations.json',
    '1e62f0a17669c511fbd091d5d0c478efc8e8e8a2595a7ca6c6fd23c35f21af2d',
    opt_in=True,
)
```

The runtime source path may be the fresh scratch path: contract matching is content/AST based, not tied to the old scratch pathname. Inject only this module's `open_logit_file`; do not patch builtins or install a global filesystem shim. Existing native reader references use that module's globals. Do not replace the streaming tar, zstd, packed-index unpacker, torch loader, logits loader API, loss function or loss settings.

5. Existing native cached reader forks one DataLoader worker; transport creates direct fds/mappings only while reading, inside that worker. No live mmap/fd is created by constructor/activation. The registered at-fork barrier prevents forking while a direct read owns its mapping. No payload bytes are reread buffered after SHA. Unsupported alignment/direct I/O, short/zero reads, identity changes and SHA failures abort without fallback.
6. Keep original completed BASE model-only load, step 0, optimizer/RNG/cursor reset, full sampler 468736, original WSD horizon 3662 with warmup 37 and decay 366, bounded completed prefix 1143/599261184 positions. Keep original teacher frozen, Top32/ghost-tail/alpha .5 unchanged. The CPU gate does not read or hash checkpoint weights and does not revoke the separate explicit checkpoint-payload SHA waiver.
7. Parent independently owns actual GPU readiness, source-only diff review, new generation creation, native launch, all-rank sample identity/update/finite-gradient/save gates and exact Engine outcome. This plan/API is ready for review, **not a runnable/armed GPU supervisor**. No training continuation is scheduled here.

## CPU import boundary

The pinned image's ordinary `megatron.core` package import eagerly loads Transformer Engine and fails without `libcuda.so.1`. Failed CPU Engine attempt 01 is retained. CPU attempt 02 imports the entire real native `utils_logits.py` and runtime file in a private package, and directly imports the real SDK `msc_utils.py` (no function stubs), avoiding only the irrelevant GPU package initializer. This is actual native reader/decoder execution, not a full GPU training import or validation of the training entrypoint. PyTorch DataLoader is configured worker=1/pin_memory=True and automatically disables pinning when no accelerator is available; it does not fake CUDA.

## Read-path mitigation, not hardware repair

O_DIRECT bypasses the discrepant buffered read path. Neither a correct direct read nor zero GPU ECC counts proves host health or identifies the kernel/storage/hardware cause. No fadvise, drop-cache, filesystem change, reset, reboot, source/input replacement or teacher regeneration is performed. Any new SHA mismatch remains a blocking file, never a reason to rebase expectations silently.
