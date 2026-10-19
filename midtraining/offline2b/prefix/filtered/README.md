# Filtered fresh-zero native prefix

Excludes whole original global iterations **86, 98, 121, 122, 126** on all eight ranks (40 tar shards, including 34 otherwise accepted shards). The native full GPTDataset is built with `[468736,120,8]` before a minimal sample-index wrapper selects 1,138 global batches. The physical/logical map and its inverse are explicit in `prefix-manifest.json`; checkpoint transactions carry the map identity and next physical/logical cursor, including final sentinel 1143.

Native tar/zstd/torch decoding, original `_meta.json`, full source cache identity, Adam/DDP/clipping/router policy and 3662/37/366 WSD schedule remain unchanged. The task-local reader subclass filters discovered URLs and maps decoded original iteration IDs into logical IDs; original cache files are never renamed/repacked. Every accepted source read uses O_DIRECT and the historical per-shard SHA, with no retry or buffered fallback. Every training microbatch is compared to the original teacher's physical-index digest. New SHA failures capture observed SHA/length and abort.

Fresh model-only BASE19074 load resets optimizer/RNG/cursor. Native synchronous full-state saves occur every 64 logical calls and at final 1138. Checkpoint payload hashing remains explicitly waived. An initial LR=0 optimizer call is not positive-LR learning. Historical six read rejections do not establish permanent file corruption.

The independent user-systemd service holds the shared all-eight-GPU lock. Repo/input mounts are read-only; only this run's output and new scratch are writable. HOME/Inductor/Triton caches start empty. Source freezes, exact Engine readbacks, all-rank receipts and charts belong only to this new generation.
