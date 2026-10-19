# Stage-based source layout (v1)

Only the four stage code trees (`pretraining/`, `midtraining/`, `sft/`, `rl/`) are versioned source roots. The runtime payload roots `datasets/` and `checkpoints/` are Git-ignored and contain zero indexed source files; they may be absent in a clean checkout or `git archive`. All six roots exist on the current local host, but that is not a fresh-checkout guarantee. Create payload directories only when a separately authorized operation needs them; do not force-add payloads or placeholder files.

Canonical stage roots are `pretraining/`, `midtraining/`, `sft/`, `rl/`. `datasets/` holds data only and `checkpoints/` weights/state only; no source was hidden under either ignored payload root. Shared `src/nemochat`, cross-stage `scripts`, `configs`, `tests`, `docs`, dashboard/tutorials/plugins remain shared.

## Routing

- `scripts/pretraining` → `pretraining/entrypoints`; BASE publication from `scripts/posttraining` → `pretraining/publication`.
- `midtraining` top-level corpus/download/recovery → `midtraining/data`, launch/native code → `midtraining/training`, YAML → `midtraining/configs`, technical reports → `midtraining/docs`.
- `midtraining/offline2b` → `midtraining/training/offline_kd`; three comparison families → `midtraining/evaluation`. Historical variants remain explicitly nested and indexed by `midtraining/archive/README.md` rather than masquerading as current trainers.
- `midtraining/sft_after_kd` is removed entirely. Preparation → `sft/training`, input proof → `sft/data/acceptance`, full-state continuation → `sft/resume/runtime`, diagnostic1248 → `sft/evaluation/benchmark1248`, failed/superseded attempts → `sft/archive`.
- SFT publication → `sft/publication/hf_sft1200`; reusable former posttraining SFT scripts → `sft/{data,training}`, configuration → `sft/configs`.
- Gym importer/config → `rl/{evaluation,configs}`; this is evaluation infrastructure, **not an implemented RL training loop**.

## Evidence and preserved payloads

Machine-readable [migration manifest](../stage-layout-migration-v1.json) maps original Git blobs and paths at `de2fda2e13e9f8a007249e5756d39e27c30ef6c8` to new paths. Exact historical source archive is locally retained at `results/maintenance/stage-layout-20261019-01/historical-source-snapshot.tar`. Git history also preserves those bytes. Existing receipts/journals/patch evidence, dataset/checkpoint payloads, expected hashes, token IDs, masks and numerical algorithms were not rebased. Edited source is **not source-freeze identical and not newly GPU-qualified**. Old receipt source paths resolve through the manifest, not compatibility symlinks.

Accepted native SFT1248 remains at `results/midtraining/sft-continue1500-02/main/checkpoint/iter_0001248`;905 remains at `results/midtraining/sft-after-kd-runtime05/main/checkpoint/iter_0000905`. Do not move/copy/rehash these large native payloads as part of source cleanup. Existing HF exports remain in `checkpoints/exports`. Future new producers should explicitly choose `datasets/<stage>`, `checkpoints/<stage>`, `results/<stage>` and fresh names. Run-bound historical recipes retain artifact/evidence defaults to avoid silently reinterpreting old proofs; they are not fresh launch authorization.

Host CPU extras use Transformers4; historical native/export runtime is5.12.1. CPU tests, offline help and clean-archive imports do not validate native scheduler/resume/CUDA behavior. Exact historical opt-in registry remains88functions/149cases; no broad skip is introduced for relocation. Known filesystem baseline failure in `test_hf_checkpoint.py` remains separately reported.

Offline-KD preparation/materialization is owned by `midtraining/data/offline_kd`, recipes by `midtraining/configs/offline_kd`, guides by `midtraining/docs/offline_kd`, optional native-cache I/O by `midtraining/cache/directio`. Native/LM-only/recovery implementations remain in the training family.
