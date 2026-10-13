# Corpus Planning, Checkpoint Persistence, and Experiment Monitoring

Status: reviewed CPU/offline source scope; see [whole-repository final review](WHOLE_REPOSITORY_REVIEW.md). Remote transport and native GPU acceptance are separate gates. No Hugging Face remote upload or W&B online run has been authorized/executed by this step.

## User-selected additions

1. Assume the corpus is too large to fully download/tokenize for planning. Use bounded representative sampling with the target tokenizer, explicit source populations and honest uncertainty/coverage limitations.
2. Persist intermediate checkpoints on Hugging Face and restore them across week-long B300 rentals.
3. Register and monitor each pipeline stage with W&B, including shared lineage and stage-specific metrics.

## Execution order and scope

- **Before rental:** define source strata and sampling design, estimate corpus tokens, plan quota/time scenarios, prepare the required bounded subset and exact token manifest.
- **Rental startup:** inspect hardware/topology/SDK lock, download a trusted pinned checkpoint bundle if resuming, verify inventory and training state, run a small resume gate before spending the full rental.
- **During training:** log actual counters/timing; checkpoint locally at completed update boundaries; enqueue immutable checkpoint upload outside the critical update path only after snapshot completion.
- **Before rental end:** reserve explicit time for final checkpoint/upload/read-back and backup. A queued upload or returned commit handle is not yet a verified recoverable artifact.
- **Between rentals:** preserve checkpoint/tokenizer/environment manifests and frozen evaluation inputs; CPU WSM can run separately where compatible. Track outputs as evaluation artifacts, never silently replace raw training state.

## Large-corpus estimates

Token counts depend on tokenizer revision and preprocessing, not the dataset card's original tokenizer or compressed bytes. Measure per-source document length, UTF-8 bytes, text tokens and boundary tokens. Distinguish raw estimates from post-filter/dedup/split estimates.

A uniform document sample over a known population supports a document-mean estimator. A hand-selected shard or prefix sample does not establish whole-source representativeness. Clustered samples need a design-aware interval or an explicit statement that uncertainty cannot be inferred from independent-document formulas. Do not present a tight CI based only on documents from one chosen shard as a whole-corpus uncertainty interval.

Population counts, weights, seed, sample selection, source/shard coverage, tokenizer/config hashes, document limits, preprocessing and declared scope belong in the estimate artifact. Preparation produces exact counts for the resulting bounded subset; it does not retroactively make the population estimate exact.

## Budget/ETA contracts

Use measured pilot throughput or recorded step timings from the actual model/length/precision/parallel layout. GBS alone determines neither runtime nor cost. GBS = microbatch × actual DP size × accumulation for the corresponding fixed-batch setup; model-parallel ranks must not multiply token counts.

Keep prepared corpus tokens, processed tokens and loss-bearing tokens separate. Report repeated exposure separately from corpus availability. Estimates cannot replace runtime counters.

Use either end-to-end measured effective throughput or training-only throughput plus overhead, not both reductions for the same losses. Price units must distinguish whole-node hourly price from per-GPU hourly price. Whole-week billing, storage/egress/CPU services and minimum charges need explicit handling. Sequence-length/stage changes require their own throughput; do not project a 4K pilot to 64K without measurement.

## Checkpoint artifact kinds

### Resumable training bundle

Contains raw model/DCP files, optimizer, scheduler configuration/state, RNG, data cursor and token counters, tokenizer/config identity, environment/source manifest and integrity inventory. Native Megatron formats remain native; hosting them in a model repository does not make them `AutoModel.from_pretrained` compatible. The bundle declares its stage, run, parent checkpoint and completed update/token boundary.

### Model-only export / WSM evaluation artifact

Contains export-compatible model/tokenizer files, origin/merge provenance and an integrity inventory. This kind cannot be used as proof of resumable optimizer state. Any HF-compatible format must be actually validated; a successful upload is not export/inference verification.

Use immutable artifact paths such as `checkpoints/<stage>/<run>/<checkpoint-id>/` in a private repo. The returned remote commit SHA and file hashes become persistent handles. Never put optimizer snapshots or auth credentials into a public export by accident. Publication/license review is distinct from private operational storage.

Local packaging, mocked Hub calls and local pack/restore tests are different evidence levels from an actual remote round trip. Real upload success requires read-back of the committed inventory/files; downloading requires an immutable revision, bounded artifact path, fresh output path and post-download validation. Treat deserializable training bundles as trusted inputs, not safe simply because they came from HF.

## W&B stage matrix

| Stage / run type | Metrics and artifacts | Current integration boundary |
|---|---|---|
| Data preparation | Source estimate scope, accepted/rejected docs/tokens, EOS, tokenizer identity, data manifest | Existing preparation entry point can be instrumented |
| Pretraining | LM CE, actual LR/phase, loss/processed tokens, step time, checkpoint IDs; frozen eval diagnostics | Tiny-HF control first; Bridge GPU path remains separate |
| Mid-training | Data blend/exposure, held-out retention, tokens/throughput, parent checkpoint | Stage contract; actual training entry point not yet implemented |
| Long-context | Per-length tokens/timing, long/short updates, retention and measured retrieval results | Schedule-only tool must use planning run type, not training |
| SFT | Assistant loss-bearing tokens, truncation/masking, generation evaluation | Stage contract; execution not yet implemented |
| RLVR | Prompts/rollouts/update tokens separately, verifier correctness, reward/KL/entropy where measured | Stage contract; execution not yet implemented |
| MOPD | Teacher scoring/generated/update tokens separately, actual teacher/student cost and quality | Stage contract; execution not yet implemented |
| Evaluation / WSM | Raw vs merged task protocols, prediction scope/failures, merge inputs/coefficients, output IDs | Importer/fixture scope must be labeled; no target benchmark inferred |

A shared project/group links the overall experiment. Each stage/run has an explicit run ID, stage label, parent artifact, tokenizer/data/config/source revisions and mode. Online resume semantics require explicit policy; offline segments sharing a run ID do not by themselves prove server-side history has resumed.

Default to disabled or explicitly offline tracking. Online calls require the user's entity/project, credentials and affirmative execution intent. Never serialize keys into run config or artifact manifests. Sanitize nested secrets without removing ordinary tokenizer/token-count metadata. Log errors as failures and finish with the correct status. Logging must not perturb optimizer/RNG trajectory.

For production distribution, rank-zero coordination and broadcasted run identity are required; CPU single-process tests do not establish distributed behavior. Reference large checkpoint artifacts by immutable HF handles where possible rather than uploading all weights twice.

## Acceptance gates

- Representative sampling/seed/population and mathematical planner tests; real tokenizer bounded sample CLI evidence.
- Local trusted bundle package/restore with hash/identity checks and malformed/unsafe-path/failure tests.
- Real W&B offline run with actual tiny-HF updates; verify resume remains exact and no online call is implied.
- Independent review of new code and docs, parent final suite rerun, explicit optional-dependency installation and correct ignored output paths.
- Remote HF and W&B online gates remain separately pending until destination/credentials/budget are confirmed.
