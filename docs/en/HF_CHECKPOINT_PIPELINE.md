# Hugging Face checkpoint pipeline for weekly B300 rentals

## Contract and evidence

The implementation transports **frozen** checkpoints; it does not create a
Megatron checkpoint, convert DCP to Transformers, or prove training resume.
`python scripts/hf_checkpoint.py --help` is usable without torch or Hub installed.
Hub imports are lazy. Remote calls require an explicit environment `HF_TOKEN`;
there is no login, token persistence, token argument, or token output. Do not use
shell tracing/SDK debug logging. The CLI deliberately suppresses SDK exceptions
because they may expose credentials. Inspect programmatic exceptions only in a
secure environment. Keep secrets out of source files and environment manifests.

| Kind | Contents | Intended consumer |
|---|---|---|
| `torch_dist_training` | Native DCP metadata and **every** model/optimizer shard; scheduler, rank RNG state, cursor, tokenizer, training manifests, environment | Trusted upstream training resume, with matching software/topology/data |
| `hf_model` | Separately produced model-only `config.json`, safetensors and optional tokenizer assets; every shard/index hashed | HF evaluation/export; compatibility requires an actual loader test |

A training bundle is **not** `from_pretrained` compatible. Never copy an old
optimizer-bearing checkpoint wholesale into a model export directory. Model-only
packages reject `.pt`, `.pth`, `.bin`, `.distcp`, `.metadata` and named training
state. This conservative safetensors-only export policy intentionally excludes
legacy pickle model weights. Packaging is not model conversion. A merged WSM
model-only output is not an optimizer-resumable checkpoint.

## Offline packaging and verification

Finish an optimizer update, synchronize all writers/ranks, and freeze the source
folder before running these commands. The packer copies and verifies the copy,
but does not provide a lock against a concurrently mutating producer.

Prepare your actual role-to-file-list JSON using
[`training_components.example.json`](../../configs/artifacts/training_components.example.json).
The example paths are **not** a training recipe. Include every rank's state and
all relevant common/sharded files. Several roles can reference the same upstream
file if state is co-located. Scheduler, cursor, training manifest and environment
roles must include nonempty JSON objects; binary upstream state can be additional
files. Record current/next global update, LR policy/horizon, token/data cursor,
RNG domains, dataset revision, tokenizer identity, resolved config, code revision,
PyTorch/Megatron/CUDA versions, rank/world size and TP/PP/DP/EP layout. Do not dump
`os.environ` or write credentials. Roles attest producer semantics; the packer
checks presence and bytes, **not** the optimizer tensor contents or exact resume.

```bash
python scripts/hf_checkpoint.py pack \
  --source /checkpoints/frozen-step-1000 --output /staging/pretrain-week01-step1000 \
  --kind torch_dist_training --stage pretrain --run week01 --checkpoint step_1000 \
  --components /checkpoints/training_components.json
python scripts/hf_checkpoint.py verify \
  --source /staging/pretrain-week01-step1000 --kind torch_dist_training
python scripts/hf_checkpoint.py upload --source /staging/pretrain-week01-step1000 \
  --repo-id OWNER/nemochat-checkpoints --dry-run
python scripts/hf_checkpoint.py restore-local \
  --source /staging/pretrain-week01-step1000 --output /restore/local-check \
  --kind torch_dist_training --trusted-inputs
```

These are local operations: dry-run does **not** check Hub access/privacy or upload.
The generated `checkpoint_inventory.json` covers exact file names, byte sizes,
and SHA256 for **every** file including each shard and index, tokenizer and state.
Unsafe paths, wildcard metacharacters, duplicate JSON keys, symlinks, special
files, missing/extra/changed files and missing shard-index targets are refused.
An HF model can be packaged from a separate clean export with `--kind hf_model`
and no `--components`; run its actual HF loader separately.

## Private Hub transfer

Provision an **existing private model repository** in a separately authorized
operation. The pipeline never automatically creates a public repository, flips
visibility, commits locally, deletes remote history, or changes a repo policy.
Set `HF_TOKEN` through the rental platform's secret manager; never put it in a
manifest, command-line token option, shell history or log. Only the following
commands perform remote operations:

```bash
python scripts/hf_checkpoint.py upload --source /staging/pretrain-week01-step1000 \
  --repo-id OWNER/nemochat-checkpoints
```

Upload checks repository visibility, rejects an occupied checkpoint namespace,
uses the observed parent commit to guard against concurrent writers, and uploads
to:

```text
stages/pretrain/runs/week01/checkpoints/step_1000/torch_dist_training/
```

Only after the resulting full commit SHA is returned, the committed manifest
hash matches the local manifest, the exact remote file set matches, and **all**
remote bytes are downloaded and verified does it emit a success receipt with
`repo_id`, `revision`, `path`, `kind`, and `manifest_sha256`. Save that receipt
on storage independent of the expiring host. Errors issue no verified receipt;
remote state may nevertheless contain a commit if failure happened at read-back.
An occupied path is intentionally never overwritten by a retry. Inspect/verify
the immutable commit or use a fresh checkpoint identity, not a mutable `main`
reference. Only one uploader should own a run namespace.

On the next rental, use the exact receipt values:

```bash
python scripts/hf_checkpoint.py download --repo-id OWNER/nemochat-checkpoints \
  --revision FULL_40_CHARACTER_COMMIT_SHA \
  --path stages/pretrain/runs/week01/checkpoints/step_1000/torch_dist_training \
  --output /checkpoints/restored-week01-step1000 --kind torch_dist_training \
  --manifest-sha256 MANIFEST_SHA256_FROM_RECEIPT --trusted-inputs
```

`main`, tags, abbreviated SHAs and mismatched kinds are refused. Downloads use
`snapshot_download(local_dir=...)` with exact, narrow allow-pattern lists: first
the manifest, then the declared files only. Repo siblings stay off disk. Hub
local download metadata stays outside the published payload. Verification
precedes Linux `renameat2(RENAME_NOREPLACE)` atomic publication on the same
filesystem. Existing outputs and symlink destination ancestors are refused.
Atomic rename is not a durable fsync/power-loss transaction. Filesystems/platforms
without Linux renameat2 are unsupported, rather than silently losing no-overwrite
protection. Training restores require explicit `--trusted-inputs` **before any
remote call** because DCP metadata/torch states may execute pickle code when a
later trainer loads them. This utility itself never deserializes them.

Public transfer is refused by default. An exception requires BOTH
`--rights-reviewed` and `--repo-policy reviewed-policy.json` where
`rights_reviewed` is true and `allow_public_repo` exactly equals the repository.
The same policy applies on download. The policy is a human rights-review
attestation, not automated licensing approval. See
[`repo_policy.private.json`](../../configs/artifacts/repo_policy.private.json).

## Upload API investigation (official sources + installed SDK)

Inspected with urllib and Python introspection on the implementation host:
installed `huggingface_hub==0.36.2` in `.venv`; system Python has no Hub package.
Sources:

- [Current/main upload guide](https://huggingface.co/docs/huggingface_hub/main/en/guides/upload)
- [0.36.0 upload guide](https://huggingface.co/docs/huggingface_hub/v0.36.0/en/guides/upload)
- [Download guide](https://huggingface.co/docs/huggingface_hub/guides/download)
- [Repository limits](https://huggingface.co/docs/hub/repositories-recommendations)

There is a real version boundary. The latest main guide describes streamed,
multi-commit `upload_folder` with automatic retry/reuse, including across hosts.
The installed **0.36.2** implementation uses `create_commit`, supports
`path_in_repo`, `parent_commit`, and returns `CommitInfo.oid`; it is the transport
used here. Re-running may reuse already-uploaded storage objects, but this code
does **not** claim durable per-task resumability or off-host upload continuation
for that old API. Partial transfer is not a verified checkpoint.

The installed `upload_large_folder` has resumable local task metadata and
multi-threaded hashing/preupload/commits, but lacks `path_in_repo`, `parent_commit`,
a custom commit message, and a commit return value (returns `None`). It can create
a repository with a dangerous public default unless `private=True` is explicit.
Its persistent upload metadata is inside the local source, and its indefinite
retries can hide permanent auth failures. **Not enabled by this pipeline.** A
future integration must stage the full repository-relative tree, exclude upload
cache metadata from checkpoint inventory, serialize writers, resolve a final
immutable SHA, and apply exactly the same private-policy/full read-back gates.
Do not pass nonexistent `resume_upload` or `multi_commits` arguments. Revalidate
signatures and return semantics before upgrading the SDK; latest docs do not
prove an older installed API has the same behavior. Large-file shard size and
repo/storage quotas must be checked before the rental deadline.

## Rental-end gate and boundaries

1. Reserve upload + **full read-back** time and disk capacity (frozen source,
   packaged copy, download temp and storage/Xet cache). Don't wait for host expiry.
2. Save a completed-update full bundle, verify it locally, and upload privately.
3. Keep the full immutable verified receipt and environment/run/data manifests
   off-host. A progress bar, commit URL or uploaded manifest alone is not success.
4. Before releasing the rental, independently restore the saved SHA to a new
   destination and verify bytes; ideally perform an actual trainer resume probe.
5. On the next B300 host verify code/runtime/topology/data/tokenizer identity;
   inventory equality does not make different distributed layouts compatible or
   guarantee tensor-exact resumption. Dataset shards themselves are not included
   unless explicitly placed in the bundle; recreate them at pinned revisions.
6. Preserve the last verified checkpoint until the next has passed these gates.
   Hub history is not a retention policy; quota and access changes remain risks.

SHA hashes/immutable revisions protect byte integrity, not author authenticity,
licensing, malicious rewritten manifests, or concurrent filesystem attackers.
Trust the producer and repo access controls. Snapshot/download and Hub CI results
are **not** trusted training-code execution evidence.

## Verification scope

`python -m pytest -q tests/test_hf_checkpoint.py` covers offline pack/restore,
all-shard inventories, malformed paths/indexes, symlinks/overwrite, pinned
revisions, credentials, privacy exceptions and mocked transport/read-back
failures. The standalone CLI pack → verify → dry-run → restore-local flow was
also exercised on real filesystem bytes with an explicitly labeled **synthetic
opaque training bundle**. That proves transport/integrity only: no actual torch
resume, remote upload/download, Hub authentication/CI, B300 execution, model
loader compatibility, model quality or multi-week off-host rental was exercised.
