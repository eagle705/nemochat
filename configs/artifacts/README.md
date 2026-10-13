# Artifact declarations

`training_components.example.json` is a **layout example**, not an executable
training recipe. Replace every path with the real frozen checkpoint's paths;
include every model and distributed-optimizer shard, every rank's RNG state,
tokenizer assets and the actual run/training/environment contracts. Roles may
reference the same DCP shard/common file if the upstream writer co-locates state.
The inventory independently covers ALL source files, not just the listed roles.
Scheduler/cursor/training_manifest/environment declarations must have nonempty
JSON objects. Put actual scheduler state and optimizer/RNG tensor state in their
upstream-native locations; declaring a file does not prove its resume semantics.

`repo_policy.private.json` intentionally authorizes no public repository. An
exception requires both the CLI `--rights-reviewed` flag and a reviewed policy
with `rights_reviewed: true`, `allow_public_repo: "owner/exact-repo"`. Do not store
HF tokens, signed URLs or sensitive environment dumps in these declarations.
