# Scoped upstream redistribution notices

This directory does not license the whole NemoChat repository, model weights,
training datasets, cached teacher outputs, or benchmark contents.

## Megatron-LM cached-logit port

`../bridge_cached_kd/{logits_saver,utils_logits,cached_logits_loss}.py` are
**modified ports**, not original NemoChat implementations. The original NVIDIA
copyright headers are retained. The modifications adapt imports/runtime binding
for the Bridge adapter; they do not change the upstream rights. The corresponding
upstream source identities and SHA256 values are in
`../evidence/installed-core-provenance.json` and `../evidence/integration-base.json`.
The one Core source snapshot retained for CPU method-comparison tests is
`../evidence/installed-core__megatron__training__distillation__cached_logits_loss.py`.

`Megatron-LM-LICENSE.txt` is the exact full license from
`/opt/Megatron-Bridge/3rdparty/Megatron-LM/LICENSE` in the already-local pinned image
`nvcr.io/nvidia/nemo@sha256:fdd6e9c7929b76c8624ddfea939dda345beaf67695d9c3994994c18f9e79b9bf`.
It was read without starting a container, using the stopped same-image container
recorded by `../run_cpu_sdk.py`. Its SHA256 is
`39f1df137a306d31698db206063f462b69c19e536f012db09a66997ad2cb96f0`.
The file states its default redistribution conditions and preserves the additional
upstream license texts; individual source-file notices still take precedence.

## Bridge seam and ModelOpt comparison snapshot

`../evidence/bridge-installed/training__gpt_step.py` is the unmodified Bridge
0.6.1 seam at base `2a8fcdd05181ffcef261192f71ff5910d0005bf2`, retained to
check/reproduce the isolated patch, not an installed SDK replacement.
`../evidence/installed__modelopt__torch__distill__plugins__megatron.py` is the
unmodified installed ModelOpt snapshot used by CPU AST-method comparisons.
Its header declares Apache-2.0 and NVIDIA copyright (2024).

`Apache-2.0.txt` is the exact full Apache license from the same image's
`/opt/Megatron-Bridge/LICENSE` (SHA256
`c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4`).
The image's Bridge and Core root `NOTICE` paths were absent; no notice was
invented. Existing per-file copyright/license notices remain intact.

Only the minimal source/provenance snapshots needed by CPU comparisons and patch
reproduction are included. Other upstream copies and historical execution logs
remain local, outside the staged release. This is a scoped source attribution,
not a legal clearance of arbitrary runtime artifacts or a fresh SDK/GPU test.
