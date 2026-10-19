# Third-party notices

## nanochat evaluation semantics

The task rendering/scoring and baseline-centering logic in `src/nemochat/eval/core.py` and related metric documentation is adapted from karpathy/nanochat, snapshot `92d63d4e8bb4df75c3b71618f31ddde2378b2bcd`.

Source: https://github.com/karpathy/nanochat . Original license:

```text
MIT License

Copyright (c) 2025 Andrej Karpathy

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## NVIDIA SDK and datasets

Bridge provider API examples are based on NVIDIA-NeMo/Megatron-Bridge, snapshot `c860f8a5bc5fddd78690f32baa0b8696774308b4`, whose source files carry Copyright (c) 2026 NVIDIA CORPORATION and Apache License 2.0: https://www.apache.org/licenses/LICENSE-2.0 . This implementation constructs a new tutorial configuration; it does not vendor the SDK.

Original ClimbMix: `nvidia/Nemotron-ClimbMix`, CC BY-NC 4.0: https://creativecommons.org/licenses/by-nc/4.0/ . Credit NVIDIA and the CLIMB authors; https://arxiv.org/abs/2504.13161 . Karpathy's seed-42 detokenized shuffle is credited separately. The derivative card's MIT label is not treated as removing original dataset conditions.

## Official WSM and NeMo Gym integration

The WSM launcher invokes, but does not vendor, `NVIDIA/Megatron-LM` PR #5114 at `ca4826e9555032b8710dfb4a4e5693c46c1613dd`. Preserve upstream notices and comply with the licenses/headers of the exact checkout; this repository's wrapper does not relicense upstream code. Source: https://github.com/NVIDIA/Megatron-LM/pull/5114 . Upstream code and its dependency environment are kept outside this repository.

The Gym importer targets official artifacts/configurations from `NVIDIA-NeMo/Gym` at `804c12a7d56a151a98ae150a352350e3c856ba47`: https://github.com/NVIDIA-NeMo/Gym . It does not vendor benchmark datasets or run/relicense their scorers. Each benchmark, model/tokenizer and dataset retains its own rights and usage conditions. Synthetic comparison fixtures test the importer and are not official benchmark data or measured model results.

This repository has since accumulated a separately scoped cached-KD source port and
minimal upstream comparison snapshots under `midtraining/patches/offline_kd_pr/`.
Their NVIDIA copyright headers, exact image/source provenance, modification notice,
and full applicable license texts are preserved in that directory's
[`licenses/README.md`](midtraining/patches/offline_kd_pr/licenses/README.md).
This is an exception to the earlier non-vendoring description, not a blanket
relicense of the SDK or this project.

The dashboard's renamed/subset Korean font is distributed under SIL OFL 1.1.
Preserve `dashboard/static/FONT-OFL.txt` and `FONT-PROVENANCE.json` with
`NemoChatKorean.ttf`; the latter records the upstream revision and subset hash.

These notices do not grant a blanket license over benchmark contents, tokenizer/model weights, teacher outputs or all tutorial datasets. Review each artifact's source rights separately before publishing.

## NeMo Lens and profiler lab extensions

The optional NeMo Lens tutorial and Nsight Bridge target `NVIDIA-NeMo/Lens` at `4d045c03186b90afdc815db46b671f6a21bc7e98`, licensed Apache-2.0 by NVIDIA: https://github.com/NVIDIA-NeMo/Lens . They import the separately installed SDK rather than vendoring its source. Keep the installed upstream license and record its revision.

The local trace UI uses a separately distributed Jaeger container. Its license and distribution notices remain those of the image/project: https://github.com/jaegertracing/jaeger . Runtime image identity belongs in execution evidence, not an implied production support claim.

Nsight Systems is a separately installed NVIDIA tool governed by its own license. A sample configuration-plugin manifest and Python NVTX instrumentation do not relicense Nsight, redistribute its binaries, or establish a public custom GUI-panel API. NVTX is distributed separately at https://github.com/NVIDIA/NVTX ; preserve applicable dependency notices. The readable MoE control and role descriptions are educational code, not vendored native expert/communication kernels.
