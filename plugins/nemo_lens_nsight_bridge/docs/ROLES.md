# Operation role cards: explicit evidence, not a kernel-name lookup

[한국어](ROLES.ko.md). Source references point to the actual functions in
[`control.py`](../lens_nsight_bridge/control.py). Each operation record stores
exact file SHA256 + function lines + snippet SHA256 + source ID; the CLI extracts
those **snapshot** lines, not this prose or a generated substitute.

| Role / source function | Inputs → output | Why a modeler cares / range meaning |
|---|---|---|
| `routing` / `routing` | `[T,D] @ [D,E] → [T,E]` | Router logits select experts; not expert feature computation. |
| `topk` / `topk` | logits → `[T,K]` weights/indices | Top-2 selected-logit softmax; active experts sparse, total parameters not all active. |
| `permutation` / `permutation` | assignments → expert-sorted token copies/weights | T×K logical assignments; stable argsort retains exact indices/weights. |
| `dispatch` / `dispatch` | token copies + expert ID → local expert batch/IDs/mask | Local gather only. Empty batches are valid; no all-to-all/NCCL/EP transport. |
| `expert_up` / `expert_up` | `[N_e,D] @ [D,2H] → [N_e,2H]` | Expert-specific dense projection; grouping/fusion is NOT implemented. |
| `activation` / `activation` | split up → `SiLU(gate) * value` | SwiGLU gate nonlinearity; eager SiLU/multiply, not a native fused kernel. |
| `expert_down` / `expert_down` | `[N_e,H] @ [H,D] → [N_e,D]` | Return expert hidden features to model width. |
| `combine` / `combine` | expert outputs × routing weight → original `[T,D]` | Scatter-add preserves contributions of each token's selected experts. |
| `backward` / `backward` | scalar mean-square loss → parameter/input grads | Autograd entry, NOT each backward kernel's originating forward source. |

`T=8, D=16, E=4, K=2, H=24` in this control. The loss is an educational
mean-square objective on random tensors, not training quality or Nemotron data.
The token-major reference validates dispatch/combination output and gradients
with the same routing/top-k weights (rtol/atol 1e-5/1e-6 for output,
1e-4/1e-6 for gradients). It does not establish native fused/distributed parity.

## Evidence levels

1. **Explicit host operation:** source-verified callable wrapped in matching
   NVTX/Lens scope (real NVTX hardware execution pending here).
2. **Kernel attribution:** only after actual CUDA launch correlation and collected
   backtrace/compiler metadata; no role inferred from time proximity.
3. **Fusion / backward / graph:** many eager launches, many-to-one fusion, backward
   forward-origin ambiguity, replay-source ambiguity remain explicit unknowns.
4. **Native UI:** range labels are the implemented surface; detailed native
   explanation/source widgets require a separately validated extension path.

Keep the raw kernel names and report rank/device/stream/correlation evidence when
adding a native GPU adapter. Do not relabel every GEMM as an expert just because
it runs near dispatch. Existing native Megatron NVTX must be reconciled first.
