"""Human role cards for explicitly annotated operations, never inferred kernels."""
ROLES = {
    "routing": ("Router scores", "Project each token to expert logits; these are not expert outputs."),
    "topk": ("Select top-k", "Choose two experts per token and normalize selected routing weights."),
    "permutation": ("Group token copies", "Sort token/expert assignments by expert; duplicates are intentional."),
    "dispatch": ("Local dispatch", "Gather selected tokens into one expert batch; no network or all-to-all."),
    "expert_up": ("Expert up GEMM", "Project expert inputs to gate and value branches."),
    "activation": ("SwiGLU", "SiLU(gate) times value; eager operations, not a native fused kernel."),
    "expert_down": ("Expert down GEMM", "Project activated expert hidden states back to model width."),
    "combine": ("Weighted combine", "Weight each expert copy then scatter-add to original token positions."),
    "backward": ("Autograd entry", "Backward entry only; does not identify each gradient kernel's forward source."),
}
