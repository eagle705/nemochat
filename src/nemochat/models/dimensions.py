"""Explicit A/B hybrid model dimensions, independent of the GPU SDK."""

from dataclasses import dataclass, asdict

BRIDGE_REVISION = "c860f8a5bc5fddd78690f32baa0b8696774308b4"
PATTERN = "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME"


@dataclass(frozen=True)
class ModelDimensions:
    variant: str = "A"
    hidden: int = 2048
    latent: int = 512
    expert_ffn: int = 4096
    shared_ffn: int = 2048
    experts: int = 128
    top_k: int = 6
    vocab: int = 131072
    mamba_heads: int = 48
    mamba_head_dim: int = 64
    mamba_groups: int = 8
    mamba_state: int = 128
    attention_heads: int = 32
    attention_head_dim: int = 128
    query_groups: int = 2
    pattern: str = PATTERN

    def validate(self):
        if self.variant not in ("A", "B"):
            raise ValueError("Only the approved A/B pilot is implemented")
        if self.pattern != PATTERN:
            raise ValueError("This tutorial uses the fixed 52-block Nano pattern")
        numeric = {key: value for key, value in asdict(self).items()
                   if key not in ("variant", "pattern")}
        if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0
               for value in numeric.values()):
            raise ValueError("Model dimensions must be positive integers")
        if self.top_k > self.experts or self.attention_heads % self.query_groups:
            raise ValueError("Invalid expert top-k or attention query groups")
        if self.mamba_heads % self.mamba_groups:
            raise ValueError("Mamba heads must divide into groups")
        expected = (512, 4096) if self.variant == "A" else (1024, 2048)
        if (self.latent, self.expert_ffn) != expected:
            raise ValueError("Latent and expert FFN dimensions must match variant")


def dimensions(variant="A"):
    if variant == "A":
        result = ModelDimensions()
    elif variant == "B":
        result = ModelDimensions(variant="B", latent=1024, expert_ffn=2048)
    else:
        raise ValueError("Choose A or B")
    result.validate()
    return result


def parameter_estimate(model):
    """Analytical weight count matching the plan, NOT instantiated model numel.

    Counts both embedding and output weights in active parameters; this is
    not a FLOPs-equivalent dense count. MTP excluded; norms/biases approximated.
    """
    model.validate()
    h = model.hidden
    inner = model.mamba_heads * model.mamba_head_dim
    mamba_blocks = model.pattern.count("M")
    moe_blocks = model.pattern.count("E")
    attention_blocks = model.pattern.count("*")
    groups_state = model.mamba_groups * model.mamba_state
    per_mamba = h * (2 * inner + 2 * groups_state + model.mamba_heads) + inner * h
    per_mamba += (inner + 2 * groups_state) * 5 + 3 * model.mamba_heads + inner
    q_width = model.attention_heads * model.attention_head_dim
    kv_width = model.query_groups * model.attention_head_dim
    per_attention = 2 * h * q_width + 2 * h * kv_width
    pieces = {
        "embedding_and_output": 2 * model.vocab * h,
        "mamba": mamba_blocks * per_mamba,
        "attention": attention_blocks * per_attention,
        "shared_experts": moe_blocks * 2 * h * model.shared_ffn,
        "routers": moe_blocks * h * model.experts,
        "norms": len(model.pattern) * h + h,
        "latent_projections": moe_blocks * 2 * h * model.latent,
        "all_routed_experts": moe_blocks * model.experts * 2 * model.latent * model.expert_ffn,
    }
    active_routed = moe_blocks * model.top_k * 2 * model.latent * model.expert_ffn
    total = sum(pieces.values())
    active = total - pieces["all_routed_experts"] + active_routed
    return {"variant": model.variant, "blocks": {"mamba": mamba_blocks,
            "moe": moe_blocks, "attention": attention_blocks},
            "components": pieces, "total_parameters": total,
            "active_parameters": active, "total_billions": total / 1e9,
            "active_billions": active / 1e9,
            "kind": "analytical estimate, not actual numel or FLOPs"}
