"""Direct Bridge provider construction; GPU instantiation is NOT CPU-tested.

API source: NVIDIA-NeMo/Megatron-Bridge c860f8a5bc5fddd78690f32baa0b8696774308b4,
hybrid_provider.py, nemotron_3_nano.py and nemotron_h_bridge.py (Apache-2.0).
This tutorial chooses no MTP, BF16, EP8 and non-overlapped shared experts.
"""

from nemochat.models.dimensions import dimensions


def build_provider(variant="A", *, sequence_length=4096):
    if sequence_length < 2:
        raise ValueError("sequence_length must be at least 2")
    d = dimensions(variant)
    # Keep SDK imports here: counting dimensions must work on a CPU-only host.
    import torch
    from megatron.core.activations import squared_relu
    from megatron.bridge.models.hybrid.hybrid_provider import HybridModelProvider

    return HybridModelProvider(
        hybrid_layer_pattern=d.pattern,
        num_layers=len(d.pattern),
        hidden_size=d.hidden,
        vocab_size=d.vocab,
        should_pad_vocab=False,
        share_embeddings_and_output_weights=False,
        mamba_num_heads=d.mamba_heads,
        mamba_head_dim=d.mamba_head_dim,
        mamba_num_groups=d.mamba_groups,
        mamba_state_dim=d.mamba_state,
        num_attention_heads=d.attention_heads,
        kv_channels=d.attention_head_dim,
        num_query_groups=d.query_groups,
        seq_length=sequence_length,
        position_embedding_type="none",
        # Residual hidden and routed latent dimensions are deliberately separate.
        ffn_hidden_size=d.expert_ffn,
        num_moe_experts=d.experts,
        moe_latent_size=d.latent,
        moe_ffn_hidden_size=d.expert_ffn,
        moe_shared_expert_intermediate_size=d.shared_ffn,
        moe_router_topk=d.top_k,
        moe_router_topk_scaling_factor=2.5,
        moe_router_num_groups=1,
        moe_router_group_topk=1,
        moe_router_score_function="sigmoid",
        moe_router_enable_expert_bias=True,
        moe_router_load_balancing_type="seq_aux_loss",
        moe_aux_loss_coeff=0.0001,
        moe_router_dtype="fp32",
        moe_grouped_gemm=True,
        moe_token_dispatcher_type="alltoall",
        moe_shared_expert_overlap=False,
        moe_permute_fusion=True,
        activation_func=squared_relu,
        gated_linear_unit=False,
        is_hybrid_model=True,
        mtp_num_layers=0,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=8,
        expert_tensor_parallel_size=1,
        context_parallel_size=1,
        sequence_parallel=False,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        bf16=True,
        attention_softmax_in_fp32=False,
        apply_rope_fusion=False,
        cuda_graph_impl="none",
    )
