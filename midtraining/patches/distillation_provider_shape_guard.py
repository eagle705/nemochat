"""Version-gated, process-local fix for unequal-shape DistillationProvider.

Never edits installed SDK files. Preserve native __setattr__ for shared runtime
settings; do not mirror student architecture assignments into the teacher.
"""
import hashlib
import inspect
from pathlib import Path

UPSTREAM_SHA256 = '21b1e634f64c67ed23e602e79dd932df62f9b7db6416dd06cffdf3eb237247b5'
PROTECTED_FIELDS = frozenset(('num_layers', 'hidden_size', 'vocab_size', 'num_attention_heads',
    'num_query_groups', 'kv_channels', 'ffn_hidden_size', 'moe_latent_size', 'moe_ffn_hidden_size',
    'num_moe_experts', 'moe_router_topk', 'moe_shared_expert_intermediate_size', 'mamba_num_heads',
    'mamba_head_dim', 'mamba_num_groups', 'mamba_state_dim', 'hybrid_layer_pattern',
    'share_embeddings_and_output_weights'))


def apply(provider_class, source_path=None):
    source = Path(source_path or inspect.getsourcefile(provider_class))
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != UPSTREAM_SHA256:
        raise RuntimeError('unreviewed DistillationProvider source; refuse automatic shape patch')
    if getattr(provider_class, '_nemochat_shape_patch', False):
        return dict(status='already_applied_process_local', upstream_sha256=digest)
    original = provider_class.__setattr__

    def preserve_independent_architecture(self, name, value):
        if name in PROTECTED_FIELDS:
            object.__setattr__(self, name, value)
        else:
            original(self, name, value)

    provider_class.__setattr__ = preserve_independent_architecture
    provider_class._nemochat_shape_patch = True
    return dict(status='applied_process_local_no_sdk_files_edited',
                upstream_path=str(source), upstream_sha256=digest,
                adapter_path=str(Path(__file__)),
                adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                protected_architecture_fields=sorted(PROTECTED_FIELDS))
