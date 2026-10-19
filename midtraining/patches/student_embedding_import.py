"""Per-import native mapping alias for the official student HF serialization.

No checkpoint tensor/byte edits or alternate loader. Rebuild the native mapping
registry for the bound student import hook only; teacher plural names stay native.
"""
import hashlib
import inspect
import json
from pathlib import Path
from types import MethodType

SINGULAR = 'backbone.embedding.weight'
PLURAL = 'backbone.embeddings.weight'
DESTINATION = 'embedding.word_embeddings.weight'
BRIDGE_SHA256 = '6ec76ceacbdf0f96fbd19bed5e03a9ce8c11eace2b38c6297bdb8e33c451f899'
LOADER_SHA256 = 'ccd201557431bfd9bd34d603398f70b6502d542b66ecdc6061b8934f4fe83a80'


def rewrite_registry(registry, available_keys, mapping_factory):
    if SINGULAR in available_keys and PLURAL in available_keys:
        raise ValueError('ambiguous singular/plural embedding source')
    if PLURAL in available_keys:
        return registry
    if SINGULAR not in available_keys:
        raise ValueError('learned source embedding absent from checkpoint index')
    mappings = []; matches = 0
    for mapping in registry.mappings:
        if mapping.megatron_param == DESTINATION:
            if mapping.hf_param != PLURAL:
                raise ValueError('unreviewed native embedding mapping')
            mappings.append(mapping_factory(megatron_param=DESTINATION, hf_param=SINGULAR))
            matches += 1
        else:
            mappings.append(mapping)
    if matches != 1:
        raise ValueError('exactly one native embedding mapping required')
    return type(registry)(*mappings)


def attach(provider, hf_path):
    from megatron.bridge.models.conversion.param_mapping import AutoMapping
    keys = set(json.loads((Path(hf_path)/'model.safetensors.index.json').read_text())['weight_map'])
    receipt = dict(checkpoint=str(hf_path), metadata_only=True, source_tensor_keys=len(keys),
                   alias_required=SINGULAR in keys, native_embedding_transform_calls=0,
                   checkpoint_bytes_audited=False, installed_sdk_files_edited=False)
    if PLURAL in keys and SINGULAR not in keys:
        receipt['status'] = 'native_plural_embedding_no_patch_needed'
        return receipt
    if SINGULAR not in keys or PLURAL in keys:
        raise ValueError('missing/ambiguous learned embedding metadata')
    hooks = [hook for hook in getattr(provider,'_pre_wrap_hooks',[])
             if getattr(getattr(hook,'func',None),'__name__',None)=='load_weights_hf_to_megatron']
    if len(hooks)!=1:
        raise RuntimeError('exact native HF load hook is required; no random fallback')
    loader = hooks[0].func
    bridge = loader.__self__
    bridge_path=Path(inspect.getsourcefile(type(bridge)))
    loader_path=Path(inspect.getsourcefile(loader))
    if (hashlib.sha256(bridge_path.read_bytes()).hexdigest()!=BRIDGE_SHA256
            or hashlib.sha256(loader_path.read_bytes()).hexdigest()!=LOADER_SHA256):
        raise RuntimeError('unreviewed native HF conversion source; refuse embedding alias patch')
    original_registry=bridge.mapping_registry

    def patched_registry(self):
        registry=rewrite_registry(original_registry(),keys,AutoMapping)
        missing=[name for name in sorted(keys) if registry.hf_to_megatron_lookup(name) is None]
        if missing:
            raise RuntimeError('native HF import does not cover all source tensor names: '+repr(missing[:10]))
        receipt['source_key_coverage']='all_index_keys_have_native_mappings'
        embedding=next(mapping for mapping in registry.mappings if mapping.megatron_param==DESTINATION)
        native_transform=embedding.hf_to_megatron
        def observed_native_transform(*args,**kwargs):
            output=native_transform(*args,**kwargs)
            if output is not None:
                receipt['native_embedding_transform_calls']+=1
                receipt['native_embedding_shape']=list(output.shape)
            return output
        embedding.hf_to_megatron=observed_native_transform
        return registry

    bridge.mapping_registry=MethodType(patched_registry,bridge)
    receipt.update(status='bound_native_student_import_alias_installed',source_key=SINGULAR,
                   native_destination=DESTINATION, upstream_bridge_sha256=BRIDGE_SHA256,
                   upstream_loader_sha256=LOADER_SHA256,
                   adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    return receipt
