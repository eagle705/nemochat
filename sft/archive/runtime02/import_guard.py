"""Fail closed on silent native destination mapping omissions.

The existing version-gated student_embedding_import adapter owns the embedding
alias and HF-source-to-native coverage. This observer additionally checks every
local native parameter/persistent buffer has a real local conversion task. It
never reads, hashes, modifies or substitutes checkpoint tensors.
"""
import hashlib
import inspect
from pathlib import Path
from types import MethodType


def destination_coverage(models,tasks,persistent_buffers):
    required={}
    for index,model in enumerate(models):
        for name,param in list(model.named_parameters())+list(persistent_buffers(model)):
            if '_extra_state' not in name:
                required[id(param)]=f'{index}:{name}'
    covered={id(t.param_weight) for t in tasks if t is not None and t.megatron_module is not None and t.param_weight is not None}
    missing=[name for key,name in required.items() if key not in covered]
    if not required or missing:
        raise PermissionError('unmapped native C parameters/buffers would remain random: '+repr(missing[:20]))
    return dict(destination_required_tensors=len(required),destination_covered_tensors=len(set(required)&covered),
                destination_mapping_complete=True)


def attach(provider):
    receipt=getattr(provider,'_nemochat_hf_import_receipt',{})
    hooks=[h for h in getattr(provider,'_pre_wrap_hooks',[]) if getattr(getattr(h,'func',None),'__name__',None)=='load_weights_hf_to_megatron']
    if len(hooks)!=1:raise PermissionError('one actual native HF import hook required')
    bridge=hooks[0].func.__self__
    from megatron.bridge.models.conversion import model_bridge
    path=Path(inspect.getsourcefile(bridge.load_weights_hf_to_megatron))
    if hashlib.sha256(path.read_bytes()).hexdigest()!=receipt.get('upstream_loader_sha256'):
        raise PermissionError('native destination guard requires version-gated existing embedding adapter')
    result=dict(destination_mapping_complete=False,native_transform_outputs=0,checkpoint_payload_hashing=False,
                installed_sdk_edited=False,upstream_loader_sha256=receipt['upstream_loader_sha256'])
    original=bridge.build_conversion_tasks
    def checked(self,hf_model,models,*args,**kwargs):
        tasks=original(hf_model,models,*args,**kwargs)
        result.update(destination_coverage(models,tasks,model_bridge.persistent_buffers))
        seen=set()
        for task in tasks:
            if task is None or id(task.mapping) in seen:continue
            seen.add(id(task.mapping));transform=task.mapping.hf_to_megatron
            def observed(*a,_transform=transform,**kw):
                output=_transform(*a,**kw)
                if output is not None:result['native_transform_outputs']+=1
                return output
            task.mapping.hf_to_megatron=observed
        return tasks
    bridge.build_conversion_tasks=MethodType(checked,bridge)
    object.__setattr__(provider,'_runtime02_import_guard',result)
    return result
