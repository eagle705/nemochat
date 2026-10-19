"""Opt-in finite activation observations for a bounded native KD diagnostic.

No tensor replacement, loss clipping, gradient repair or acceptance relaxation.
CUDA synchronization changes timing; never use these samples as throughput data.
"""
import json
import math
from pathlib import Path
import re
import time


def tensors(value):
    import torch
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (tuple, list)):
        return [tensor for item in value for tensor in tensors(item)]
    if isinstance(value, dict):
        return [tensor for item in value.values() for tensor in tensors(item)]
    return []


def install(model, label, path):
    import torch
    path = Path(path)
    handles = []
    for name, layer in model.named_modules():
        if name not in ('embedding', 'output_layer', 'decoder.final_norm') and not re.fullmatch(r'decoder\.layers\.\d+', name):
            continue
        def observe(module, inputs, output, name=name):
            with torch.no_grad():
                for index, value in enumerate(tensors(output)):
                    if not (value.is_floating_point() or value.is_complex()):
                        continue
                    detached = value.detach()
                    nonfinite = int((~torch.isfinite(detached)).sum().item())
                    maximum = float(detached.abs().max().item()) if detached.numel() else 0.
                    row = dict(model=label, module=name, tensor=index, shape=list(value.shape),
                               dtype=str(value.dtype), nonfinite_elements=nonfinite,
                               max_abs=maximum if math.isfinite(maximum) else repr(maximum),
                               observation_only=True, timestamp_ns=time.time_ns())
                    with path.open('a') as stream:
                        stream.write(json.dumps(row, allow_nan=False)+'\n')
            return None
        handles.append(layer.register_forward_hook(observe))
    return handles
