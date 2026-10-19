"""No-device installed API discovery; metadata only, never model tensors."""
import inspect, importlib, json
from pathlib import Path
import torch
OUT=Path('/raid/yoon/nemochat/results/midtraining/sft-continue1500-01/discovery');OUT.mkdir(parents=True,exist_ok=True)
for name in ('megatron.bridge.training.callbacks','megatron.bridge.training.setup','megatron.bridge.training.optimizer_param_scheduler','megatron.bridge.training.model_load_save','megatron.bridge.training.checkpointing','megatron.bridge.training.train','megatron.bridge.data.samplers','megatron.bridge.training.pretrain'):
 try:
  m=importlib.import_module(name)
 except ImportError as e:
  print(name,repr(e));continue
 (OUT/(name+'.py')).write_text(inspect.getsource(m));print(name,m.__file__)
p=Path('/raid/yoon/nemochat/results/midtraining/sft-after-kd-runtime05/main/checkpoint/iter_0000905')
print('inventory',[(x.name,x.stat().st_size) for x in p.iterdir()])
for x in p.rglob('train_state.pt'):
 state=torch.load(x,map_location='cpu',weights_only=False);print('TRAIN_STATE',repr(state));(OUT/'train-state.txt').write_text(repr(state))
