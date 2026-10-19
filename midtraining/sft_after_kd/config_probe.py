"""No-weight, no-device SDK ConfigContainer probe; never calls native pretrain."""
import importlib.util
import json
from pathlib import Path
import sys
import traceback
import os

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/os.environ.get('SFT_CONFIG_PROBE_DIR','results/midtraining/sft-after-kd-01/config-probe')
OUT.mkdir(exist_ok=True)
try:
    import yaml
    spec=importlib.util.spec_from_file_location('sft_config_probe_native',ROOT/'midtraining/sft_after_kd/native_sft_v1.py')
    native=importlib.util.module_from_spec(spec);sys.modules[spec.name]=native;spec.loader.exec_module(native)
    c=yaml.safe_load((ROOT/'midtraining/sft_after_kd/recipe-v1.yaml').read_text())
    m=json.loads((ROOT/'datasets/nemotron_sft/nemochat_baby/prepared-full-03/data/manifest.json').read_text())
    c=native.stage_plan(dict(c,max_updates=15,max_hours=.75),m)
    c['load_hf_weights']=False
    from megatron.bridge.data.base import DatasetProvider
    from dataclasses import dataclass
    @dataclass
    class ConfigOnlyDataset(DatasetProvider):
        def build_datasets(self,ctx):
            raise RuntimeError('config-only probe must never build datasets')
    cfg,parent=native.make_config(c,m,ConfigOnlyDataset,OUT,False)
    (OUT/'native-config.json').write_text(json.dumps(parent.config_snapshot(cfg.to_dict()),indent=2,allow_nan=False)+'\n')
    (OUT/'result.json').write_text(json.dumps(dict(status='NATIVE_CONFIG_CONSTRUCTED_NO_WEIGHTS_NO_GPU',native_updates=0,weight_loading=False,device_requests=[],checkpoint_integrity='not_checked_explicit_user_waiver'),indent=2)+'\n')
except Exception as exc:
    (OUT/'result.json').write_text(json.dumps(dict(status='CONFIG_CONSTRUCTION_BLOCKED_NO_GPU',error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc(),native_updates=0,weight_loading=False,device_requests=[]),indent=2)+'\n')
    raise
