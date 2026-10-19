"""Export actual final KD1000 global DCP student records, never BASE weights."""
import argparse, collections, importlib.util, json, os, shutil, time
from pathlib import Path
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from transformers import AutoModelForCausalLM, AutoTokenizer
ROOT=Path('/raid/yoon/nemochat')
OUT=ROOT/'results/midtraining/hf-kd1000-conversion-01'
SOURCE=ROOT/'results/midtraining/offline2b-prefixtrain-1000-01/checkpoint/iter_0001000'
ASSETS=ROOT/'checkpoints/exports/nemochat_baby_base_bf16'
TARGET=ROOT/'checkpoints/exports/nemochat_mid_super35_distill_hf_bf16_01'
BASE_CODE=ROOT/'results/pretraining/nemochat_baby/hf_transformers_conversion/convert_base.py'
spec=importlib.util.spec_from_file_location('reviewed_base_schema',BASE_CODE)
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)

def selected(metadata,model_prefix=''):
    model={};excluded={}
    for key,record in metadata.state_dict_metadata.items():
        normalized=key.removeprefix(model_prefix) if key.startswith(model_prefix) else key
        if key.startswith(model_prefix) and normalized.startswith(('embedding.','decoder.','output_layer.')):
            if hasattr(record,'size'):model[key]=record
            else:excluded[key]='model_backend_extra_state'
        elif key.startswith(('optimizer.','rng_state/','common_state/')):excluded[key]='training_only'
        else:raise ValueError('unclassified source key '+key)
    assert len(model)==619 and sum(v.properties.dtype==torch.float32 for v in model.values())==23
    return model,excluded

def convert(schema_only=False,source=SOURCE,target=TARGET,report_dir=OUT,model_prefix="",source_step=1000,stage="midtraining_lm_plus_logits_kd"):
    global SOURCE,TARGET,OUT
    SOURCE=Path(source);TARGET=Path(target);OUT=Path(report_dir);OUT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);start=time.monotonic()
    reader=dcp.FileSystemReader(SOURCE);meta=reader.read_metadata();records,excluded=selected(meta,model_prefix)
    tok=AutoTokenizer.from_pretrained(ASSETS/'tokenizer',local_files_only=True,trust_remote_code=False)
    config=base.make_config(tok)
    with torch.device('meta'):model=AutoModelForCausalLM.from_config(config,trust_remote_code=False)
    expected=model.state_dict()
    state={k:torch.empty(v.size,dtype=v.properties.dtype,device='meta' if schema_only else 'cpu') for k,v in records.items()}
    if not schema_only:
        assert TARGET.is_dir() and not any(TARGET.iterdir()),'fresh empty target required'
        print('Loading ONLY actual KD1000 student tensors',len(state),str(SOURCE),flush=True)
        dcp.load(state,storage_reader=reader,no_dist=True,planner=DefaultLoadPlanner(flatten_state_dict=False,flatten_sharded_tensors=False))
        for key,tensor in state.items():
            if not torch.isfinite(tensor).all():raise ValueError('nonfinite source tensor '+key)
        print('All source tensors loaded and finite',time.monotonic()-start,flush=True)
    normalized={k.removeprefix(model_prefix):v for k,v in state.items()}
    converted,provenance=base.map_state(normalized)
    provenance={k:[model_prefix+s for s in values] for k,values in provenance.items()}
    assert set(converted)==set(expected),(set(converted)-set(expected),set(expected)-set(converted))
    for key,tensor in converted.items():assert tensor.shape==expected[key].shape,(key,tensor.shape,expected[key].shape)
    assert {s for values in provenance.values() for s in values}==set(state)
    if schema_only:
        report={'status':'passed','source':str(SOURCE),'native_model_tensor_records':len(state),'hf_tensor_records':len(converted),'excluded_records':len(excluded),'schema_exact':True}
        (OUT/'schema_result.json').write_text(json.dumps(report,indent=2));print(report);return
    result=model.load_state_dict(converted,strict=True,assign=True)
    assert not result.missing_keys and not result.unexpected_keys
    biases={k:t for k,t in model.state_dict().items() if 'e_score_correction_bias' in k}
    assert len(biases)==23 and all(t.dtype==torch.float32 for t in biases.values())
    model.save_pretrained(TARGET,safe_serialization=True,max_shard_size='2GB')
    for asset in (ASSETS/'tokenizer').iterdir():
        if asset.is_file():shutil.copyfile(asset,TARGET/asset.name)
    for asset in TARGET.iterdir():
        if asset.is_file():os.chown(asset,1001,1001);os.chmod(asset,0o640)
    os.chown(TARGET,1001,1001)
    report={'status':'exported_not_yet_reloaded','stage':stage,'source':str(SOURCE),'source_step':source_step,'output':str(TARGET),'source_weights':'actual_selected_native_model_only_NOT_original_BASE', 'model_prefix':model_prefix,'source_tensor_records':len(state),'hf_tensor_records':len(converted),'excluded_records':len(excluded),'source_dtype_counts':dict(collections.Counter(str(t.dtype) for t in state.values())),'all_source_tensors_finite':True,'all_source_model_keys_mapped':True,'parameter_numel':sum(p.numel() for p in model.parameters()),'router_correction_dtype':'torch.float32','router_correction_count':len(biases),'source_integrity':'byte_SHA_CRC_audits_explicitly_waived_not_run','base_usage':'config_tokenizer_and_read_only_mapping_code_only_no_BASE_weights_loaded','trust_remote_code':False,'bridge_entrypoint_executed':False,'mapping':'existing reviewed pinned Bridge registry plus global DCP adapters','elapsed_seconds':time.monotonic()-start,'files':{p.name:p.stat().st_size for p in TARGET.iterdir() if p.is_file()}}
    (OUT/'mapping_provenance.json').write_text(json.dumps(provenance,indent=2))
    (OUT/'excluded_native_records.json').write_text(json.dumps(excluded,indent=2))
    (OUT/'export_result.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--schema-only',action='store_true');p.add_argument('--source',type=Path,default=SOURCE);p.add_argument('--output',type=Path,default=TARGET);p.add_argument('--report-dir',type=Path,default=OUT);p.add_argument('--model-prefix',default='',help='Exact explicitly selected model wrapper prefix; unknown namespaces are fatal');p.add_argument('--source-step',type=int,default=1000);p.add_argument('--stage',default='midtraining_lm_plus_logits_kd');a=p.parse_args();convert(a.schema_only,a.source,a.output,a.report_dir,a.model_prefix,a.source_step,a.stage)
