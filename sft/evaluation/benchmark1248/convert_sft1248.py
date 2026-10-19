"""Task-local SFT adapter to the proven same-architecture global-DCP exporter."""
import importlib.util,json,shutil,os,collections
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'results/midtraining/sft1248-benchmark-gpu-01'
SOURCE=ROOT/'results/midtraining/sft-continue1500-02/main/checkpoint/iter_0001248'
TARGET=ROOT/'checkpoints/exports/nemochat_sft1248_hf_bf16_01'
TOKENIZER=ROOT/'datasets/tokenizers/nemotron_super35/fc067b34f6b514014d2f8bddba855d2e4bfdfba6'
spec=importlib.util.spec_from_file_location('proven_export',ROOT/'midtraining/publication/export_kd1000/convert_kd1000.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
def selected(metadata,model_prefix=''):
    assert model_prefix==''
    model={};excluded={}
    for key,record in metadata.state_dict_metadata.items():
        if key.startswith(('embedding.','decoder.','output_layer.')):
            if hasattr(record,'size'):model[key]=record
            else:excluded[key]='model_backend_extra_state'
        elif key.startswith(('chained_0.optimizer.','chained_1.optimizer.','rng_state/','common_state/')):
            excluded[key]='training_only_observed_SFT_chained_optimizer_or_rng_common'
        else:raise ValueError('unclassified source key '+key)
    assert len(model)==619 and sum(v.properties.dtype==torch.float32 for v in model.values())==23
    return model,excluded
c.selected=selected
original_config=c.base.make_config
def make_config(_):
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(TOKENIZER,local_files_only=True,trust_remote_code=False)
    config=original_config(tok)
    config.max_position_embeddings=16384
    return config
c.base.make_config=make_config
if __name__=='__main__':
    import sys
    schema='--schema-only' in sys.argv
    accepted=json.loads((ROOT/'results/midtraining/sft-continue1500-02/complete.json').read_text())
    assert accepted['final_step']==1248 and accepted['checkpoint']==str(SOURCE)
    metadata=c.dcp.FileSystemReader(SOURCE).read_metadata()
    selected(metadata)
    (OUT/'source_metadata.json').write_text(json.dumps({'source':str(SOURCE),'source_step':1248,'metadata_only':True,'keys':list(metadata.state_dict_metadata)},indent=2))
    if not schema:c.convert(True,SOURCE,TARGET,OUT,'',1248,'selected_assistant_official_nonthinking_SFT_fullstate_continuation')
    c.convert(schema,SOURCE,TARGET,OUT,'',1248,'selected_assistant_official_nonthinking_SFT_fullstate_continuation')
    if not schema:
        old=c.ASSETS/'tokenizer'
        for asset in old.iterdir():
            if asset.is_file() and (TARGET/asset.name).exists():(TARGET/asset.name).unlink()
        for asset in TOKENIZER.iterdir():
            if asset.is_file():
                assert asset.suffix!='.py'
                destination=TARGET/('TOKENIZER_UPSTREAM_README.md' if asset.name=='README.md' else asset.name)
                previous=ROOT/'checkpoints/exports/nemochat_sft905_hf_bf16_01'/destination.name
                assert previous.read_bytes()==asset.read_bytes(),asset.name
                shutil.copyfile(previous,destination);os.chown(destination,1001,1001);os.chmod(destination,0o640)
        p=OUT/'export_result.json';r=json.loads(p.read_text())
        r.update(tokenizer_path=str(TOKENIZER),tokenizer_revision=TOKENIZER.name,tokenizer_assets='exact_migrated_SFT_assets_copied_verbatim',embedding_rows='unchanged_no_resize_no_permutation',initial_source_C=str(ROOT/'checkpoints/exports/nemochat_mid_super35_distill_hf_bf16_01'),native_generation='sft-continue1500-02/main',resume_source_step=905,segment_updates=343,load_mode='native_fullstate_continuation_before_model_only_export',native_HF_logit_parity='not_run',model_prefix='',source_weights='actual_FINAL_SFT1248_native_model_only_NOT_C_or_SFT905',base_usage='read_only_architecture_and_mapping_only_no_BASE_or_C_weight_load',files={f.name:f.stat().st_size for f in TARGET.iterdir() if f.is_file()})
        p.write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2),flush=True)
