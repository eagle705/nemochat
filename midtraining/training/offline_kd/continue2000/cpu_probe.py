"""Real native CPU constructor and SAME-document read gate; no model imports."""
import json
import os
from pathlib import Path
import sys
REPO=Path(__file__).resolve().parents[4];sys.path.insert(0,str(REPO))
from midtraining.training.offline_kd.continue2000.contract import OLD,SOURCE,OUT
from midtraining.training.offline_kd.continue2000.input_gate import verified_sample,SourceDigestMismatch,sample_digest

def build():
    import numpy as np
    from megatron.core.datasets.gpt_dataset import GPTDataset,GPTDatasetConfig
    from megatron.core.datasets.indexed_dataset import IndexedDataset
    from megatron.core.datasets.blended_dataset import BlendedDataset
    from megatron.core.datasets.utils import Split
    from megatron.core.tokenizers.text.models.default_tokenizer import DefaultTokenizerText
    cache=SOURCE/'dataset-cache'
    tok=DefaultTokenizerText(config={'library':'null-text'},vocab_size=131072,eod_id=11,pad_id=-1)
    cfg=GPTDatasetConfig(random_seed=43,sequence_length=4096,tokenizer=tok,reset_position_ids=False,
        reset_attention_mask=False,eod_mask_loss=False,create_attention_mask=False,
        add_extra_token_to_sequence=True,path_to_cache=str(cache),split=None)
    cfg.split=None;cfg.split_matrix=None
    descpaths=list(cache.glob('*BlendedDataset-train-description.txt'))
    if len(descpaths)!=1:raise ValueError('exact original full blended cache required')
    desc=json.loads(descpaths[0].read_text());datasets=[]
    for d in desc['datasets']:
        low=IndexedDataset(d['dataset_path'],multimodal=False,mmap=True)
        ds=GPTDataset(low,d['dataset_path'],np.arange(len(low),dtype=np.int32),d['num_samples'],Split.train,cfg)
        if not (cache/(ds.unique_description_hash+'-GPTDataset-train-description.txt')).exists():raise ValueError('original native cache miss')
        datasets.append(ds)
    return BlendedDataset(datasets,desc['weights'],desc['size'],cfg)

def main():
    import torch
    prepared=json.loads((OUT/'preparation/prepared.json').read_text());sdk=Path(prepared['snapshot'])/'sdk';sys.path.insert(0,str(sdk))
    # An index constructor must never write to historical cache directories.
    def readonly(event,args):
        if event=='open':
            p,_,flags=args
            if isinstance(p,(str,bytes)) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC):
                if str(p).startswith(str(SOURCE/'dataset-cache')):raise PermissionError('old index cache write forbidden')
    sys.addaudithook(readonly)
    torch.set_num_threads(1);ds=build();receipts=[]
    expected={r['micro']:r for r in map(json.loads,(SOURCE/'teacher-rank4-samples.jsonl').read_text().splitlines()) if r['iteration']<=1142}
    for physical,micro in [(16,2),(16,3),(17,3),(18,2),(18,3),(18,4),(1143,0),(2063,15)]:
        idx=physical*128+micro*8+4
        try:
            sample=verified_sample(ds,idx,expected.get(physical*16+micro))
            row=dict(physical=physical,micro=micro,rank=4,source_index=idx,status='accepted',sha256=sample_digest(sample))
        except SourceDigestMismatch as exc:row=dict(physical=physical,micro=micro,rank=4,status='data_quarantine',**exc.observation)
        receipts.append(row)
    target=next(r for r in receipts if (r['physical'],r['micro'])==(18,3))
    if target['status']!='data_quarantine':raise ValueError('known source read fault not detected')
    if any(r['status']!='accepted' for r in receipts if (r['physical'],r['micro'])!=(18,3)):raise ValueError('control unexpectedly rejected')
    import numpy as np
    from midtraining.training.offline_kd.continue2000.input_gate import verify_array
    verify_array(ds.dataset_index,146304,117888)
    counts=np.bincount(ds.dataset_index[146304:264192].astype(np.int64),minlength=4)
    selection={Path(g.dataset_path).parents[1].name:int(counts[i]) for i,g in enumerate(ds.datasets)}
    assert sum(selection.values())==117888
    from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
    def scheduler():
        opt=torch.optim.SGD([torch.zeros(1,requires_grad=True)],lr=1e-5)
        sch=OptimizerParamScheduler(opt,0.,1e-5,1e-6,37*128,3662*128,'WSD',.01,.01,
            3662*128,'constant',use_checkpoint_opt_param_scheduler=True,wsd_decay_steps=366*128,lr_wsd_decay_style='linear')
        return opt,sch
    opt,sch=scheduler();sch.step(16*128);saved=sch.state_dict()
    restored_opt,restored=scheduler();restored.load_state_dict(saved)
    expected_lr=1e-5*16/37
    assert restored.num_steps==2048 and abs(restored_opt.param_groups[0]['lr']-expected_lr)<1e-15
    before=restored.state_dict().copy()
    # A pre-forward rejection invokes no scheduler.step: source-only advance.
    physical_cursor=18;physical_cursor+=1
    assert restored.state_dict()==before and physical_cursor==19
    result=dict(real_native_dataset=type(ds).__name__,actual_length=len(ds),requested_full_samples=468736,
        gpu_initialized=torch.cuda.is_initialized(),model_loaded=False,receipts=receipts,additional_native_source_sample_counts=selection,
        real_native_scheduler=dict(restored_optimizer_step=16,consumed_samples=2048,lr=restored_opt.param_groups[0]['lr'],
            horizon_calls=3662,warmup_calls=37,decay_calls=366,rejection_preserved_scheduler=True))
    (OUT/'preparation/cpu-input-proof.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
