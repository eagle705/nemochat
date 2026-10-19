"""Native Bridge ordinary B=1.0LM, fixed C input replay, BASE fresh0."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import time

import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[5] / 'midtraining/training/offline_kd'))
from launch import REPO,STUDENT,KEYS,PARALLEL
sys.path.insert(0,str(REPO))
from midtraining.training.offline_kd.prefix.contract import build_full_datasets
from midtraining.training.offline_kd.ablation_lm_only.ordinary02.replay import Replay, ReplayDataset, durable, binding_for_save
from midtraining.training.offline_kd.ablation_lm_only.ordinary02.objective import forward_step
REPLAY=None
STOP=1000
OUT=Path('/output')
C=None
RANK=int(os.environ.get('RANK','0'))
STAGE=None

def publish(name,value):
    durable(OUT/(name+'.json'),value)
    print(name,json.dumps(value,allow_nan=False),flush=True)

def kd_helpers():
    spec=importlib.util.spec_from_file_location('offline2b_accepted_kd',REPO/'midtraining/training/native_kd.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def tokenizer_config():
    from megatron.bridge.training.config import TokenizerConfig
    return TokenizerConfig(tokenizer_type='NullTokenizer',vocab_size=131072,padded_vocab_size=131072,
                           use_tokenizer_vocab_size=False,null_tokenizer_eod_id=11)

def dataset_config():
    from megatron.bridge.training.config import GPTDatasetConfig
    manifest=json.loads(Path(C['manifest_path']).read_text())
    if hashlib.sha256(Path(C['manifest_path']).read_bytes()).hexdigest()!=C['manifest_sha256']:
        raise ValueError('corpus manifest changed')
    paths=[[str(REPO/manifest['sources'][key][split]['prefix']) for key in KEYS] for split in ('train','validation')]
    return GPTDatasetConfig(seq_length=4096,random_seed=C['seed'],blend=None,
        blend_per_split=[(paths[0],C['weights']),(paths[1],C['weights']),None],split=None,
        num_dataset_builder_threads=1,num_workers=0,dataloader_type='single',persistent_workers=False,
        eod_mask_loss=False,reset_position_ids=False,reset_attention_mask=False,
        skip_getting_attention_mask_from_dataset=True,add_extra_token_to_sequence=True,
        path_to_cache=str(OUT/'dataset-cache'))

def dataset_budget():
    return [468736,120,8]

def audited_common():
    from megatron.bridge.training.gpt_step import _forward_step_common
    seen={'micro':0,'masked_tokens':0}
    def common(state,iterator,model,return_schedule_plan=False):
        batch=next(iterator)
        if model.training:
            logical=int(state.train_state.step)
            micro=seen['micro']%16
            REPLAY.actual(batch,logical,micro)
            seen['micro']+=1
            seen['masked_tokens']+=int(batch['loss_mask'].sum())
        return _forward_step_common(state,iter([batch]),model,return_schedule_plan)
    return common,seen

def student():
    import torch
    import torch.distributed as dist
    from megatron.bridge.training.pretrain import pretrain
    from megatron.bridge.training.callbacks import Callback
    from megatron.bridge.training.config import (ConfigContainer,TrainingConfig,OptimizerConfig,SchedulerConfig,
        LoggerConfig,CheckpointConfig,DistributedDataParallelConfig,ValidationConfig,RNGConfig,runtime_config_update)
    from megatron.core.utils import unwrap_model
    replay_manifest=json.loads((OUT/'replay-manifest.json').read_text())
    assert replay_manifest['lm_coefficient']==C['lm_coefficient']==1.0
    mapping=replay_manifest['accepted_physical_mapping']
    expected_bytes=(OUT/f'expected-rank{RANK}.jsonl').read_bytes()
    assert hashlib.sha256(expected_bytes).hexdigest()==replay_manifest['expected_sha256'][str(RANK)]
    expected={r['micro']:r for r in map(json.loads,expected_bytes.splitlines())}
    def gather(row):
        ranks=[None]*8;dist.all_gather_object(ranks,row);return ranks
    global REPLAY
    start=0
    if RANK==0:publish('restart-lineage',dict(source='original_completed_BASE19074',source_hf=str(REPO/STUDENT),model_only=True,
        optimizer_reset=True,rng_reset=True,cursor_reset=True,start_step=0,schedule_horizon=C['train_iters'],
        bounded_stop=STOP,lm_coefficient=1.0,kd_disabled=True,comparison_kind='practical_deployed_recipe',coefficient_matched_ablation=False))
    kd=kd_helpers();provider=kd.provider_from_hf(REPO/STUDENT,load_weights=True)
    kd.configure_provider(provider,dict(parallel=PARALLEL,sequence_length=4096))
    provider.hidden_dropout=0.;provider.attention_dropout=0.;kd.check_provider_shapes(provider)
    interval=16
    from megatron.bridge.data.base import DatasetProvider
    from megatron.bridge.data.utils import pretrain_train_valid_test_datasets_provider
    class FullBudgetProvider(DatasetProvider):
        seq_length=4096
        def build_datasets(self,context):
            publish(f'{STAGE}-rank{RANK}-dataset-budget',dict(full=dataset_budget(),
                segment_requested=[context.train_samples,context.valid_samples,context.test_samples]))
            train,valid,test=build_full_datasets(C,context,pretrain_train_valid_test_datasets_provider,dataset_config())
            if train is None:raise RuntimeError('DP8 requires full native dataset on every rank')
            replay=Replay(train,mapping,expected,RANK,gather,OUT)
            globals()['REPLAY']=replay
            return ReplayDataset(replay),valid,test
    cfg=ConfigContainer(model=provider,
        train=TrainingConfig(train_iters=STOP,micro_batch_size=1,global_batch_size=128,
                             exit_duration_in_mins=None),
        optimizer=OptimizerConfig(lr=1e-5,min_lr=1e-6,weight_decay=.01,bf16=True,params_dtype=torch.bfloat16,
            use_distributed_optimizer=False,overlap_param_gather=False,clip_grad=1.,adam_beta1=.9,adam_beta2=.999,adam_eps=1e-8),
        ddp=DistributedDataParallelConfig(grad_reduce_in_fp32=True,overlap_grad_reduce=False,
            use_distributed_optimizer=False,overlap_param_gather=False,average_in_collective=False),
        scheduler=SchedulerConfig(lr_decay_style='WSD',lr_decay_iters=C['train_iters'],
            lr_warmup_iters=C['warmup_iters'],lr_wsd_decay_iters=C['decay_iters'],
            start_weight_decay=.01,end_weight_decay=.01,weight_decay_incr_style='constant',
            use_checkpoint_opt_param_scheduler=False),
        dataset=FullBudgetProvider(dataloader_type='single',num_workers=0,persistent_workers=False),tokenizer=tokenizer_config(),
        logger=LoggerConfig(log_interval=1,wandb_project=None,wandb_entity=None,wandb_exp_name=None,
            wandb_save_dir=str(OUT/'sdk-logs'),save_config_filepath=str(OUT/f'{STAGE}-final-config.yaml')),
        checkpoint=CheckpointConfig(save=str(OUT/'checkpoint'),
            load=None,pretrained_checkpoint=None,save_interval=interval,
            save_optim=True,save_rng=True,load_optim=False,load_rng=False,finetune=False,
            exit_on_missing_checkpoint=True,async_save=False,fully_parallel_save=False,ckpt_format='torch_dist',
            save_tokenizer_assets=False,verify_integrity=False,most_recent_k=3),
        validation=ValidationConfig(eval_iters=1,eval_interval=250,eval_global_batch_size=8),
        rng=RNGConfig(seed=C['seed']),mixed_precision='bf16_mixed')
    runtime_config_update(cfg)
    if RANK==0:cfg.to_yaml(str(OUT/f'{STAGE}-config.yaml'))
    common,seen=audited_common()
    def adapter(state,iterator,model,return_schedule_plan=False):
        return forward_step(common,state,iterator,model,return_schedule_plan)
    updates=[];saves=[];timer={};held=[]
    class Audit(Callback):
        def on_data_init_start(self,ctx):
            assert int(ctx.state.train_state.step)==start
            assert int(ctx.state.train_state.consumed_train_samples)==start*128
        def on_train_start(self,ctx):
            kd.check_provider_shapes(unwrap_model(ctx.model[0]).config)
            # Inspect actual native optimizer groups before deciding on the TE handle-pool cap.
            counts=[len(g['params']) for g in ctx.optimizer.param_groups]
            if max(counts,default=0)*4>20000:
                from optimizer_compat import install_and_verify
                compatibility=install_and_verify()
            else:compatibility=dict(installed=False,reason='EP8 native groups within TE handle pool')
            publish(f'{STAGE}-rank{RANK}-loaded',dict(optimizer_group_tensors=counts,compatibility=compatibility,
                hf_import=getattr(provider,'_nemochat_hf_import_receipt',{}),teacher_present=False))
        def on_train_step_start(self,ctx):
            logical=int(ctx.state.train_state.step)
            physical=REPLAY.begin(logical)
            torch.cuda.synchronize();timer.update(start=time.perf_counter(),lr=float(ctx.optimizer.param_groups[0]['lr']),physical=physical)
        def on_train_step_end(self,ctx):
            torch.cuda.synchronize();step=int(ctx.state.train_state.step)+1
            local=seen['masked_tokens'];seen['masked_tokens']=0
            counts=torch.tensor([local],device='cuda',dtype=torch.int64);dist.all_reduce(counts)
            row=dict(rank=RANK,update=step,metrics={str(k):float(v) for k,v in ctx.loss_dict.items()},
                grad_norm=None if ctx.grad_norm is None else float(ctx.grad_norm),lr_used=timer['lr'],
                skipped=bool(ctx.skipped_iter),seconds=time.perf_counter()-timer['start'],
                local_masked_tokens=local,allrank_masked_tokens=int(counts.item()),
                logical_iteration=step-1,physical_iteration=timer['physical'],
                global_batch_size=128,sequence_length=4096,peak_memory_bytes=torch.cuda.max_memory_allocated())
            with (OUT/f'{STAGE}-rank{RANK}-updates.jsonl').open('a') as s:s.write(json.dumps(row)+'\n')
            updates.append(row)
            if row['skipped'] or row['grad_norm'] is None or row['grad_norm']<=0 or row['lr_used']<0 or not all(
                math.isfinite(v) for v in [*row['metrics'].values(),row['grad_norm'],row['lr_used']]):
                raise FloatingPointError('actual nonfinite/zero-gradient/skipped update')
            assert abs(row['metrics']['total loss']-row['metrics']['lm loss'])<1e-6
            assert row['allrank_masked_tokens']==128*4096
            REPLAY.complete()
            assert REPLAY.calls==step
            if RANK==0:
                durable(OUT/'replay-state.json',REPLAY.snapshot())
                publish('student-progress',dict(stage=STAGE,update=step,total=STOP,schedule_horizon=C['train_iters'],physical=timer['physical'],replay_only=True))
        def on_checkpoint_save(self,ctx):
            step=int(ctx.state.train_state.step)
            b=binding_for_save(REPLAY,step,hashlib.sha256((OUT/'replay-manifest.json').read_bytes()).hexdigest(),OUT/'checkpoint'/f'iter_{step:07d}')
            assert REPLAY.calls==step
            if step==16:ctx.state.cfg.checkpoint.save_interval=64
            checkpoint=OUT/'checkpoint'/f'iter_{step:07d}'
            assert checkpoint.is_dir() and any(checkpoint.rglob('*.distcp'))
            if RANK==0:durable(OUT/f'replay-checkpoint-step{step}.json',REPLAY.snapshot())
            assert b['next_iteration']==step
            # Native callback is after synchronous save. Allrank acknowledgement
            # precedes the only atomic host-readable cursor commit.
            ranks=[None]*8;dist.all_gather_object(ranks,dict(rank=RANK,step=step,binding=b))
            assert all(r['step']==step and r['binding']==b for r in ranks)
            if RANK==0:
                publish(f'{STAGE}-checkpoint-step{step}-transaction',dict(binding=b,ranks=ranks,
                    native_synchronous_save_completed=True,save_optim=True,save_rng=True,
                    checkpoint_integrity=C['checkpoint_integrity']))
                publish(f'checkpoint-step{step}-transaction',dict(binding=b,ranks=ranks,
                    native_synchronous_save_completed=True,save_optim=True,save_rng=True,
                    checkpoint_integrity=C['checkpoint_integrity']))
            saves.append(step);dist.barrier()
        def on_eval_end(self,ctx):
            held.append(int(ctx.state.train_state.step))
            publish(f'{STAGE}-rank{RANK}-heldout',dict(step=int(ctx.state.train_state.step),
                mode='native-disjoint-GPTDataset-LM-only',evaluations=len(held)))
    pretrain(cfg,adapter,callbacks=[Audit()])
    completed=REPLAY.calls
    assert [r['update'] for r in updates]==list(range(start+1,completed+1))
    assert completed==STOP
    if completed:assert completed in saves
    if RANK==0:durable(OUT/'final-replay-state.json',REPLAY.snapshot())
    publish(f'{STAGE}-rank{RANK}-complete',dict(rank=RANK,completed=completed,
        optimizer_calls=len(updates),positive_updates=sum(r['lr_used']>0 for r in updates),
        skipped=sum(r['skipped'] for r in updates),nonfinite=0,save_steps=saves,heldout_steps=held,
        operational_resume=False,restored_step=start,checkpoint_integrity=C['checkpoint_integrity']))


def main():
    global C,STAGE
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['student']);a=p.parse_args()
    STAGE=a.stage;C=json.loads((OUT/'run-config.json').read_text())
    student()

if __name__=='__main__':main()
