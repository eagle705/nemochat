"""Bounded sparse frozen-teacher regeneration using unchanged native writer hooks.

Only an explicit parent-authored repair-request.json can select iterations.
All ranks replay the same selected global iterations (EP8 collectives).
Original cache, source and sample receipts remain read-only.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
REPO = Path('/raid/yoon/nemochat')
SOURCE = REPO/'results/midtraining/offline2b-production1920-01'
OUT = Path('/output')
RANK = int(os.environ.get('RANK', '0'))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO/'midtraining/offline2b'))
from repair_contract import repair_plan, attach_native_writer


def main():
    import torch
    import torch.distributed as dist
    from megatron.bridge.models.conversion.auto_bridge import AutoBridge
    from megatron.bridge.training.cached_kd.logits_saver import LogitsSaverHooks
    from megatron.bridge.training.cached_kd.runtime import bind_runtime
    from megatron.bridge.training.tokenizers.tokenizer import build_tokenizer
    from megatron.bridge.data.utils import pretrain_train_valid_test_datasets_provider
    from megatron.bridge.data.samplers import build_pretraining_data_loader
    from megatron.core import parallel_state
    spec = importlib.util.spec_from_file_location('original_native', REPO/'midtraining/offline2b/native.py')
    original = importlib.util.module_from_spec(spec); spec.loader.exec_module(original)
    original.C = json.loads((SOURCE/'run-config.json').read_text())
    original.OUT = OUT
    request = json.loads((OUT/'repair-request.json').read_text())
    plan = repair_plan(request['iterations'])
    assert request['cache_id'] == original.cache_id()
    assert original.dataset_budget() == [468736, 120, 8]
    expected = {r['micro']: r for r in map(json.loads, (SOURCE/f'teacher-rank{RANK}-samples.jsonl').read_text().splitlines())}
    kd = original.kd_helpers()
    if RANK == 0: original.publish('repair-tokenizer', kd.tokenizer_gate(REPO/original.STUDENT, REPO/original.TEACHER))
    provider = AutoBridge.from_hf_pretrained(str(REPO/original.TEACHER), trust_remote_code=False,
        local_files_only=True).to_megatron_provider(load_weights=True)
    kd.configure_provider(provider, dict(parallel=original.PARALLEL, sequence_length=4096))
    provider.hidden_dropout = 0.; provider.attention_dropout = 0.; provider.finalize()
    model = provider.provide_distributed_model(wrap_with_ddp=False, bf16=True)[0]
    model.requires_grad_(False); model.train()
    assert parallel_state.get_data_parallel_world_size() == 8
    assert parallel_state.get_expert_model_parallel_world_size() == 8
    assert all(p.dtype == torch.bfloat16 for p in model.parameters())
    original.publish(f'repair-rank{RANK}-loaded', dict(frozen=kd.check_teacher(model), optimizer=False, dropout=0))
    dataset = original.dataset_config(); dataset.tokenizer = build_tokenizer(original.tokenizer_config()); dataset.finalize()
    train, _, _ = pretrain_train_valid_test_datasets_provider(original.dataset_budget(), dataset)
    original.publish(f'repair-rank{RANK}-dataset', dict(budget=original.dataset_budget(), native_dataset=type(train).__name__))
    completed = []
    for row in plan:
        iteration = row['iteration']
        state = SimpleNamespace(cfg=SimpleNamespace(model=provider),
            train_state=SimpleNamespace(step=iteration, consumed_train_samples=row['consumed_samples']))
        loader = build_pretraining_data_loader(train, consumed_samples=row['consumed_samples'], dataloader_type='single',
            micro_batch_size=1, num_workers=0, data_sharding=True, persistent_workers=False,
            data_parallel_rank=parallel_state.get_data_parallel_rank(), data_parallel_size=8)
        iterator = iter(loader)
        cache = OUT/'repair-cache'/f'iteration-{iteration}'
        runtime = original.runtime(state)
        with bind_runtime(runtime):
            saver = LogitsSaverHooks(str(cache), k=32, p=None, min_k=1, save_dtype='fp16')
            from megatron.core.utils import unwrap_model
            attach_native_writer(saver, model, unwrap_model)
            try:
                for micro in range(16):
                    batch = next(iterator); h = hashlib.sha256()
                    for key in ('tokens', 'labels', 'loss_mask', 'position_ids'):
                        h.update(batch[key].cpu().contiguous().numpy().tobytes())
                    sample = dict(micro=iteration*16+micro, iteration=iteration, rank=RANK,
                        sha256=h.hexdigest(), masked_tokens=int(batch['loss_mask'].sum()))
                    if sample != expected[sample['micro']]:
                        raise ValueError('repair native sampler bytes differ from original teacher')
                    with (OUT/f'repair-rank{RANK}-samples.jsonl').open('a') as stream:
                        stream.write(json.dumps(sample)+'\n')
                    gpu = {k: (v.cuda(non_blocking=True) if v is not None else None) for k, v in batch.items()}
                    with torch.no_grad():
                        model(input_ids=gpu['tokens'], position_ids=gpu['position_ids'], attention_mask=None, labels=gpu['labels'])
                torch.cuda.synchronize()
                path, writes, meta, msc = saver.take_pending_data()
                saver._write_batched_tar(path, writes, meta, msc)
            finally:
                saver.remove_hooks()
        # Exercise the unchanged native tar reader and tensor decoder before readiness.
        from megatron.bridge.training.cached_kd.utils_logits import iter_logprobs_tar_entries, decode_logprobs_payload
        shard = cache/f'cp0_dp{RANK}__{iteration}.tar'
        import tarfile
        with tarfile.open(SOURCE/'cache'/shard.name, 'r:') as prior, tarfile.open(shard, 'r:') as fresh:
            assert prior.extractfile('_meta.json').read() == fresh.extractfile('_meta.json').read()
        entries = list(iter_logprobs_tar_entries(str(shard), start_iteration=iteration))
        assert len(entries) == 1 and entries[0].iteration == iteration
        values, indices = decode_logprobs_payload(entries[0].data)
        assert len(values) == len(indices) == 16
        for value, index in zip(values, indices):
            assert value.shape == index.shape == (4096, 1, 32)
            assert torch.isfinite(value).all() and (index >= 0).all() and (index < 131072).all()
        comparison = dict(rank=RANK, iteration=iteration, original_decodable=RANK != 7)
        if RANK != 7:
            prior_entries = list(iter_logprobs_tar_entries(str(SOURCE/'cache'/shard.name), start_iteration=iteration))
            old_values, old_indices = decode_logprobs_payload(prior_entries[0].data)
            assert len(old_values) == len(old_indices) == 16
            assert all(torch.isfinite(value).all() for value in old_values)
            comparison.update(values_bitwise_equal=all(torch.equal(a,b) for a,b in zip(values,old_values)),
                indices_bitwise_equal=all(torch.equal(a,b) for a,b in zip(indices,old_indices)),
                same_position_max_abs_logprob_difference=max(float((a.float()-b.float()).abs().max()) for a,b in zip(values,old_values)),
                index_position_agreement=sum(int((a==b).sum()) for a,b in zip(indices,old_indices))/sum(a.numel() for a in indices),
                original_and_regenerated_finite=True)
        original.publish(f'repair-rank{RANK}-comparison', comparison)
        kd.check_teacher(model); dist.barrier()
        completed.append(iteration)
        original.publish(f'repair-rank{RANK}-progress', dict(iterations=completed, optimizer_calls=0,
            all_microbatch_digests_match=True, cache_id=original.cache_id()))
    original.publish(f'repair-rank{RANK}-complete', dict(iterations=completed, optimizer_calls=0,
        frozen=kd.check_teacher(model), all_microbatch_digests_match=True, cache_id=original.cache_id()))
    dist.barrier(); dist.destroy_process_group()


if __name__ == '__main__':
    main()
