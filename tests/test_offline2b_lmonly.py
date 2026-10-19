"""Coefficient-matched objective and exact input replay regression gates."""
import json
from pathlib import Path
import tempfile
import unittest
import torch
from midtraining.offline2b.ablation_lm_only.objective import coefficient_matched_loss
from midtraining.offline2b.ablation_lm_only.replay import Replay, ReplayDataset, digest, validate_mapping

ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/'midtraining/offline2b/ablation_lm_only'
C=ROOT/'results/midtraining/offline2b-prefixtrain-1000-01'

class LMOnly(unittest.TestCase):
    def test_objective_half_gradient_and_independent_unweighted_reporting(self):
        x=torch.tensor([2.,4.],requires_grad=True)
        def native(out):
            loss=out.sum();count=torch.tensor(2,dtype=torch.int)
            return loss,count,{'lm loss':torch.stack((loss.detach().clone(),count))}
        total,count,metrics=coefficient_matched_loss(native,x)
        self.assertEqual(float(total),3.)
        self.assertEqual(metrics['lm loss'].tolist(),[6.,2.])
        self.assertEqual(metrics['total loss'].tolist(),[3.,2.])
        total.backward()
        self.assertEqual(x.grad.tolist(),[.5,.5])
        total.detach().div_(16)
        self.assertEqual(metrics['lm loss'].tolist(),[6.,2.])
        self.assertEqual(metrics['total loss'].tolist(),[3.,2.])
        with self.assertRaises(ValueError):coefficient_matched_loss(native,x,1.)

    def mapping(self):
        return [json.loads(x)['physical_iteration'] for x in (C/'student-rank0-updates.jsonl').read_text().splitlines()]

    def test_actual_c_mapping_allranks(self):
        mapping=self.mapping();validate_mapping(mapping)
        self.assertEqual(len(set(range(1091))-set(mapping)),91)
        for rank in range(8):
            rows=list(map(json.loads,(C/f'student-rank{rank}-updates.jsonl').read_text().splitlines()))
            self.assertEqual([r['physical_iteration'] for r in rows],mapping)
        with self.assertRaises(ValueError):validate_mapping(mapping[::-1])

    def test_preforward_failure_is_global_no_reselection(self):
        mapping=self.mapping();reads=[]
        sample=dict(tokens=torch.arange(4096),labels=torch.arange(4096),
                    loss_mask=torch.ones(4096),position_ids=torch.arange(4096))
        class Source:
            def __len__(self):return 468736
            def __getitem__(self,index):reads.append(index);return sample
        expected={m:dict(iteration=0,micro=m,rank=3,sha256=digest(sample),masked_tokens=4096) for m in range(16)}
        gathered=[]
        def gather(row):
            gathered.append(row)
            return [dict(row,rank=r,ok=(r!=5)) for r in range(8)]
        with tempfile.TemporaryDirectory(dir='/home/yoon/.hermes/cache/scratch') as d:
            replay=Replay(Source(),mapping,expected,3,gather,d)
            with self.assertRaises(RuntimeError):replay.begin(0)
            self.assertEqual(replay.calls,0);self.assertIsNone(replay.samples)
            self.assertEqual(reads,[m*8+3 for m in range(16)])
            self.assertTrue((Path(d)/'preforward-failure-rank3-logical0.json').exists())

    def test_snapshot_and_sampler_order(self):
        mapping=self.mapping()
        sample=dict(tokens=torch.arange(4096),labels=torch.arange(4096),
                    loss_mask=torch.ones(4096),position_ids=torch.arange(4096))
        class Source:
            def __len__(self):return 468736
            def __getitem__(self,index):return sample
        expected={m:dict(iteration=0,micro=m,rank=2,sha256=digest(sample),masked_tokens=4096) for m in range(16)}
        gather=lambda row:[dict(row,rank=r) for r in range(8)]
        with tempfile.TemporaryDirectory(dir='/home/yoon/.hermes/cache/scratch') as d:
            replay=Replay(Source(),mapping,expected,2,gather,d);ds=ReplayDataset(replay)
            replay.begin(0)
            self.assertNotEqual(ds[2]['tokens'].data_ptr(),sample['tokens'].data_ptr())
            with self.assertRaises(IndexError):ds[3]
            for m in range(16):replay.actual(ds[m*8+2],0,m)
            replay.complete()
            self.assertEqual(replay.snapshot()['committed_mapping'],[0])
            self.assertEqual(replay.snapshot()['next_physical'],1)

    def test_native_no_kd_or_custom_optimizer_loop(self):
        n=(HERE/'native.py').read_text();o=(HERE/'objective.py').read_text()
        self.assertIn('pretrain(cfg,adapter,callbacks=[Audit()])',n)
        self.assertNotIn('CachedKD',n)
        self.assertNotIn('optimizer.step(',n)
        self.assertNotIn('decode_logprobs',n)
        self.assertIn('LM_COEFFICIENT = 0.5',o)
        self.assertIn('masked_next_token_loss', (ROOT/'midtraining/offline2b/ablation_lm_only/README.md').read_text())
        for token in ['most_recent_k=3','save_optim=True,save_rng=True','if step==16:ctx.state.cfg.checkpoint.save_interval=64','lr_warmup_iters=C[\'warmup_iters\']']:
            self.assertIn(token,n)

if __name__=='__main__':unittest.main()
