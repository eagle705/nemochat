"""TDD gates for superseding ordinary-LM B02; historical B01 remains untouched."""
import importlib
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/'midtraining/training/offline_kd/ablation_lm_only/ordinary02'

class OrdinaryLM(unittest.TestCase):
    def module(self,name):
        self.assertTrue((HERE/f'{name}.py').exists(),f'{name} implementation missing')
        return importlib.import_module('midtraining.training.offline_kd.ablation_lm_only.ordinary02.'+name)

    def test_lm_coefficient_one_and_actual_gradient_one(self):
        objective=self.module('objective')
        x=torch.tensor([2.,4.],requires_grad=True)
        def native(out):
            lm=out.sum();n=torch.tensor(2)
            return lm,n,{'lm loss':torch.stack((lm.detach().clone(),n))}
        total,count,report=objective.ordinary_lm_loss(native,x)
        self.assertEqual(objective.LM_COEFFICIENT,1.)
        self.assertEqual(float(total.detach()),6.)
        self.assertEqual(report['lm loss'].tolist(),[6.,2.])
        self.assertEqual(report['total loss'].tolist(),[6.,2.])
        total.backward();self.assertEqual(x.grad.tolist(),[1.,1.])
        total.detach().div_(16)
        self.assertEqual(report['lm loss'].tolist(),[6.,2.])
        self.assertEqual(report['total loss'].tolist(),[6.,2.])

    def test_replay_preserves_native_scalar_metadata_and_exact_sampler(self):
        replay_module=self.module('replay')
        import json
        c=ROOT/'results/midtraining/offline2b-prefixtrain-1000-01'
        mapping=[json.loads(line)['physical_iteration'] for line in (c/'student-rank0-updates.jsonl').read_text().splitlines()]
        reads=[]
        sample=dict(tokens=torch.arange(4096),labels=torch.arange(4096),loss_mask=torch.ones(4096),
                    position_ids=torch.arange(4096),dataset_id=np.int16(2))
        class Source:
            def __len__(self):return 468736
            def __getitem__(self,index):reads.append(index);return sample
        expected={m:dict(iteration=0,micro=m,rank=3,sha256=replay_module.digest(sample),masked_tokens=4096) for m in range(16)}
        gather=lambda row:[dict(row,rank=r) for r in range(8)]
        with tempfile.TemporaryDirectory(dir='/home/yoon/.hermes/cache/scratch') as d:
            replay=replay_module.Replay(Source(),mapping,expected,3,gather,d)
            replay.begin(0);ds=replay_module.ReplayDataset(replay)
            self.assertEqual(reads,[m*8+3 for m in range(16)])
            self.assertEqual(ds[3]['dataset_id'],np.int16(2))
            self.assertNotEqual(ds[3]['tokens'].data_ptr(),sample['tokens'].data_ptr())
            for micro in range(16):replay.actual(ds[micro*8+3],0,micro)
            replay.complete()
            self.assertEqual(replay.snapshot()['committed_mapping'],[0])
            self.assertEqual(replay.snapshot()['next_physical'],1)

    def test_actual_digest_error_is_collective_and_preserves_evidence(self):
        import json
        m=self.module('replay')
        c=ROOT/'results/midtraining/offline2b-prefixtrain-1000-01'
        mapping=[json.loads(line)['physical_iteration'] for line in (c/'student-rank0-updates.jsonl').read_text().splitlines()]
        class Source:
            def __len__(self):return 468736
        gather_calls=[]
        def gather(row):
            gather_calls.append(row)
            return [dict(row,rank=r) for r in range(8)]
        expected={0:dict(iteration=0,micro=0,rank=0,sha256='a'*64,masked_tokens=4096)}
        with tempfile.TemporaryDirectory(dir='/home/yoon/.hermes/cache/scratch') as d:
            replay=m.Replay(Source(),mapping,expected,0,gather,d)
            with self.assertRaises(RuntimeError):replay.actual(dict(tokens='read failure'),0,0)
            self.assertEqual(len(gather_calls),1)
            self.assertFalse(gather_calls[0]['ok'])
            self.assertIn('error_type',gather_calls[0])
            self.assertTrue((Path(d)/'actual-preforward-failure-rank0-logical0-micro0.json').exists())
            self.assertEqual(replay.calls,0);self.assertEqual(replay.used,0)

    def test_native_save_binding_requires_complete_all8_fullstate_files(self):
        from types import SimpleNamespace
        m=self.module('replay')
        self.assertTrue(hasattr(m,'binding_for_save'),'native save binding gate missing')
        state=dict(completed=16,committed_mapping=list(range(16)),next_physical=16)
        replay=SimpleNamespace(calls=16,samples=None,snapshot=lambda:state)
        with tempfile.TemporaryDirectory(dir='/home/yoon/.hermes/cache/scratch') as d:
            checkpoint=Path(d)/'iter_0000016';checkpoint.mkdir()
            with self.assertRaises(ValueError):m.binding_for_save(replay,16,'a'*64,checkpoint)
            (checkpoint/'.metadata').touch()
            for rank in range(8):(checkpoint/f'__{rank}_0.distcp').touch()
            binding=m.binding_for_save(replay,16,'a'*64,checkpoint)
            self.assertEqual(binding['next_iteration'],16)
            self.assertEqual(binding['consumed_samples'],2048)
            self.assertEqual(binding['replay_state'],state)
            with self.assertRaises(ValueError):m.binding_for_save(replay,17,'a'*64,checkpoint)

    def test_native_entrypoint_freezes_ordinary_recipe_and_early_save(self):
        self.assertTrue((HERE/'native.py').exists(),'ordinary native entrypoint missing')
        code=(HERE/'native.py').read_text()
        self.assertIn('pretrain(cfg,adapter,callbacks=[Audit()])',code)
        self.assertIn('lm_coefficient=1.0',code)
        self.assertIn('binding_for_save(',code)
        self.assertIn('load_weights=True',code)
        self.assertIn('adam_beta2=.999',code)
        self.assertIn('most_recent_k=3',code)
        self.assertNotIn('CachedKD',code)
        self.assertNotIn('optimizer.step(',code)

if __name__=='__main__':unittest.main()
