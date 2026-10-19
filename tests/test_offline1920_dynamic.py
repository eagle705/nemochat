"""CPU-only test faults; never inject or fabricate GPU cache corruption."""
import importlib.util
import pathlib
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor

ROOT = pathlib.Path(__file__).resolve().parents[1]

def module():
    path = ROOT/'midtraining/offline2b/prefix/dynamic/broker.py'
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location('dynamic_broker_test', path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m

class AllRanks:
    def __init__(self):
        self.barrier = threading.Barrier(8, timeout=20)
        self.rows = [None]*8
    def gather(self, rank, row):
        self.rows[rank] = row
        self.barrier.wait()
        result = list(self.rows)
        self.barrier.wait()
        return result

class DynamicTest(unittest.TestCase):
    def test_late_bad_shard_is_collectively_skipped_before_forward(self):
        m = module()
        self.assertIsNotNone(m, 'dynamic allrank broker is not implemented')
        group = AllRanks()
        with tempfile.TemporaryDirectory() as directory:
            def rank_run(rank):
                reads=[]; forwards=[]; lr=[]
                def reader(physical):
                    reads.append(physical)
                    if (rank,physical)==(7,3):
                        raise m.CacheRejected(dict(error='TEST_ONLY_SHA_REJECTION', expected_sha256='a'*64,
                            observed_sha256='b'*64, expected_bytes=1, observed_bytes=1))
                    return ('immutable-native-decode',rank,physical)
                broker=m.Broker(rank, reader, lambda row:group.gather(rank,row),
                    pathlib.Path(directory), limit=7, historical=(1,))
                broker.prime()
                while broker.current is not None:
                    physical=broker.begin(len(forwards))
                    forwards.append(physical)
                    lr.append(len(lr))
                    self.assertEqual(broker.take(), ('immutable-native-decode',rank,physical))
                    broker.complete()
                return forwards, reads, lr, broker.snapshot()
            with ThreadPoolExecutor(max_workers=8) as pool:
                results=list(pool.map(rank_run,range(8)))
            self.assertTrue(all(x[0]==[0,2,4,5,6] for x in results))
            self.assertTrue(all(x[1]==[0,2,3,4,5,6] for x in results))
            self.assertTrue(all(x[2]==list(range(5)) for x in results))
            self.assertEqual(len({str(x[3]) for x in results}),1)
            self.assertEqual(results[0][3]['optimizer_calls'],5)
            self.assertEqual(results[0][3]['next_unconsumed_physical'],7)
            self.assertEqual(results[0][3]['quarantined_physical'],[1,3])

class CursorTest(unittest.TestCase):
    def test_resume_replays_unconsumed_lookahead_not_scanned_cursor(self):
        m=module()
        with tempfile.TemporaryDirectory() as d:
            gather=lambda row:[dict(row,rank=r) for r in range(8)]
            b=m.Broker(0,lambda p:('decoded',p),gather,pathlib.Path(d),limit=8,historical=(3,))
            b.prime();b.begin(0);b.take();b.complete()
            state=b.snapshot()
            self.assertEqual(state['next_unconsumed_physical'],1)
            self.assertEqual(state['scanned_cursor'],3)
            restored=m.Broker(0,lambda p:('decoded',p),gather,pathlib.Path(d),limit=8,historical=(3,))
            restored.restore(state)
            self.assertEqual(restored.begin(1),1)
            self.assertEqual(restored.take(),('decoded',1))

    def test_suffix_exhaustion_and_no_data(self):
        m=module();gather=lambda row:[dict(row,rank=r) for r in range(8)]
        with tempfile.TemporaryDirectory() as d:
            b=m.Broker(0,lambda p:p,gather,pathlib.Path(d),limit=5,historical=(2,3,4))
            b.prime();b.begin(0);b.take();b.complete()
            self.assertTrue(b.is_last)
            b.begin(1);b.take();b.complete()
            self.assertIsNone(b.current)
            self.assertEqual(b.snapshot()['optimizer_calls'],2)
            n=m.Broker(0,lambda p:p,gather,pathlib.Path(d),limit=3,historical=(0,1,2))
            n.prime();self.assertIsNone(n.current)
            self.assertEqual(n.snapshot()['optimizer_calls'],0)

    def test_mapping_bug_is_fatal_not_quarantined(self):
        m=module();gather=lambda row:[dict(row,rank=r) for r in range(8)]
        with tempfile.TemporaryDirectory() as d:
            def bad(_):raise ValueError('teacher/student digest mismatch: mapping bug')
            b=m.Broker(0,bad,gather,pathlib.Path(d),limit=5,historical=())
            with self.assertRaisesRegex(RuntimeError,'nonquarantinable'):b.prime()
            self.assertEqual(b.exclusions,[])

class ReaderContractTest(unittest.TestCase):
    def test_dataset_maps_original_full_sampler_without_reloading(self):
        import sys
        sys.path.insert(0,str(ROOT))
        try:
            from midtraining.offline2b.prefix.dynamic.reader import DynamicDataset
        except ImportError:
            self.fail('verified synchronous native reader is not implemented')
        m=module();gather=lambda row:[dict(row,rank=r) for r in range(8)]
        class Full:
            def __len__(self):return 468736
            def __getitem__(self,i):raise AssertionError('must use already verified retained row')
        with tempfile.TemporaryDirectory() as d:
            b=m.Broker(2,lambda p:([],[],[('physical',p*128+i*8+2) for i in range(16)]),
                gather,pathlib.Path(d),limit=4,historical=(1,))
            b.prime();b.begin(0)
            ds=DynamicDataset(Full(),b,2)
            self.assertEqual(len(ds),468736)
            self.assertEqual(ds[2],('physical',2))
            b.take();b.complete();b.begin(1)
            self.assertEqual(ds[128+8+2],('physical',2*128+8+2))
            with self.assertRaises(IndexError):_ = ds[128+3]

def cpu_gloo_tracer():
    """Opt-in TEST ONLY: actual 8-process CPU DDP, never GPU cache injection."""
    import os,json
    import torch
    import torch.distributed as dist
    from datetime import timedelta
    dist.init_process_group('gloo',timeout=timedelta(seconds=90))
    rank=dist.get_rank();assert dist.get_world_size()==8
    m=module();out=pathlib.Path(os.environ['DYNAMIC_CPU_TEST_OUT'])
    def gather(row):
        rows=[None]*8;dist.all_gather_object(rows,row);return rows
    rejected=[]
    def reader(p):
        if (rank,p) in ((7,2),(3,5),(6,8),(2,9)):
            rejected.append(p)
            raise m.CacheRejected(dict(error='CPU_TEST_ONLY_REJECTION',expected_sha256='a'*64,observed_sha256='b'*64,expected_bytes=1,observed_bytes=1))
        return p
    b=m.Broker(rank,reader,gather,out,limit=10,historical=())
    model=torch.nn.parallel.DistributedDataParallel(torch.nn.Linear(1,1,bias=False))
    optimizer=torch.optim.SGD(model.parameters(),lr=0.01)
    schedule=torch.optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=0.9)
    b.prime();forward=[];lr=[];weights=[]
    while b.current is not None:
        p=b.begin(len(forward));assert p not in (2,5,8,9)
        value=b.take();assert value==p
        optimizer.zero_grad();loss=(model(torch.tensor([[float(p+1)]]))-1).square().sum()
        loss.backward();before=model.module.weight.detach().clone();lr.append(optimizer.param_groups[0]['lr'])
        optimizer.step();schedule.step()
        assert not torch.equal(before,model.module.weight.detach())
        forward.append(p);weights.append(float(model.module.weight.item()));b.complete()
    assert forward==[0,1,3,4,6,7] and b.calls==6 and schedule.last_epoch==6
    allstates=gather(dict(rank=rank,forward=forward,lr=lr,weights=weights,state=b.snapshot()))
    assert all(r['forward']==forward and r['lr']==lr and r['weights']==weights for r in allstates)
    torch.save(dict(model=model.module.state_dict(),optimizer=optimizer.state_dict(),scheduler=schedule.state_dict(),dynamic=b.snapshot()),out/f'cpu-test-rank{rank}-checkpoint.pt')
    dist.barrier()
    if rank==0:
        m.durable(out/'cpu-gloo-proof.json',dict(scope='TEST_ONLY_CPU_DDP_NOT_NATIVE_GPU',allrank=allstates,
            good_immediately_after_bad=True,no_invalid_forward=True,optimizer_lr_updates=6,allrank_agreement=True,
            suffix_exhausted=True,no_deadlock=True))
    dist.destroy_process_group()

if __name__=='__main__':
    import os
    if os.environ.get('DYNAMIC_CPU_GLOO')=='1':cpu_gloo_tracer()
    else:unittest.main()
