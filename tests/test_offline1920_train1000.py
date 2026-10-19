"""Fresh-1000 native generation regression contracts."""
import ast
import importlib.util
from pathlib import Path
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/'midtraining/training/offline_kd/prefix/dynamic/train1000'
class Train1000(unittest.TestCase):
    def test_bounded_cap_keeps_unconsumed_lookahead_and_history(self):
        self.assertTrue((HERE/'broker.py').exists(), 'fresh capped broker missing')
        spec=importlib.util.spec_from_file_location('train1000_broker',HERE/'broker.py')
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        gather=lambda row:[dict(row,rank=r) for r in range(8)]
        with tempfile.TemporaryDirectory(dir='/home/yoon/.hermes/cache/scratch') as d:
            reads=[]
            def reader(p):reads.append(p);return p
            b=m.Broker(0,reader,gather,Path(d),accepted_limit=32)
            b.prime()
            while not b.finished:
                b.begin(b.calls);b.take();b.complete()
            s=b.snapshot()
            self.assertEqual(b.calls,32)
            self.assertEqual(s['committed_mapping'],[p for p in range(34) if p not in (18,20)])
            self.assertEqual(s['next_unconsumed_physical'],34)
            self.assertEqual(s['quarantined_physical'],[18,20])
            self.assertLessEqual(max(reads),35)
            c=m.Broker(0,reader,gather,Path(d),accepted_limit=1000)
            c.restore(s)
            self.assertEqual(c.begin(32),34)
    def test_source_digest_rejection_records_indices_and_structural_errors_stay_fatal(self):
        import sys
        sys.path.insert(0,str(ROOT))
        from midtraining.training.offline_kd.prefix.dynamic.train1000 import reader,broker
        self.assertTrue(hasattr(reader,'validate_source_digest'),'pre-forward source digest policy missing')
        expected=dict(micro=18*16+3,iteration=18,rank=4,sha256='a'*64,masked_tokens=4096)
        with self.assertRaises(broker.CacheRejected) as rejected:
            reader.validate_source_digest(18,3,4,2332,expected,'b'*64,4096)
        o=rejected.exception.observation
        self.assertEqual((o['candidate_physical'],o['micro'],o['source_index']),(18,3,2332))
        self.assertEqual((o['expected_sha256'],o['observed_sha256']),('a'*64,'b'*64))
        with self.assertRaises(ValueError):
            reader.validate_source_digest(18,3,4,2333,expected,'b'*64,4096)
        with self.assertRaises(ValueError):
            reader.validate_source_digest(18,3,4,2332,dict(expected,rank=5),'b'*64,4096)
        reader.validate_source_digest(18,3,4,2332,expected,'a'*64,4096)

    def test_native_fresh_owned_final_save(self):
        self.assertTrue((HERE/'native.py').exists(),'fresh native missing')
        n=(HERE/'native.py').read_text();s=(HERE/'supervisor.py').read_text()
        self.assertIn('STOP=1000',n)
        self.assertIn('dynamic.train1000.broker',n)
        self.assertIn('accepted_limit=STOP',n)
        self.assertIn('assert BROKER.finished',n)
        self.assertNotIn('assert BROKER.current is None',n)
        self.assertIn('if completed:assert completed in saves',n)
        self.assertIn('most_recent_k=3',n)
        self.assertIn('if step==16:ctx.state.cfg.checkpoint.save_interval=64',n)
        self.assertIn('optimizer_reset=True,rng_reset=True,cursor_reset=True,start_step=0',n)
        self.assertIn('offline2b-prefixtrain-1000-01',s)
        self.assertNotIn("copytree(old/'inductor'",s)
        self.assertIn('NetworkMode=\'none\'',s)
        for p in HERE.glob('*.py'):ast.parse(p.read_text())
if __name__=='__main__':unittest.main()
