"""Restart generation contracts and real CPU broker/cursor tests."""
import ast
import importlib.util
import pathlib
import unittest

ROOT=pathlib.Path(__file__).resolve().parents[1]
HERE=ROOT/'midtraining/offline2b/prefix/dynamic/restart01'

class RestartContract(unittest.TestCase):
    def test_fresh_generation_native_entrypoint_checkpoint_and_retention(self):
        self.assertTrue((HERE/'native.py').exists(), 'new restart generation is missing')
        native=(HERE/'native.py').read_text()
        supervisor=(HERE/'supervisor.py').read_text()
        prepare=(HERE/'prepare.py').read_text()
        self.assertIn('dynamic.restart01.broker',native)
        self.assertIn('pretrain(cfg,adapter,callbacks=[Audit()])',native)
        self.assertIn('most_recent_k=3',native)
        self.assertIn('interval=16',native)
        self.assertIn('if step==16:ctx.state.cfg.checkpoint.save_interval=64',native)
        self.assertIn('offline2b-prefixtrain-dynamic-restart-01',supervisor)
        self.assertIn('offline2b-prefixtrain-dynamic-restart-01',prepare)
        self.assertNotIn('dynamic-01/attempt02',native+supervisor+prepare)
        self.assertNotIn("copytree(old/'inductor'",supervisor)
        self.assertIn("'HF_HOME=",supervisor)
        self.assertLess(native.index('physical=BROKER.begin(logical)'),native.index('def on_train_step_end'))
        self.assertIn('step=int(ctx.state.train_state.step)+1',native)
        self.assertIn('BROKER.complete()',native)
        self.assertIn('assert BROKER.calls==step and len(BROKER.mapping)==step',native)
        for p in HERE.glob('*.py'):ast.parse(p.read_text())

    def test_owned_broker_lookahead_committed_cursor_exclusion_and_exhaustion(self):
        self.assertTrue((HERE/'broker.py').exists(), 'new broker is missing')
        spec=importlib.util.spec_from_file_location('restart_broker_test',HERE/'broker.py')
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        import tempfile
        gather=lambda row:[dict(row,rank=r) for r in range(8)]
        with tempfile.TemporaryDirectory(dir='/home/yoon/.hermes/cache/scratch') as d:
            def reader(p):
                if p==3:raise m.CacheRejected(dict(error='CPU_TEST_ONLY_NEW_BAD'))
                return ('retained',p)
            b=m.Broker(0,reader,gather,pathlib.Path(d),limit=7,historical=(1,6))
            b.prime();self.assertEqual(b.begin(0),0);self.assertEqual(b.take(),('retained',0));b.complete()
            s=b.snapshot();self.assertEqual(s['committed_mapping'],[0])
            self.assertEqual(s['next_unconsumed_physical'],2)
            self.assertEqual(s['prefetched_unconsumed'],[2,4])
            self.assertEqual(s['quarantined_physical'],[1,3])
            calls=[]
            while b.current is not None:
                calls.append(b.begin(b.calls));b.take();b.complete()
            self.assertEqual(calls,[2,4,5])
            self.assertEqual(b.snapshot()['next_unconsumed_physical'],7)
            self.assertEqual(b.snapshot()['committed_mapping'],[0,2,4,5])
            self.assertEqual(b.snapshot()['quarantined_physical'],[1,3,6])

if __name__=='__main__':unittest.main()
