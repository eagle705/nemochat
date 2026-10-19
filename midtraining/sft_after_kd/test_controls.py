"""CPU controls only; no fixtures are benchmark acceptance."""
import importlib.util
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent

def load():
    spec = importlib.util.spec_from_file_location('sft_controls', HERE/'controls.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

class Controls(unittest.TestCase):
    def test_source_and_deadline_are_immutable(self):
        self.assertTrue((HERE/'controls.py').exists(), 'C-only controls missing')
        m = load()
        self.assertEqual(m.DEADLINE.isoformat(), '2026-10-19T12:00:00+09:00')
        self.assertEqual(m.SOURCE, 'checkpoints/exports/nemochat_mid_super35_distill_hf_bf16_01')
        for source in ('BASE', 'checkpoints/exports/nemochat_mid_lmonly_hf_bf16_02'):
            with self.assertRaisesRegex(ValueError, 'C LM.KD1000'):
                m.validate_source(source)
        with self.assertRaisesRegex(ValueError, 'deadline'):
            m.budget(m.DEADLINE.timestamp(), [20.]*10, 1000)

class GateControls(unittest.TestCase):
    def test_parent_contract_and_scores_fail_closed(self):
        m = load()
        self.assertTrue(hasattr(m, 'accept_benchmark'), 'benchmark gate missing')
        with self.assertRaisesRegex(ValueError, 'parent-reviewed producer contract'):
            m.accept_benchmark({}, None)
        proof = dict(suite_sha256=m.SUITE_SHA, backend='gpu-official-transformers', clean_exit=True,
                     gpu_lock_released=True, models={})
        with self.assertRaisesRegex(ValueError, 'all three'):
            m.accept_benchmark(proof, {'reviewed': True})
        for key in ('A', 'C', 'B'):
            proof['models'][key] = dict(question_ids=list(range(400)), heldout_window_ids=list(range(64)),
                                       heldout_targets=65536, clean_exit=True, score=.25, heldout_nll=2.)
        self.assertEqual(m.accept_benchmark(proof, {'reviewed': True})['models'], ['A','C','B'])
        proof['models']['B']['question_ids'][1] = 0
        with self.assertRaisesRegex(ValueError, 'coverage'):
            m.accept_benchmark(proof, {'reviewed': True})

class ProcessControls(unittest.TestCase):
    def test_pidfd_real_exit_timeout_and_stale_identity(self):
        m = load()
        self.assertTrue(hasattr(m, 'wait_exact'), 'exact pidfd wait missing')
        import subprocess, sys
        child = subprocess.Popen([sys.executable,'-u','-c',
            "import sys; print('READY',flush=True); sys.stdin.readline()"],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
        try:
            self.assertEqual(child.stdout.readline().strip(),'READY')
            identity = m.process_identity(child.pid)
            with self.assertRaisesRegex(TimeoutError, 'prerequisite'):
                m.wait_exact(identity, .02)
            self.assertIsNone(child.poll())
            stale = dict(identity,starttime=identity['starttime']+1)
            with self.assertRaisesRegex(ValueError, 'identity'):
                m.wait_exact(stale,.02)
            child.stdin.write('exit\n');child.stdin.flush()
            import os
            os.waitid(os.P_PID,child.pid,os.WEXITED|os.WNOWAIT)
            m.wait_exact(identity,1.)
            child.wait(timeout=1)
        finally:
            if child.poll() is None:
                child.kill();child.wait()
            child.stdin.close();child.stdout.close()

class NativeControls(unittest.TestCase):
    def test_native_recipe_is_c_only_and_finite_pool(self):
        import importlib.util,sys,json
        p=HERE/'native_sft_v1.py'
        self.assertTrue(p.exists(), 'versioned C native adapter missing')
        spec=importlib.util.spec_from_file_location('sft_native_v1',p)
        native=importlib.util.module_from_spec(spec);sys.modules[spec.name]=native;spec.loader.exec_module(native)
        self.assertEqual(native.SOURCE,load().SOURCE)
        import yaml
        c=yaml.safe_load((HERE/'recipe-v1.yaml').read_text())
        manifest=json.loads((HERE.parents[1]/'datasets/nemotron_sft/nemochat_baby/prepared-full-03/data/manifest.json').read_text())
        plan=native.stage_plan(dict(c,max_updates=15,max_hours=.75),manifest)
        self.assertEqual(plan['global_batch_size'],128)
        self.assertEqual(plan['epochs'],1)
        self.assertEqual(plan['updates'],15)
        self.assertEqual(plan['available_updates'],34135)
        bad=dict(c,source_checkpoint='BASE')
        with self.assertRaisesRegex(ValueError,'LM.KD1000'):
            native.stage_plan(bad,manifest)
        self.assertEqual(native.next_update_lr(2,10,.5,1.,1e-5,1e-6),1e-6)
        self.assertEqual(native.next_update_lr(10,10,100.,1.,1e-5,1e-6),1e-6)
        self.assertEqual(native.next_update_lr(1,10,100.,1.,1e-5,1e-6),1e-5)

class PreviewControls(unittest.TestCase):
    def test_inert_preview_and_start_refusal(self):
        import subprocess,sys,json
        p=HERE/'controller.py'
        self.assertTrue(p.exists(), 'conditional controller preview missing')
        result=subprocess.run([sys.executable,str(p),'--preview'],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        preview=json.loads(result.stdout)
        self.assertEqual(preview['gpu_launches'],0)
        self.assertTrue(preview['producer_contract_bound'])
        self.assertIn('--property=Restart=no',preview['supervisor_argv'])
        result=subprocess.run(['bash',str(HERE/'start.sh'),'--start'],capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('stable SFT data completion proof missing',result.stderr)

    def test_budget_is_finite_and_single_epoch(self):
        m=load();plan=m.budget(m.DEADLINE.timestamp()-7200,[20.]*10,1000)
        self.assertEqual(plan['updates'],8)
        self.assertFalse(plan['allow_repetition'])
        with self.assertRaisesRegex(ValueError,'ten finite'):
            m.budget(m.DEADLINE.timestamp()-7200,[float('nan')]*10,1000)

class ProducerControls(unittest.TestCase):
    def test_missing_exact_gpu_marker_is_rejected(self):
        import tempfile,importlib.util
        p=HERE/'benchmark_gate.py'
        self.assertTrue(p.exists(),'actual GPU producer binding missing')
        spec=importlib.util.spec_from_file_location('sft_benchmark_gate',p)
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError,'exact GPU completion marker'):
                m.accept_actual(Path(tmp))

if __name__ == '__main__':
    unittest.main()
