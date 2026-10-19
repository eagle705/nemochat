"""Regression coverage for the actual checkpoint SyncManager failure seam."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import controller as c


class ShortTmpdirTests(unittest.TestCase):
    def test_every_native_payload_sets_short_temp_before_python(self):
        with patch.object(c,'socket_bind',return_value=[]):
            for segment in ('config','calibration','reload','main'):
                p=c.container_payload(segment,c.OUT/'unused',300,segment=='config')
                env=dict(x.split('=',1) for x in p['Env'])
                self.assertEqual(env['TMPDIR'],'/nct')
                self.assertEqual(env['TMP'],'/nct')
                self.assertEqual(env['TEMP'],'/nct')
                opts=p['HostConfig']['Tmpfs']['/nct'].split(',')
                self.assertTrue({'rw','exec','nosuid','nodev','mode=1777','size=8g'}<=set(opts))
                self.assertNotIn('noexec',opts)
                self.assertTrue(p['HostConfig']['ReadonlyRootfs'])
                self.assertEqual(p['HostConfig']['CapDrop'],['ALL'])
                self.assertIn(segment+'-home',env['HOME'])
                for key in ('TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR','CUDA_CACHE_PATH'):
                    self.assertTrue(env[key].startswith(env['HOME']+'/'))
                if segment=='config':self.assertEqual(p['HostConfig']['DeviceRequests'],[])

    def test_cpu_integration_preserves_short_temp_and_fresh_caches(self):
        import integrate_snapshot as integration
        home=c.OUT/'test-cpu-home'
        env=dict(x.split('=',1) for x in c.process_env(home,1))
        self.assertEqual(env['TMPDIR'],'/nct')
        self.assertEqual(env['TMP'],'/nct')
        self.assertTrue(all(env[k].startswith(str(home)+'/') for k in ('TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR','CUDA_CACHE_PATH')))
        self.assertNotIn("'TMPDIR='+str(home/'tmp')",Path(integration.__file__).read_text())

    def test_real_short_unix_lifecycle_and_long_path_negative(self):
        # Host scratch only; no use or modification of global /tmp.
        with tempfile.TemporaryDirectory(dir='/home/yoon/.hermes/cache/scratch') as base:
            code='''import os,socket,tempfile\nassert tempfile.gettempdir()==os.environ['TMPDIR']\np=tempfile.mkdtemp(prefix='pymp-')+'/listener-test'\ns=socket.socket(socket.AF_UNIX);s.bind(p);s.listen();s.close();os.unlink(p);os.rmdir(os.path.dirname(p))\n'''
            env=dict(os.environ,TMPDIR=base)
            run=subprocess.run([sys.executable,'-c',code],env=env,capture_output=True,text=True)
            self.assertEqual(run.returncode,0,run.stderr)
            long=Path(base)/('x'*100);long.mkdir()
            run=subprocess.run([sys.executable,'-c',code],env=dict(env,TMPDIR=str(long)),capture_output=True,text=True)
            self.assertNotEqual(run.returncode,0)
            self.assertIn('AF_UNIX path too long',run.stderr)

if __name__=='__main__':unittest.main()
