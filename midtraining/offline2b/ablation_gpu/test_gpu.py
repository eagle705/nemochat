import importlib.util
from pathlib import Path
import unittest
HERE=Path(__file__).parent
class GPUContract(unittest.TestCase):
 def test_gpu_evaluator_contract(self):
  p=HERE/'eval.py'
  self.assertTrue(p.exists(),'GPU scorer missing')
  s=p.read_text()
  for token in ["device='cuda:0'","model.to(device)","device=device","torch.cuda.manual_seed_all(17)","allow_tf32=False","'loading_info':loading_metadata(info)","'gpu_uuid'","torch.isfinite(p).all()"]:self.assertIn(token,s)
  self.assertNotIn('assert not torch.cuda.is_available()',s)
  self.assertNotIn("device.type=='cpu'",s)
 def test_gpu_spec_and_clean_exit(self):
  import sys
  sys.path.insert(0,str(HERE))
  import controller as c
  spec=c.spec('A','smoke')
  self.assertEqual(spec['HostConfig']['DeviceRequests'][0]['DeviceIDs'],['0'])
  self.assertEqual(spec['HostConfig']['NetworkMode'],'none')
  self.assertNotIn('CUDA_VISIBLE_DEVICES=',spec['Env'])
  self.assertIn('CUDA_VISIBLE_DEVICES=0',spec['Env'])
  self.assertNotEqual(c.spec('A','smoke')['Env'][0],c.spec('A','full')['Env'][0])
  with self.assertRaises(ValueError):c.clean_engine({'Id':'x','State':{'Running':False,'ExitCode':1,'OOMKilled':False}},'x')
  c.clean_engine({'Id':'x','State':{'Running':False,'ExitCode':0,'OOMKilled':False}},'x')
 def test_loading_metadata_json_safe(self):
  import importlib.util,json
  sp=importlib.util.spec_from_file_location('gpu_eval',HERE/'eval.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
  self.assertEqual(json.loads(json.dumps(m.loading_metadata({'missing_keys':set(),'unexpected_keys':[],'nested':{'keys':{'a'}}}))),{'missing_keys':[],'unexpected_keys':[],'nested':{'keys':['a']}})
 def test_engine_normalized_gpu_scope(self):
  import controller as c
  expected=c.spec('C','full')['HostConfig']['DeviceRequests']
  actual=[dict(expected[0],Count=0,Options={})]
  c.validate_devices(actual,expected)
  actual[0]['DeviceIDs']=['7']
  with self.assertRaises(ValueError):c.validate_devices(actual,expected)
if __name__=='__main__':unittest.main()
