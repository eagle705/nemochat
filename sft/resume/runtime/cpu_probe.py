"""Pinned CPU control tests, not a claim of native model load/save."""
import argparse,ctypes,importlib,json,os,subprocess,sys,tempfile,time
from pathlib import Path
from types import SimpleNamespace as S
from unittest.mock import patch
import controller as c
import contract

def child(queue):
 queue.put({'tmpdir':tempfile.gettempdir(),'env':os.environ['TMPDIR']})
 queue._tls.connection.close()

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 import torch,torch.multiprocessing as mp,numpy as np
 assert not torch.cuda.is_available() and tempfile.gettempdir()==os.environ['TMPDIR']=='/nct'
 with mp.Manager() as manager:
  queue=manager.Queue();proc=mp.get_context('spawn').Process(target=child,args=(queue,));proc.start();proc.join(30);assert proc.exitcode==0
  result=queue.get(timeout=10);assert result=={'tmpdir':'/nct','env':'/nct'};queue._tls.connection.close()
 fixture=Path(tempfile.mkdtemp(prefix='exec-'));src=fixture/'probe.c';src.write_text('int value(void){return 37;}\nint main(void){return value()!=37;}\n')
 subprocess.run(['cc',str(src),'-o',str(fixture/'probe')],check=True);subprocess.run([str(fixture/'probe')],check=True)
 subprocess.run(['cc','-shared','-fPIC',str(src),'-o',str(fixture/'probe.so')],check=True);assert ctypes.CDLL(str(fixture/'probe.so')).value()==37
 subprocess.run([sys.executable,str(c.HERE/'native.py'),'--config',a.config,'--manifest',str(c.MANIFEST),'--output',str(c.OUT/'config-sdk03'),'--config-only'],check=True)
 from megatron.bridge.data.samplers import MegatronPretrainingSampler
 from sealed_input import rows_class
 import resume_sft
 n=4369280;seq=16384
 opened={}
 for split in ('train',):
  opened['sft:'+split+'-tokens.npy']=S(array=np.broadcast_to(np.zeros(1,dtype='<u4'),(n,seq+1)))
  opened['sft:'+split+'-masks.npy']=S(array=np.broadcast_to(np.zeros(1,dtype='u1'),(n,seq)))
 snapshot=S(opened=opened,datasets={})
 Rows=rows_class(resume_sft.PreparedRows,snapshot)
 rows=Rows(None,'train',34135*128,128,seed=42,epochs=1,sequence_length=seq)
 expected=np.random.default_rng(42).permutation(n)
 first=[]
 for rank in range(8):
  sampler=iter(MegatronPretrainingSampler(34135*128,115840,1,rank,8))
  indices=[next(sampler)[0] for _ in range(16)]
  assert indices==list(range(115840+rank,115968,8))
  assert [rows.expected_row(i) for i in indices]==expected[indices].tolist()
  first.append({'rank':rank,'first_source_index':indices[0],'first_original_row':rows.expected_row(indices[0])})
 train=importlib.import_module('megatron.bridge.training.train')
 # Call the real native cap-decision function; intercept only save I/O/log barrier.
 state=S(cfg=S(train=S(exit_signal_handler=False,exit_duration_in_mins=None,exit_interval=915),checkpoint=S(save='fixture-not-model',save_interval=250,non_persistent_save_interval=None)),train_state=S(step=915))
 calls=[]
 with patch.object(train,'save_checkpoint_and_time',lambda *args,**kw:calls.append(int(args[0].train_state.step))),patch.object(train,'barrier_and_log',lambda *a:None):
  assert train.checkpoint_and_decide_exit(state,[],None,None,0,None,None) is True
 assert calls==[915]
 state.train_state.step=914;calls=[]
 with patch.object(train,'save_checkpoint_and_time',lambda *args,**kw:calls.append(1)),patch.object(train,'check_nvrx_straggler_detection',lambda *a:False):
  state.nvrx_straggler_manager=None
  assert train.checkpoint_and_decide_exit(state,[],None,None,0,None,None) is False
 assert calls==[]
 # The real checkpoint metadata fixture was read by the no-device discovery, never weight payloads.
 text=(c.OUT/'discovery/train-state.txt').read_text();assert "tensor(905)" in text and 'tensor(115840)' in text
 proof=dict(status='PINNED_CPU_RESUME_CURSOR_CAP_MANAGER_CONFIG_PASS',first_unconsumed_by_rank=first,native_sampler_horizon=34135,native_sampler_consumed_samples=115840,seed=42,native_cap_decision_saved_step=calls or 915,cap_save_io_intercepted_control_fixture=True,actual_native_weights_load=False,actual_native_fullstate_save=False,manager_spawn_queue_pass=True,short_exec_tmpdir='/nct',gpu_requests=0)
 c.write(a.output/'proof.json',proof);print(json.dumps(proof,indent=2))
if __name__=='__main__':main()
