"""CPU-only real tensors and sealed descriptors; never GPU learning evidence."""
import sys, unittest, hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import torch

class RowTests(unittest.TestCase):
    def fixture(self):
        import sealed_input as s
        tokens=np.arange(9,dtype='<u4');mask=np.array([0,1,0,1,1,0,1,0],dtype='u1')
        item=dict(tokens=torch.tensor(tokens[:-1].astype('int64')),labels=torch.tensor(tokens[1:].astype('int64')),loss_mask=torch.tensor(mask,dtype=torch.float32),position_ids=torch.arange(8),row_index=torch.tensor(2),source_index=torch.tensor(2),split='train',snapshot_id='fixture')
        return s,item,s.row_digest(2,tokens,mask)
    def test_exact_shifted_inputs_are_cloned_and_authenticated(self):
        s,item,digest=self.fixture()
        result=s.verify_item(item,digest,2,'train','fixture',8)
        self.assertTrue(torch.equal(result['tokens'],item['tokens']))
        self.assertNotEqual(result['tokens'].data_ptr(),item['tokens'].data_ptr())
        self.assertEqual(digest,hashlib.sha256(b'nemochat-sft-row-v1\0'+(2).to_bytes(8,'little')+np.arange(9,dtype='<u4').tobytes()+bytes([0,1,0,1,1,0,1,0])).digest())
    def test_wrong_row_shift_mask_position_dtype_role_snapshot_rejected(self):
        s,item,digest=self.fixture()
        changes=[('row_index',torch.tensor(3)),('tokens',item['tokens']+1),('labels',item['labels']+1),('loss_mask',item['loss_mask']+.5),('position_ids',torch.arange(8)+1),('tokens',item['tokens'].float()),('split','heldout'),('snapshot_id','wrong')]
        for key,value in changes:
            with self.subTest(key=key),self.assertRaises((ValueError,PermissionError)):
                s.verify_item(dict(item,**{key:value}),digest,2,'train','fixture',8)
    def test_descriptor_seals_access_header_and_lifetime(self):
        import sealed_input as s
        import fcntl, io, mmap, os, types
        from unittest.mock import patch
        # Public client fixture lives only in test module. Production has no stub.
        objects={};manifest=dict(train_dialogues=2,heldout_dialogues=2,sequence_length=8)
        for split in ('train','heldout'):
            for fid,value,header in [(f'sft:{split}-tokens.npy',np.arange(18,dtype='<u4').reshape(2,9),True),(f'sft:{split}-masks.npy',np.zeros((2,8),dtype='u1'),True),('rowhash:'+split,np.zeros((2,32),dtype='u1'),False)]:
                buffer=io.BytesIO()
                if header:np.save(buffer,value);raw=buffer.getvalue()
                else:raw=value.tobytes()
                fd=os.memfd_create(fid,os.MFD_ALLOW_SEALING);os.write(fd,raw);fcntl.fcntl(fd,fcntl.F_ADD_SEALS,15)
                ro=os.open('/proc/self/fd/'+str(fd),os.O_RDONLY);os.close(fd)
                a=value.copy();a.flags.writeable=False
                objects[fid]=types.SimpleNamespace(fd=ro,array=a,close=lambda:None)
        fid='sft:heldout-dialogues.jsonl';fd=os.memfd_create(fid,os.MFD_ALLOW_SEALING);os.write(fd,b'{}\n');fcntl.fcntl(fd,fcntl.F_ADD_SEALS,15)
        ro=os.open('/proc/self/fd/'+str(fd),os.O_RDONLY);os.close(fd);objects[fid]=types.SimpleNamespace(fd=ro,array=None,close=lambda:None)
        ready={'snapshot_id':'fixture','files':{fid:{'bytes':os.fstat(obj.fd).st_size} for fid,obj in objects.items()}}
        stub=types.ModuleType('midtraining.sft_after_kd.sealed_snapshot03.client');stub.open_snapshot=lambda ready,ids:objects
        try:
            with patch.dict(sys.modules,{stub.__name__:stub}):
                snapshot=s.Snapshot(ready,manifest)
                self.assertIs(snapshot.opened['sft:train-tokens.npy'],objects['sft:train-tokens.npy'])
                objects['sft:train-tokens.npy'].array=np.zeros((2,8),dtype='<u4')
                with self.assertRaises(PermissionError):s.Snapshot(ready,manifest)
                ready['files']['sft:train-tokens.npy']['bytes']+=1
                with self.assertRaises(PermissionError):s.Snapshot(ready,manifest)
        finally:
            for obj in objects.values():os.close(obj.fd)

    def test_guard_catches_loader_exception_before_forward(self):
        import sealed_input as s
        called=[]
        def broken():raise OSError('source read fault')
        with self.assertRaises(RuntimeError):s.collective_read(broken,lambda item:item,lambda ok:called.append(ok) or ok)
        self.assertEqual(called,[False])
    def test_other_rank_failure_suppresses_local_good_forward(self):
        import sealed_input as s
        with self.assertRaises(RuntimeError):s.collective_read(lambda:3,lambda item:item,lambda ok:False)

if __name__=='__main__':unittest.main()
