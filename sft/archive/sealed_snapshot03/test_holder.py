import fcntl
import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from sft.archive.sealed_snapshot03.client import digest_fd

class CaptureTests(unittest.TestCase):
    def test_real_direct_fixture_sealed_readonly(self):
        from sft.archive.sealed_snapshot03.holder import capture
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as directory:
            p = Path(directory)/'fixture'; gold = bytes(range(256))*49 + b'last-short-eof'
            p.write_bytes(gold)
            e = {'bytes': len(gold), 'sha256': hashlib.sha256(gold).hexdigest()}
            report = {}
            fd = capture(p, e, report)
            try:
                self.assertEqual(digest_fd(fd,len(gold)),e['sha256'])
                self.assertEqual(fcntl.fcntl(fd,1034),15)
                self.assertEqual(fcntl.fcntl(fd,fcntl.F_GETFL)&os.O_ACCMODE,os.O_RDONLY)
                self.assertEqual(len(report['views']),1)
                with self.assertRaises(OSError): os.pwrite(fd,b'x',0)
            finally: os.close(fd)
    def test_vote_never_rebases_expected_sha(self):
        from sft.archive.sealed_snapshot03.holder import capture
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as directory:
            p = Path(directory)/'fixture'; p.write_bytes(b'x'*12299)
            report = {}
            with self.assertRaisesRegex(ValueError,'whole_sha256'):
                capture(p,{'bytes':12299,'sha256':'0'*64},report)
            self.assertEqual(len(report['views']),3)
            self.assertTrue(report['all_descriptors_closed_on_failure'])
    def test_observed_page_vote(self):
        from sft.archive.sealed_snapshot03.holder import vote_block
        a,b,c = b'a'*4096+b'z'*13,b'b'*4096+b'z'*13,b'b'*4096+b'y'*13
        self.assertEqual(vote_block(a,b,c)[0], b'b'*4096+b'z'*13)

class SemanticTests(unittest.TestCase):
    def test_catalog_exact_rows_and_readonly(self):
        import numpy as np
        from sft.archive.sealed_snapshot03.holder import derive_catalog
        from sft.archive.sealed_snapshot03.client import row_digest, OpenedSnapshotFile
        t=np.arange(12,dtype='<u4').reshape(3,4);m=np.zeros((3,3),dtype='u1');m[:,1]=1
        fd,meta=derive_catalog(t,m,'train')
        try:
            expected=b''.join(row_digest(i,t[i],m[i]) for i in range(3))
            self.assertEqual(os.pread(fd,96,0),expected)
            obj=OpenedSnapshotFile(os.dup(fd),meta)
            try:self.assertEqual(obj.array.shape,(3,32));self.assertFalse(obj.array.flags.writeable)
            finally:obj.close()
        finally:os.close(fd)
    def test_typed_index_bin_link_and_bad_pointer_rejection(self):
        import struct
        from sft.archive.sealed_snapshot03.holder import audit_index
        def check(pointer):
            data=b'MMIDIDX\0\0'+struct.pack('<QBQQ',1,4,2,3)+struct.pack('<2i',3,2)+struct.pack('<2q',0,pointer)+struct.pack('<3q',0,1,2)
            fd=os.memfd_create('index-fixture',os.MFD_ALLOW_SEALING)
            try:
                os.write(fd,data)
                return audit_index(fd,{'bytes':20,'sha256':'fixture'}, {'documents':2,'tokens':5})
            finally:os.close(fd)
        self.assertEqual(check(12)['dtype_code'],4)
        with self.assertRaisesRegex(ValueError,'idx_pointers'):check(13)
    def test_array_bounds_masks_and_targets_fail_closed(self):
        import numpy as np
        from types import SimpleNamespace
        from sft.archive.sealed_snapshot03.holder import audit_array
        with self.assertRaisesRegex(ValueError,'array_semantics'):audit_array(SimpleNamespace(array=np.array([[131072]],dtype='<u4')),'tokens',[1,1])
        with self.assertRaisesRegex(ValueError,'array_semantics'):audit_array(SimpleNamespace(array=np.array([[2]],dtype='u1')),'masks',[1,1],1)
        with self.assertRaisesRegex(ValueError,'target_count'):audit_array(SimpleNamespace(array=np.array([[0]],dtype='u1')),'masks',[1,1],1)

if __name__ == '__main__': unittest.main()
