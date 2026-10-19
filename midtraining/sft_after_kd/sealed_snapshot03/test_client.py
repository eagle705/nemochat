import hashlib
import struct
import unittest
import numpy as np

class RowDigestTests(unittest.TestCase):
    def test_exact_protocol_including_actual_nul(self):
        from midtraining.sft_after_kd.sealed_snapshot03.client import row_digest
        tokens = np.array([1, 2, 131071], dtype='<u4')
        mask = np.array([0, 1], dtype='u1')
        expected = hashlib.sha256(b'nemochat-sft-row-v1\0' + struct.pack('<Q', 7) + tokens.tobytes() + mask.tobytes()).digest()
        self.assertEqual(row_digest(7, tokens, mask), expected)
        self.assertNotEqual(row_digest(8, tokens, mask), expected)

class DescriptorTests(unittest.TestCase):
    def test_unsealed_and_readwrite_rejected(self):
        import os, fcntl
        from midtraining.sft_after_kd.sealed_snapshot03.client import validate_descriptor
        fd = os.memfd_create('test',os.MFD_ALLOW_SEALING)
        try:
            os.write(fd,b'abc')
            meta = dict(bytes=3,seals=15,access_mode='O_RDONLY')
            with self.assertRaisesRegex(ValueError,'seals'): validate_descriptor(fd,meta)
            fcntl.fcntl(fd,1033,15)
            with self.assertRaisesRegex(ValueError,'access_mode'): validate_descriptor(fd,meta)
        finally: os.close(fd)
    def test_catalog_sha_checked(self):
        import os, fcntl
        from midtraining.sft_after_kd.sealed_snapshot03.client import OpenedSnapshotFile
        fd = os.memfd_create('catalog',os.MFD_ALLOW_SEALING)
        try:
            os.write(fd,b'X'*32); fcntl.fcntl(fd,1033,15)
            ro = os.open('/proc/self/fd/'+str(fd),os.O_RDONLY)
            try:
                with self.assertRaisesRegex(ValueError,'catalog_sha256'):
                    OpenedSnapshotFile(ro,dict(file_id='rowhash:train',bytes=32,sha256='0'*64,seals=15,access_mode='O_RDONLY',shape=[1,32],dtype='uint8'))
            finally: os.close(ro)
        finally: os.close(fd)
    def test_stale_socket_refused(self):
        from midtraining.sft_after_kd.sealed_snapshot03.client import open_snapshot
        ready = dict(schema='nemochat-sealed-full03-v1',status='accepted_sealed_snapshot_full03',snapshot_id='fixture',sft_manifest_sha256='x',c_manifest_sha256='y',holder_identity={},producer_source_freeze_sha256='z',socket_path='/nonexistent/sealed03/socket',files={'sft:x':{'file_id':'sft:x'}})
        with self.assertRaises(FileNotFoundError): open_snapshot(ready,['sft:x'])

class ProtocolTests(unittest.TestCase):
    def exchange(self, fault=''):
        import array, fcntl, io, json, os, socket, tempfile, threading
        from midtraining.sft_after_kd.sealed_snapshot03.client import open_snapshot, request_header
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR'],prefix='s03t-') as directory:
            path=directory+'/s'
            # UNIX paths are bounded even when TMPDIR is long.
            self.assertLess(len(path.encode()),108)
            a=np.arange(6,dtype='<u4').reshape(2,3);b=io.BytesIO();np.save(b,a);data=b.getvalue()
            fd=os.memfd_create('fixture',os.MFD_ALLOW_SEALING);os.write(fd,data);fcntl.fcntl(fd,1033,15)
            ro=os.open('/proc/self/fd/'+str(fd),os.O_RDONLY)
            meta=dict(file_id='sft:fixture.npy',bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),seals=15,access_mode='O_RDONLY',shape=[2,3],dtype='uint32')
            ready=dict(schema='nemochat-sealed-full03-v1',status='accepted_sealed_snapshot_full03',snapshot_id='fixture',sft_manifest_sha256='x',c_manifest_sha256='y',holder_identity={'pid':123},producer_source_freeze_sha256='z',socket_path=path,files={'sft:fixture.npy':meta})
            server=socket.socket(socket.AF_UNIX,socket.SOCK_SEQPACKET);server.bind(path);server.listen(1)
            errors=[]
            def send():
                try:
                    with server.accept()[0] as peer:
                        request=json.loads(peer.recv(65536));reply=dict(request)
                        if fault=='header_identity':reply['snapshot_id']='altered'
                        rights=[ro,ro] if fault=='fd_count' else [ro]
                        if fault=='shape_dtype':reply['file']['shape']=[3,2]
                        if fault=='truncation':
                            peer.sendmsg([json.dumps(reply).encode()+b' '*100000],[(socket.SOL_SOCKET,socket.SCM_RIGHTS,array.array('i',rights))])
                        else:peer.sendmsg([json.dumps(reply).encode()],[(socket.SOL_SOCKET,socket.SCM_RIGHTS,array.array('i',rights))])
                except Exception as error:errors.append(error)
            worker=threading.Thread(target=send);worker.start()
            try:
                if fault:
                    with self.assertRaisesRegex(ValueError,fault):open_snapshot(ready,['sft:fixture.npy'])
                else:
                    obj=open_snapshot(ready,['sft:fixture.npy'])['sft:fixture.npy']
                    self.assertTrue(np.array_equal(obj.array,a));self.assertFalse(obj.array.flags.writeable)
                    obj.close();self.assertIsNone(obj.fd)
            finally:
                worker.join(timeout=5);server.close();os.close(ro);os.close(fd)
            self.assertFalse(worker.is_alive());self.assertFalse(errors)
    def test_real_descriptor_array(self):self.exchange()
    def test_wrong_host_header(self):self.exchange('header_identity')
    def test_ancillary_fd_count(self):self.exchange('fd_count')
    def test_truncated_payload(self):self.exchange('truncation')

if __name__ == '__main__':
    unittest.main()
