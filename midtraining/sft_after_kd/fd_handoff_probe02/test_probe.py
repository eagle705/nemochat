"""Local controls only; never reacquire original dataset or launch Docker."""
import ast
import fcntl
import hashlib
import mmap
import os
import struct
import unittest
import probe

class Controls(unittest.TestCase):
    def test_consumer_compiles(self):
        compile(probe.CONSUMER, 'consumer.py', 'exec')

    def test_genuine_fixture(self):
        data = probe.fixture()
        self.assertEqual(data[:8], b'\x93NUMPY\x01\x00')
        length = struct.unpack('<H', data[8:10])[0]
        header = ast.literal_eval(data[10:10+length].decode().strip())
        self.assertEqual(header, {'descr': '<u4', 'fortran_order': False, 'shape': (3, 4)})
        self.assertEqual(struct.unpack('<12I', data[10+length:]), tuple(range(12)))

    def test_sealed_ro_descriptor(self):
        data = probe.fixture()
        fd = os.memfd_create('fd02-unit', os.MFD_ALLOW_SEALING)
        ro = None
        try:
            os.write(fd, data)
            with self.assertRaisesRegex(AssertionError, 'seals'):
                probe.validate_descriptor(fd, {'bytes': len(data)})
            self.assertEqual(probe.seal(fd)['mask'], 15)
            with self.assertRaisesRegex(AssertionError, 'access_mode'):
                probe.validate_descriptor(fd, {'bytes': len(data)})
            ro = os.open('/proc/self/fd/'+str(fd), os.O_RDONLY)
            probe.validate_descriptor(ro, {'bytes': len(data)})
            with mmap.mmap(ro, 0, access=mmap.ACCESS_READ) as mm:
                self.assertEqual(hashlib.sha256(mm).hexdigest(), hashlib.sha256(data).hexdigest())
                with self.assertRaises(TypeError):
                    mm[0] = 0
        finally:
            if ro is not None:
                os.close(ro)
            os.close(fd)

if __name__ == '__main__':
    unittest.main()
