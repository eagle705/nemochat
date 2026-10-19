import hashlib
import tempfile
import unittest
from pathlib import Path
import numpy as np
import verify

class Tests(unittest.TestCase):
    def test_full_stream_direct_eof_and_fail_closed_expected_hash(self):
        with tempfile.TemporaryDirectory(dir='/home/yoon/.hermes/cache/scratch') as d:
            p=Path(d)/'tokens.npy'
            a=np.array([[10,18,11],[0,29,131072]],dtype='<u4');np.save(p,a)
            expected={'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
            r=verify.check_file(p,expected,'tokens',Path(d),chunk=4096)
            self.assertTrue(r['bytes_accepted'])
            self.assertFalse(r['semantic_accepted'])
            self.assertEqual(r['audit']['reserved_counts']['18'],1)
            self.assertEqual(r['audit']['reserved_counts']['29'],1)
            self.assertEqual(r['audit']['out_of_bounds'],1)
            expected['sha256']='0'*64
            self.assertFalse(verify.check_file(p,expected,'tokens',Path(d),chunk=4096)['bytes_accepted'])

    def test_new_sft_aliases_are_measured_not_blanket_rejected(self):
        with tempfile.TemporaryDirectory(dir='/home/yoon/.hermes/cache/scratch') as d:
            p=Path(d)/'tokens.npy';np.save(p,np.array([[10,18,29,11]],dtype='<u4'))
            e={'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
            r=verify.check_file(p,e,'tokens',Path(d),chunk=4096)
            self.assertTrue(r['semantic_accepted'])
            self.assertEqual(r['audit']['reserved_counts']['18'],1)
            self.assertEqual(r['audit']['reserved_counts']['29'],1)

    def test_complete_aligned_multi_chunk_direct_stream(self):
        with tempfile.TemporaryDirectory(dir='/home/yoon/.hermes/cache/scratch') as d:
            p=Path(d)/'corpus.bin';p.write_bytes(np.arange(4096,dtype='<i4').tobytes())
            e={'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
            r=verify.check_file(p,e,'corpus',Path(d),chunk=4096)
            self.assertTrue(r['bytes_accepted']);self.assertFalse(r['semantic_accepted'])
            self.assertEqual(r['audit']['elements'],4096)

    def test_invalid_mask_and_short_npy_eof(self):
        with tempfile.TemporaryDirectory(dir='/home/yoon/.hermes/cache/scratch') as d:
            p=Path(d)/'masks.npy';np.save(p,np.array([[0,1,2]],dtype='u1'))
            e={'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
            r=verify.check_file(p,e,'masks',Path(d),chunk=4096)
            self.assertTrue(r['bytes_accepted']);self.assertFalse(r['semantic_accepted'])
            self.assertEqual(r['audit']['invalid_mask_values'],1)

if __name__=='__main__': unittest.main()
