import sys, unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[6]))
from midtraining.training.offline_kd.prefix.dynamic.attempt02.reader import DynamicDataset
class SurplusTest(unittest.TestCase):
    def test_native_blended_builder_surplus_is_not_a_shortened_budget(self):
        class FullNative:
            def __len__(self):return 468738
        dataset=DynamicDataset(FullNative(),None,0)
        self.assertEqual(len(dataset),468738)
    def test_direct_io_capability_or_code_error_is_fatal_not_quarantine(self):
        from midtraining.training.offline_kd.prefix.dynamic.attempt02 import reader
        import errno
        self.assertTrue(reader.quarantinable_transport(RuntimeError('size/SHA mismatch path=x')))
        self.assertTrue(reader.quarantinable_transport(RuntimeError('post-read stat/path mismatch')))
        self.assertTrue(reader.quarantinable_transport(OSError(errno.EIO,'read failed')))
        self.assertFalse(reader.quarantinable_transport(RuntimeError('statx DIO alignment unavailable; no fallback')))
        self.assertFalse(reader.quarantinable_transport(OSError(errno.EINVAL,'unsupported O_DIRECT')))
        self.assertFalse(reader.quarantinable_transport(RuntimeError('unexpected implementation bug')))
    def test_native_source_shorter_than_original_budget_is_fatal(self):
        class Short:
            def __len__(self):return 468735
        with self.assertRaises(ValueError):DynamicDataset(Short(),None,0)
if __name__=='__main__':unittest.main()
