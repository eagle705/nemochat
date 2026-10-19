"""Metadata-only exact mapping and explicit namespace refusal controls."""
import importlib.util, unittest
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.distributed.checkpoint as dcp
HERE=Path(__file__).parent
spec=importlib.util.spec_from_file_location('kd_converter',HERE/'convert_kd1000.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
class MappingControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.meta=dcp.FileSystemReader(c.SOURCE).read_metadata()
    def test_actual_metadata_and_wrapped_namespace(self):
        records,excluded=c.selected(self.meta)
        self.assertEqual(len(records),619);self.assertEqual(len(excluded),7860)
        for prefix in ['', 'module.student.']:
            metadata=SimpleNamespace(state_dict_metadata={prefix+k:v for k,v in records.items()})
            selected,_=c.selected(metadata,prefix)
            state={k.removeprefix(prefix):torch.empty(v.size,device='meta',dtype=v.properties.dtype) for k,v in selected.items()}
            converted,provenance=c.base.map_state(state)
            self.assertEqual(len(converted),447)
            self.assertEqual({s for values in provenance.values() for s in values},set(state))
    def test_unknown_namespace_is_fatal(self):
        metadata=SimpleNamespace(state_dict_metadata=dict(self.meta.state_dict_metadata))
        metadata.state_dict_metadata['teacher.weight']=next(iter(self.meta.state_dict_metadata.values()))
        with self.assertRaisesRegex(ValueError,'unclassified source key teacher.weight'):c.selected(metadata)
    def test_unmapped_model_key_is_fatal(self):
        records,_=c.selected(self.meta)
        state={k:torch.empty(v.size,device='meta',dtype=v.properties.dtype) for k,v in records.items()}
        state['decoder.unknown.weight']=torch.empty(1,device='meta')
        with self.assertRaisesRegex(ValueError,'unmapped native'):c.base.map_state(state)
if __name__=='__main__':unittest.main()
