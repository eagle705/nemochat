"""Explicit logical optimizer to original physical global batch mapping."""
import hashlib
import json
EXCLUDED=(86,98,121,122,126)
PHYSICAL=tuple(i for i in range(1143) if i not in EXCLUDED)
INVERSE=tuple(None if i in EXCLUDED else PHYSICAL.index(i) for i in range(1143))
def plan():
    return dict(excluded_global_iterations=list(EXCLUDED),logical_to_physical=list(PHYSICAL),physical_to_logical=list(INVERSE),end_physical_iteration=1143,logical_updates=1138,kept_shards=9104,sampler_budget=[468736,120,8])
def identity():
    return hashlib.sha256(json.dumps(plan(),sort_keys=True).encode()).hexdigest()
def cursor(next_logical):
    if not 0<=next_logical<=len(PHYSICAL):raise ValueError('invalid logical cursor')
    return dict(filtered_mapping_sha256=identity(),next_logical_iteration=next_logical,next_physical_iteration=PHYSICAL[next_logical] if next_logical<len(PHYSICAL) else 1143)
def validate_cursor(binding,next_logical):
    if binding!=cursor(next_logical):raise ValueError('filtered resume cursor mismatch')
class FilteredDataset:
    def __init__(self,source):
        if len(source)<468736:raise ValueError('unchanged full native dataset budget required')
        self.source=source
    def __len__(self):return len(PHYSICAL)*128
    def __getitem__(self,i):
        if not isinstance(i,int) or not 0<=i<len(self):raise IndexError(i)
        return self.source[PHYSICAL[i//128]*128+i%128]
