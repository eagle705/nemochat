"""Filtered global-batch cursor contract; no torch/GPU needed."""
import importlib
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

def test_global_gap_dataset_and_cursor_contract():
    m=importlib.import_module('midtraining.training.offline_kd.prefix.filtered.mapping')
    p=m.plan()
    assert len(p['logical_to_physical'])==1138
    assert p['logical_to_physical'][85:88]==[85,87,88]
    assert p['logical_to_physical'][-1]==1142
    assert p['physical_to_logical'][86] is None
    assert len(set(p['logical_to_physical']))==1138
    ds=m.FilteredDataset(range(468736))
    assert len(ds)==1138*128
    for logical,physical in enumerate(p['logical_to_physical']):
        for offset in range(128): assert ds[logical*128+offset]==physical*128+offset
    assert m.cursor(86)['next_physical_iteration']==87
    assert m.cursor(1138)['next_physical_iteration']==1143
    assert m.cursor(128)['next_physical_iteration']==133
    m.validate_cursor(m.cursor(128),128)
    import pytest
    wrong=m.cursor(128);wrong['next_physical_iteration']=128
    with pytest.raises(ValueError):m.validate_cursor(wrong,128)
    with pytest.raises(IndexError): ds[len(ds)]

def test_reader_sparse_physical_selection_and_decode_mapping():
    from midtraining.training.offline_kd.prefix.filtered.reader import select_urls,remap_decoded
    from midtraining.training.offline_kd.prefix.filtered.mapping import PHYSICAL
    urls=[f'/cache/cp0_dp0__{i}.tar' for i in range(1143)]
    chosen=select_urls(urls,set(),86)
    assert chosen[0]=='/cache/cp0_dp0__87.tar'
    assert len(chosen)==1138-86
    v,ind=object(),object()
    for logical,physical in enumerate(PHYSICAL):
        assert remap_decoded((physical,v,ind))==(logical,v,ind)
    import pytest
    with pytest.raises(ValueError):remap_decoded((86,v,ind))
    assert select_urls(urls,set(urls),0)==[]

def test_transport_new_sha_failure_has_observed_evidence():
    from midtraining.training.offline_kd.prefix.filtered.transport import DirectTransport,expectation
    import tempfile,hashlib,pytest
    with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'results/midtraining/offline2b-prefixtrain-filtered-01') as d:
        f=Path(d)/'cp0_dp0__0.tar';f.write_bytes(b'a'*4096)
        t=DirectTransport([expectation(f,'0'*64)])
        with pytest.raises(RuntimeError) as error:t.read_bytes(str(f))
        assert hashlib.sha256(b'a'*4096).hexdigest() in str(error.value)
        assert '4096' in str(error.value)
