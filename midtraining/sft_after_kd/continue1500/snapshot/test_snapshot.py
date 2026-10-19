import importlib.util
from pathlib import Path

def test_fast_holder_exists_and_rejects_inherited_mismatch():
    p=Path(__file__).with_name('holder.py')
    assert p.exists(), 'new bounded exact-byte holder is missing'
    spec=importlib.util.spec_from_file_location('continuation_holder',p)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    prior={'files':{'x':{'bytes':1,'sha256':'a'}}}
    assert m.inherit_identical(prior,{'x':{'bytes':1,'sha256':'a'}})
    try:m.inherit_identical(prior,{'x':{'bytes':1,'sha256':'b'}})
    except ValueError:pass
    else:raise AssertionError('mismatching bytes inherited')
