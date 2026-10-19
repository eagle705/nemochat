import importlib
import os
import hashlib
import numpy as np
from sft.archive.sealed_snapshot03 import holder as old


def recovery():
    name='sft.archive.snapshot.recovery02.holder'
    try:return importlib.import_module(name)
    except ModuleNotFoundError:return old


def test_whole_chunk_fastpath_equivalence():
    m=recovery()
    assert hasattr(m,'vote_chunk'), 'missing optimized chunk majority'
    a=b'A'*4096+b'B'*4096+b'end'
    d=b'X'*4096+b'Y'*4096+b'xxx'
    mixed=b'A'*4096+b'Y'*4096+b'zzz'
    fixtures=[(a,a,a),(a,a,d),(a,d,a),(d,a,a),(a,d,mixed),
              (b'abc',b'def',b'ghi'),(b'',b'',b'')]
    for triple in fixtures:
        result,stats=m.vote_chunk(*triple)
        assert bytes(result)==bytes(old.vote_block(*triple)[0])
        assert stats['fastpath_chunks']+stats['fallback_chunks']==1
        if stats['fastpath_chunks']:
            assert not any(stats[k] for k in ('unanimous','majority','tie','changed'))


def test_parallel_catalog_identical_serial():
    m=recovery()
    assert hasattr(m,'derive_catalog_parallel'), 'missing bounded parallel catalog'
    rng=np.random.default_rng(123)
    tokens=rng.integers(0,131072,size=(17017,97),dtype=np.uint32)
    masks=rng.integers(0,2,size=(17017,96),dtype=np.uint8)
    oldfd,oldmeta=old.derive_catalog(tokens,masks,'fixture')
    newfd,newmeta=m.derive_catalog_parallel(tokens,masks,'fixture')
    try:
        assert oldmeta==newmeta
        data=os.pread(newfd,newmeta['bytes'],0)
        assert data==os.pread(oldfd,oldmeta['bytes'],0)
        assert len(data)==len(tokens)*32
        assert all(data[i:i+32]!=bytes(32) for i in range(0,len(data),32))
        assert newmeta['row_domain_hex']==b'nemochat-sft-row-v1\0'.hex()
    finally:os.close(oldfd);os.close(newfd)
