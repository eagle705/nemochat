"""Independent approved 3662-call materializer contract."""
import importlib.util
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
def module():
    path = ROOT/'midtraining/data/offline_kd/materialize.py'
    assert path.is_file(), 'actual materializer required'
    s = importlib.util.spec_from_file_location('materializer_test',path)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m

def test_approved_budget_and_scarce_capacity():
    m = module(); b = m.budget()
    assert b['train_iters'] == 3662
    assert b['processed_tokens'] == 1919942656
    assert b['train_tokens'] == dict(code_concepts=1292808193,algorithmic=192966657,formal_logic=57901057,climbmix_replay=385921025)
    import json
    c = json.loads((ROOT/'results/midtraining/offline2b-corpus-01/capacity.json').read_text())
    assert m.capacity_check(c)['code_concepts'] == 262453
    c['sources']['code_concepts']['tokens'] = 1292808192
    import pytest
    with pytest.raises(ValueError,match='capacity'): m.capacity_check(c)

def test_actual_native_prefix_tail_filter_and_fresh_audit(tmp_path):
    import sqlite3, hashlib
    m = module(); rec = m.helpers(); p = rec.producer()
    db = sqlite3.connect(':memory:'); db.row_factory = sqlite3.Row
    db.execute('CREATE TABLE docs(id INTEGER,source TEXT,text TEXT,content_hash TEXT,split TEXT)')
    class Tok:
        def encode(self,text,**kw): return [18] if text == 'bad' else [100+ord(c) for c in text]
        def decode(self,ids,**kw): return ''.join(chr(x-100) for x in ids)
    tok = Tok(); readers = {}; cursors = {}; excluded = {}
    for j,k in enumerate(m.KEYS[:3]):
        readers[k] = [tok.encode(chr(97+j))+[11]]; cursors[k] = j*10
        for i,text in enumerate(['bad',chr(100+j)*3]):
            rid = j*10+i+1; sha = hashlib.sha256(text.encode()).hexdigest()
            db.execute('INSERT INTO docs VALUES(?,?,?,?,?)',(rid,k,text,sha,'train'))
            if text == 'bad': excluded[rid] = dict(content_hash=sha,token_sha256=p.token_digest([18,11]))
    receipts, counts = m.write_training(tmp_path,db,readers,cursors,
        [tok.encode('replay')+[11],[999,11]],tok,{k:5 for k in m.KEYS},
        set(),set(),lambda text:False,excluded,100,100,lambda **kw:None)
    assert counts['reserved_id_documents_excluded'] == 3
    assert receipts['code_concepts']['tokens'] == 6
    assert receipts['climbmix_replay']['tokens'] == 7
    assert m.fresh_train_audit(receipts,set(),lambda **kw:None)['training_duplicate_token_documents'] == 0
    readers['code_concepts'] = [[18,11]]
    import pytest
    with pytest.raises(ValueError,match='reserved'):
        m.write_training(tmp_path/'bad',db,readers,cursors,[],tok,{k:5 for k in m.KEYS},
            set(),set(),lambda text:False,excluded,100,100,lambda **kw:None)

def test_execution_refuses_unapproved_or_changed_recipe_before_writes():
    import pytest
    m = module()
    with pytest.raises(ValueError,match='approval'): m.execute({'approved':False})
    with pytest.raises(ValueError,match='budget'): m.validate_request({'approved':True,'budget':{}})

def test_fresh_audit_rejects_duplicate_and_heldout(tmp_path):
    import pytest
    m = module(); p = m.helpers().producer()
    w = p.NativeWriter(tmp_path/'one',11,131072)
    try:
        w.append([101,102,11]); entry = w.publish(); entry['truncated_documents'] = 0
    finally: w.close()
    with pytest.raises(ValueError,match='duplicate'):
        m.fresh_train_audit({'one':dict(entry),'two':dict(entry)},set(),lambda **kw:None)
    with pytest.raises(ValueError,match='heldout'):
        m.fresh_train_audit({'one':dict(entry)},{p.token_digest([101,102,11])},lambda **kw:None)

def test_whole_replay_mismatch_exclusion_does_not_rewrite(tmp_path):
    import sqlite3,pytest
    m = module()
    class Tok:
        def decode(self,ids,**kw): return 'roundtrip' if ids == [105] else 'lossy'
        def encode(self,text,**kw): return [105] if text == 'roundtrip' else [999]
    db = sqlite3.connect(':memory:');db.row_factory=sqlite3.Row
    db.execute('CREATE TABLE docs(id INTEGER,source TEXT,text TEXT,content_hash TEXT,split TEXT)')
    readers={k:[[101+i,11]] for i,k in enumerate(m.KEYS[:3])}
    replay=[[106,11],[105,11]]
    receipts,counts=m.write_training(tmp_path,db,readers,{k:0 for k in readers},replay,Tok(),
        {k:2 for k in m.KEYS},set(),set(),lambda t:False,{},100,100,lambda **kw:None)
    assert counts['non_roundtrip_documents_excluded']==1
    reader=m.helpers().producer().native_sdk().IndexedDataset(receipts['climbmix_replay']['prefix'])
    ids=reader[0]
    assert ids.tolist()==[105,11]
    assert replay==[[106,11],[105,11]]
