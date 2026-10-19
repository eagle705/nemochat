"""Offline2B corpus controls; fixtures are not production data."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def module():
    path = ROOT / 'midtraining/data/offline_kd/prepare.py'
    assert path.is_file(), 'new independent producer is required'
    spec = importlib.util.spec_from_file_location('offline2b_prepare', path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_exact_budget_includes_native_reserve():
    m = module()
    plan = m.budget()
    assert plan['train_iters'] == 3815
    assert plan['processed_tokens'] == 2000158720
    assert plan['train_tokens'] == dict(code_concepts=1346818049, algorithmic=201027585,
                                      formal_logic=60317697, climbmix_replay=402042881)
    assert plan['cycling'] is False


def test_census_measures_tail_without_repeating_prefix():
    import sqlite3
    m = module()
    db = sqlite3.connect(':memory:')
    db.row_factory = sqlite3.Row
    db.execute('CREATE TABLE docs(id INTEGER,source TEXT,text TEXT,content_hash TEXT,split TEXT)')
    import hashlib
    for i,text in enumerate(['a','bb','ccc','heldout','ddd','ccc']):
        db.execute('INSERT INTO docs VALUES(?,?,?,?,?)', (i,'code_concepts',text,hashlib.sha256(text.encode()).hexdigest(),'train'))
    class Tokenizer:
        def encode(self, text, **kw):
            return [100+ord(c) for c in text]
    tok = Tokenizer()
    prefix = [tok.encode('a')+[11], tok.encode('bb')+[11]]
    reserved = {hashlib.sha256('heldout'.encode()).hexdigest()}
    seen = set()
    out = m.census_source(db,'code_concepts',prefix,tok,set(),reserved,seen,100)
    assert out['tokens'] == 13
    assert out['documents'] == 4
    assert out['native_reused_documents'] == 2
    assert out['tail_tokenized_documents'] == 4
    assert out['duplicate_token_documents_skipped'] == 1
    assert out['heldout_text_documents_skipped'] == 1
    assert len(seen) == 4


def test_replay_roundtrip_mismatch_is_excluded_without_mutating_tokens():
    m = module()
    class Tokenizer:
        def decode(self, ids, **kw):
            return 'decoded'
        def encode(self, text, **kw):
            return [101,102]
    ids = [100,102,11]
    result = m.replay_candidate(ids,Tokenizer())
    assert result['accepted'] is False
    assert result['reason'] == 'non_roundtrip_token_document_excluded'
    assert result['first_mismatch_position'] == 0
    assert ids == [100,102,11]


def test_capacity_report_refuses_manifest_for_any_source_shortfall():
    m = module()
    plan = m.budget()
    sources = {k: {'tokens': v} for k,v in plan['train_tokens'].items()}
    sources['code_concepts']['tokens'] -= 123
    report = m.capacity_report(sources)
    assert report['status'] == 'blocked_insufficient_unique_source_capacity'
    assert report['shortfalls'] == {'code_concepts': 123}
    assert report['training_release'] is False
    assert report['manifest_published'] is False


def test_cli_preview_is_inert():
    import subprocess, json, sys
    proc = subprocess.run([sys.executable,str(ROOT/'midtraining/data/offline_kd/prepare.py'),'--preview'],
                          capture_output=True,text=True,check=True)
    assert proc.stdout, 'preview must emit the actual independent budget'
    out = json.loads(proc.stdout)
    assert out['budget']['train_iters'] == 3815
    assert out['execute'] is False


def test_census_requires_review_before_creating_output(tmp_path):
    import pytest
    m = module()
    output = tmp_path/'fresh'
    with pytest.raises(ValueError,match='source review'):
        m.execute_census(output,{'approved':False})
    assert not output.exists()


def test_census_rejects_corrupt_native_cursor_before_counting():
    import sqlite3, hashlib, pytest
    m = module()
    db = sqlite3.connect(':memory:')
    db.row_factory = sqlite3.Row
    db.execute('CREATE TABLE docs(id INTEGER,source TEXT,text TEXT,content_hash TEXT,split TEXT)')
    db.execute('INSERT INTO docs VALUES(0,?,?,?,?)', ('code_concepts','a',hashlib.sha256(b'a').hexdigest(),'train'))
    class Tokenizer:
        def encode(self, text, **kw):
            return [100]
    with pytest.raises(ValueError,match='cursor mismatch'):
        m.census_source(db,'code_concepts',[[101,11]],Tokenizer(),set(),set(),set(),100)


def test_replay_rejects_reserved_ids_instead_of_skipping_them():
    import pytest
    m = module()
    with pytest.raises(ValueError,match='reserved IDs'):
        m.replay_candidate([100,18,11],None)


def test_capacity_cannot_release_even_when_all_token_thresholds_met():
    m = module()
    sources = {k:{'tokens':v} for k,v in m.budget()['train_tokens'].items()}
    out = m.capacity_report(sources)
    assert out['status'] == 'capacity_measured_not_materialized'
    assert out['training_release'] is False
    assert out['shortfalls'] == {}


def test_census_preserves_reserved_token_documents_in_tail():
    import sqlite3,hashlib
    m = module()
    db = sqlite3.connect(':memory:')
    db.row_factory = sqlite3.Row
    db.execute('CREATE TABLE docs(id INTEGER,source TEXT,text TEXT,content_hash TEXT,split TEXT)')
    db.execute('INSERT INTO docs VALUES(0,?,?,?,?)',('formal_logic','a',hashlib.sha256(b'a').hexdigest(),'train'))
    class Tokenizer:
        def encode(self,text,**kw):
            return [100]
    p = m.helper('midtraining/data/recover_corpus.py').producer()
    out = m.census_source(db,'formal_logic',[],Tokenizer(),{p.token_digest([100,11])},set(),set(),100)
    assert out['tokens'] == 0
    assert out['heldout_token_documents_skipped'] == 1


def test_diagnostic_counts_forbidden_ids_but_never_accepts_document():
    import sqlite3,hashlib
    m = module()
    db = sqlite3.connect(':memory:')
    db.row_factory = sqlite3.Row
    db.execute('CREATE TABLE docs(id INTEGER,source TEXT,text TEXT,content_hash TEXT,split TEXT)')
    for i,text in enumerate(['bad','good']):
        db.execute('INSERT INTO docs VALUES(?,?,?,?,?)',(i,'algorithmic',text,hashlib.sha256(text.encode()).hexdigest(),'train'))
    class Tokenizer:
        def encode(self,text,**kw):
            return [18,18,29] if text == 'bad' else [100]
    out = m.census_source(db,'algorithmic',[],Tokenizer(),set(),set(),set(),100,diagnose_forbidden=True)
    assert out['tokens'] == 2
    assert out['documents'] == 1
    assert out['forbidden_id_documents_rejected'] == 1
    assert out['rejected_reserved_id_occurrences'] == {'18':2,'29':1}
    assert out['reserved_ids_18_through_29'] == 0
    assert out['source_compatibility_gate'] == 'blocked_forbidden_source_ids_require_explicit_review'


def test_source_alias_gate_remains_blocked_even_with_sufficient_safe_capacity():
    m = module()
    sources = {k:{'tokens':v} for k,v in m.budget()['train_tokens'].items()}
    sources['algorithmic']['source_compatibility_gate'] = 'blocked_forbidden_source_ids_require_explicit_review'
    out = m.capacity_report(sources)
    assert out['status'] == 'blocked_source_tokenizer_compatibility'
    assert out['source_compatibility_blockers'] == ['algorithmic']
    assert out['training_release'] is False


def test_original_replay_readback_preserves_known_upstream_clipping(tmp_path):
    m = module()
    p = m.helper('midtraining/data/recover_corpus.py').producer()
    writer = p.NativeWriter(tmp_path/'corpus',11,131072)
    writer.append([100,101,11])
    entry = writer.publish()
    original = {'train':entry,'last_document_truncated':True}
    reader, measured = m.read_original_replay(original)
    assert len(reader) == 1
    assert measured['truncated_documents'] == 1
    assert measured['upstream_last_document_truncated'] is True
    assert measured['new_generation_truncated_documents'] == 0
    assert len(measured['bin_sha256']) == 64
