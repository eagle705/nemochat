"""Small handwritten CPU fixtures using the preserved real native indexed SDK."""
import importlib.util
import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT/'midtraining/data/recover_corpus.py'


def module():
    assert SCRIPT.is_file(), 'recovery producer missing'
    spec = importlib.util.spec_from_file_location('recovery_fixture', SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Tokenizer:
    def __init__(self):
        self.calls = []
        self.texts = {}

    def encode(self, text, **kw):
        assert kw == dict(add_special_tokens=False, truncation=False)
        self.calls.append(text)
        ids = [ord(c)+100 for c in text]
        self.texts[tuple(ids)] = text
        return ids

    def decode(self, ids, **kw):
        return ''.join(chr(int(i)-100) for i in ids)


def fixture(tmp_path):
    m = module()
    p = m.producer()
    tok = Tokenizer()
    path = tmp_path/'catalog.sqlite'
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE docs(id INTEGER PRIMARY KEY,source TEXT,text TEXT,content_hash TEXT UNIQUE,group_id TEXT,split TEXT)')
        db.execute('CREATE TABLE reserved_tokens(hash TEXT PRIMARY KEY)')
        db.execute('CREATE TABLE selected_tokens(hash TEXT PRIMARY KEY)')
        for i in range(30):
            text = f'code document {i}'
            db.execute('INSERT INTO docs VALUES(?,?,?,?,?,?)', (i,'code_concepts',text,p.hashlib.sha256(text.encode()).hexdigest(),'a'*64,'validation' if i == 1 else 'train'))
            if i != 1:
                db.execute('INSERT INTO selected_tokens VALUES(?)', (p.token_digest(tok.encode(text,add_special_tokens=False,truncation=False)+[11]),))
    writer = p.NativeWriter(tmp_path/'old/train/corpus',11,131072)
    for i in (0,2,3,4,5,6,7,8,9,10):
        writer.append(tok.encode(f'code document {i}',add_special_tokens=False,truncation=False)+[11])
    entry = writer.publish()
    entry['truncated_documents'] = 0
    writer.close()
    tok.calls.clear()
    return m,p,tok,path,entry


def test_cursor_reuses_native_prefix_and_checks_only_small_sample(tmp_path):
    m,p,tok,path,entry = fixture(tmp_path)
    digest = p.file_digest(path)
    with m.open_catalog(path,digest) as db:
        proof = m.infer_cursor(db,entry,{},tok,expected_cursor=10)
    assert proof['last_catalog_id'] == 10
    assert proof['documents'] == 10
    assert len(tok.calls) <= 8
    assert tok.calls[0] == 'code document 0'
    assert tok.calls[-1] == 'code document 10'
    assert p.file_digest(path) == digest
    assert proof['native']['bin_sha256'] == p.file_digest(entry['prefix']+'.bin')


@pytest.mark.parametrize('counter',['oversized_documents','reserved_token_documents_skipped','duplicate_token_documents_skipped'])
def test_cursor_rejects_any_historical_skip(tmp_path,counter):
    m,p,tok,path,entry = fixture(tmp_path)
    with m.open_catalog(path,p.file_digest(path)) as db:
        with pytest.raises(ValueError,match='zero-skip'):
            m.infer_cursor(db,entry,{counter:1},tok)


@pytest.mark.parametrize('fault',['wrong_cursor','wrong_tokens','wrong_hash','wrong_native','bad_shape'])
def test_cursor_fails_closed_on_lineage_or_shape_change(tmp_path,fault):
    m,p,tok,path,entry = fixture(tmp_path)
    if fault == 'wrong_tokens':
        entry['tokens'] += 1
    if fault in ('wrong_hash','wrong_native'):
        with sqlite3.connect(path) as db:
            db.execute('UPDATE docs SET '+('content_hash' if fault == 'wrong_hash' else 'text')+'=? WHERE id=0',('b'*64 if fault == 'wrong_hash' else 'different',))
    if fault == 'bad_shape':
        entry['documents'] += 1
    with m.open_catalog(path,p.file_digest(path)) as db:
        with pytest.raises(ValueError):
            m.infer_cursor(db,entry,{},tok,expected_cursor=9 if fault == 'wrong_cursor' else 10)


@pytest.mark.parametrize('fault',['missing','malformed','missing_table','digest'])
def test_catalog_is_readonly_and_never_repaired(tmp_path,fault):
    m,p,tok,path,entry = fixture(tmp_path)
    digest = p.file_digest(path)
    if fault == 'missing':
        path.unlink()
    elif fault == 'malformed':
        path.write_bytes(b'not a database')
        digest = p.file_digest(path)
    elif fault == 'missing_table':
        with sqlite3.connect(path) as db:
            db.execute('DROP TABLE reserved_tokens')
        digest = p.file_digest(path)
    else:
        digest = '0'*64
    before = path.read_bytes() if path.exists() else None
    with pytest.raises((ValueError,FileNotFoundError,sqlite3.DatabaseError)):
        with m.open_catalog(path,digest):
            pass
    assert (path.read_bytes() if path.exists() else None) == before


def recovery_fixture(tmp_path):
    m = module()
    p,tok = m.producer(),Tokenizer()
    config = dict(max_document_tokens=1000000,max_replay_scan_tokens=2000000)
    recipe = p.load_config()
    path = tmp_path/'full.sqlite'
    origin = dict(phase='failed',counts={},prepared_sources={})
    def text(key,i):
        return f'{key} synthetic {i} '+('x'*5000)
    def publish(prefix, texts):
        writer = p.NativeWriter(prefix,11,131072)
        for value in texts:
            writer.append(tok.encode(value,add_special_tokens=False,truncation=False)+[11])
        result = writer.publish()
        writer.close()
        result['truncated_documents'] = 0
        return result
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE docs(id INTEGER PRIMARY KEY,source TEXT,text TEXT,content_hash TEXT UNIQUE,group_id TEXT,split TEXT)')
        db.execute('CREATE TABLE reserved_tokens(hash TEXT PRIMARY KEY)')
        db.execute('CREATE TABLE selected_tokens(hash TEXT PRIMARY KEY)')
        for ki,key in enumerate(m.KEYS[:3]):
            for i in range(140):
                value = text(key,i)
                db.execute('INSERT INTO docs VALUES(?,?,?,?,?,?)',(ki*1000+i,key,value,p.hashlib.sha256(value.encode()).hexdigest(),'a'*64,'train'))
                if (key == 'code_concepts' and i <= 10) or key != 'code_concepts':
                    db.execute('INSERT INTO selected_tokens VALUES(?)',(p.token_digest(tok.encode(value,add_special_tokens=False,truncation=False)+[11]),))
        db.execute('INSERT INTO reserved_tokens VALUES(?)',(p.token_digest(tok.encode(text('code_concepts',11),add_special_tokens=False,truncation=False)+[11]),))
    seed = dict(sources={})
    for key in m.KEYS:
        val = publish(tmp_path/'old'/key/'validation/corpus',[f'frozen {key} '+('y'*5000)])
        for suffix in ('bin','idx'):
            val[suffix+'_sha256'] = p.file_digest(val['prefix']+'.'+suffix)
        seed['sources'][key] = dict(validation=val)
        origin['prepared_sources'][key] = dict(validation=val)
    for key,n in [('code_concepts',10),('algorithmic',15)]:
        origin['prepared_sources'][key]['train'] = publish(tmp_path/'old'/key/'train/corpus',[text(key,i) for i in range(n)])
    for split in ('train','validation'):
        entry = publish(tmp_path/'replay'/split/'corpus',[f'replay {split} {i} '+('z'*5000) for i in range(100 if split == 'train' else 2)])
        recipe['replay'][split+'_prefix'] = entry['prefix']
    reservations = tmp_path/'heldout-reservations.jsonl'
    reservations.write_text('')
    proof = dict(catalog=path,catalog_sha256=p.file_digest(path),origin=origin,
        full_reservations=reservations,seed=seed,benchmark_questions={'FORBIDDEN'},original_reserved_hashes=set(),
        inherited_full_scan_provenance=dict(fixture=True,origin_phase='failed'),expected_code_cursor=9)
    wide = dict(weights=m.WEIGHTS,sources={key:dict(train=dict(tokens=10**9),validation=dict(tokens=5001)) for key in m.KEYS})
    gate = m.native().capacity_gate(wide,1,128,4096)
    plan = dict(train_iters=1,train_tokens={k:v['required_tokens'] for k,v in gate['sources'].items()})
    tok.calls.clear()
    return m,p,tok,config,recipe,proof,plan


def test_real_native_recovery_end_to_end_does_not_reencode_old_code(tmp_path):
    m,p,tok,config,recipe,proof,plan = recovery_fixture(tmp_path)
    out = tmp_path/'recovered'
    result = m.run_preparation(config,recipe,out,proof,tok,plan,lambda:None,{})
    p.atomic_json(out/'success.json',dict(phase='prepared_verified',result=result))
    manifest,digest = m.native().load_prepared_manifest(out/'manifest.json')
    assert digest == result['manifest_sha256']
    assert manifest['weights'] == m.WEIGHTS
    assert manifest['recovery_did_rescan_raw'] is False
    assert manifest['inherited_full_scan_provenance']['origin_phase'] == 'failed'
    assert manifest['audit']['training_duplicate_token_documents'] == 0
    assert manifest['audit']['train_heldout_token_document_overlap'] == 0
    assert set(manifest['reserved_id_audit']['counts'].values()) == {0}
    assert all(manifest['sources'][k]['train']['tokens'] >= plan['train_tokens'][k] for k in m.KEYS)
    for key in m.KEYS:
        for suffix in ('bin','idx'):
            assert manifest['sources'][key]['validation'][suffix+'_sha256'] == proof['seed']['sources'][key]['validation'][suffix+'_sha256']
    old_calls = [t for t in tok.calls if t.startswith('code_concepts synthetic ') and int(t.split()[2]) < 10]
    assert len(old_calls) <= 7
    assert result['counts']['historical_selected_documents_skipped'] >= 1
    assert result['counts']['reserved_token_documents_skipped'] >= 1
    assert p.file_digest(proof['catalog']) == proof['catalog_sha256']
    assert (out/'code-cursor-proof.json').is_file()
    cursor_proof = json.loads((out/'code-cursor-proof.json').read_text())
    assert cursor_proof['recovery_source_sha256'] == p.file_digest(SCRIPT)
    assert (out/'selection.sqlite').is_file()


@pytest.mark.parametrize('fault',['exhausted','reserved_id','text_hash','cursor','duplicate_reused'])
def test_recovery_preserves_failure_and_never_publishes_success(tmp_path,fault):
    m,p,tok,config,recipe,proof,plan = recovery_fixture(tmp_path)
    if fault == 'exhausted':
        plan['train_tokens']['code_concepts'] = 10**9
    elif fault == 'reserved_id':
        encode = tok.encode
        tok.encode = lambda text,**kw: [18] if text.startswith('formal_logic') else encode(text,**kw)
    elif fault == 'text_hash':
        with sqlite3.connect(proof['catalog']) as db:
            db.execute("UPDATE docs SET text='changed' WHERE source='formal_logic' AND id=2000")
        proof['catalog_sha256'] = p.file_digest(proof['catalog'])
    elif fault == 'cursor':
        proof['expected_code_cursor'] = 8
    else:
        proof['origin']['prepared_sources']['algorithmic']['train'] = proof['origin']['prepared_sources']['code_concepts']['train'].copy()
    out = tmp_path/'failed'
    with pytest.raises((ValueError,RuntimeError)):
        m.run_preparation(config,recipe,out,proof,tok,plan,lambda:None,{})
    assert not (out/'success.json').exists()
    assert not (out/'manifest.json').exists()
    assert p.file_digest(proof['catalog']) == proof['catalog_sha256']


def test_cross_source_dedup_and_full_unpublished_heldout_exclusion(tmp_path):
    m,p,tok,config,recipe,proof,plan = recovery_fixture(tmp_path)
    encode = tok.encode
    def alias(text,**kw):
        if text.startswith('formal_logic synthetic 0 '):
            return encode('code_concepts synthetic 0 '+('x'*5000),**kw)
        if text.startswith('formal_logic synthetic 1 '):
            return encode('code_concepts synthetic 11 '+('x'*5000),**kw)
        return encode(text,**kw)
    tok.encode = alias
    # Reserve an ORIGINAL Replay document outside the frozen validation window.
    reader = p.native_sdk().IndexedDataset(recipe['replay']['train_prefix'])
    forbidden = p.token_digest(reader[0])
    with sqlite3.connect(proof['catalog']) as db:
        db.execute('INSERT INTO reserved_tokens VALUES(?)',(forbidden,))
    proof['catalog_sha256'] = p.file_digest(proof['catalog'])
    out = tmp_path/'dedup'
    result = m.run_preparation(config,recipe,out,proof,tok,plan,lambda:None,{})
    assert result['counts']['duplicate_token_documents_skipped'] >= 1
    assert result['counts']['reserved_token_documents_skipped'] >= 3
    native = p.native_sdk().IndexedDataset(result['sources']['climbmix_replay']['train']['prefix'])
    assert forbidden not in {p.token_digest(native[i]) for i in range(len(native))}
    assert result['audit']['training_duplicate_token_documents'] == 0
    assert result['audit']['train_heldout_token_document_overlap'] == 0


@pytest.mark.parametrize('fault',['frozen_validation','replay_scan_bound','replay_roundtrip','clipped_tail','native_reserved_id'])
def test_native_recovery_gates_are_not_metadata_only(tmp_path,fault):
    m,p,tok,config,recipe,proof,plan = recovery_fixture(tmp_path)
    if fault == 'frozen_validation':
        proof['seed']['sources']['code_concepts']['validation'] = dict(proof['seed']['sources']['code_concepts']['validation'],bin_sha256='0'*64)
    elif fault == 'replay_scan_bound':
        config['max_replay_scan_tokens'] = 1
    elif fault == 'replay_roundtrip':
        tok.decode = lambda *args,**kw:'not original text'
    elif fault == 'clipped_tail':
        # Existing train has 100 documents; requiring all tokens must fail because
        # the upstream last complete-looking document is conservatively omitted.
        reader = p.native_sdk().IndexedDataset(recipe['replay']['train_prefix'])
        plan['train_tokens']['climbmix_replay'] = int(reader.sequence_lengths.sum())
    else:
        entry = proof['origin']['prepared_sources']['algorithmic']['train']
        with open(entry['prefix']+'.bin','r+b') as stream:
            stream.write(np.asarray([18],dtype='<i4').tobytes())
    out = tmp_path/'blocked'
    with pytest.raises((ValueError,RuntimeError)):
        m.run_preparation(config,recipe,out,proof,tok,plan,lambda:None,{})
    assert not (out/'manifest.json').exists()


def test_approved_config_uses_exact_native_surplus_budget():
    m = module()
    config = m.load_config()
    m.validate_config(config)
    wide = dict(weights=m.WEIGHTS,sources={k:dict(train=dict(tokens=10**12),validation=dict(tokens=4097)) for k in m.KEYS})
    gate = m.native().capacity_gate(wide,3200,128,4096)
    assert config['train_tokens'] == {k:v['required_tokens'] for k,v in gate['sources'].items()}
    config['weights']['formal_logic'] = .1
    with pytest.raises(ValueError):
        m.validate_config(config)


@pytest.mark.parametrize('fault', [None, 'snapshot_hash', 'snapshot_bytes',
                                  'relocated_history', 'missing_snapshot',
                                  'producer_pin', 'historical_skip', 'parts_binding'])
def test_history_pins_frozen_source_not_mutated_live_native(tmp_path,monkeypatch,fault):
    m = module()
    p = m.producer()
    def pin(name,value):
        path = tmp_path/name
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value))
        return dict(path=str(path),sha256=p.file_digest(path))
    snapshots,old_files = {},{}
    # Handwritten portable history, not the real production receipt chain.
    # Literal pre-migration identifiers must not follow the active source layout.
    for name in ('midtraining/production_corpus.py', 'midtraining/production_corpus.yaml',
                 'midtraining/prepare_corpus.py', 'midtraining/corpus.yaml',
                 'midtraining/longrun_native.py',
                 'results/pretraining/nemochat_baby/climbmix/corpus_sdk.py'):
        entry = pin('snapshot/'+name,dict(historical=name))
        snapshots[name] = entry
        old_files[name] = entry['sha256']
    monkeypatch.setattr(m,'ORIGIN_PRODUCER_SHA256',old_files['midtraining/production_corpus.py'])
    source_review = dict(approved=True,files=old_files)
    failure = dict(phase='failed',source_review=source_review,source_code_sha256=old_files['midtraining/production_corpus.py'],counts={},
        insufficient_capacity=dict(source='formal_logic'),prepared_sources={k:dict(train=dict(prefix=k,documents=1,tokens=100,bin_bytes=400,idx_bytes=82)) for k in m.KEYS[:2]})
    failure_pin = pin('failure.json',failure)
    parts = dict(origin_failure_sha256=failure_pin['sha256'],origin_catalog_sha256='a'*64,
        status='published_parts_readback_and_current_digests_recorded',parts={k:dict(failure['prepared_sources'][k]['train'],bin_sha256='b'*64,idx_sha256='c'*64) for k in m.KEYS[:2]})
    review = dict(origin_failure=failure_pin,origin_review=pin('old_review.json',source_review),
        origin_catalog=dict(path=str(tmp_path/'catalog.sqlite'),sha256='a'*64),origin_snapshot=snapshots,
        origin_parts=pin('parts.json',parts))
    # Active allowlist remains relocated and cannot replace historical keys.
    assert set(m.SOURCE_FILES) == {
        'midtraining/data/production_corpus.py', 'midtraining/configs/production_corpus.yaml',
        'midtraining/data/prepare_corpus.py', 'midtraining/configs/corpus.yaml',
        'midtraining/training/longrun_native.py',
        'results/pretraining/nemochat_baby/climbmix/corpus_sdk.py'}
    assert set(m.REVIEW_FILES) == set(m.SOURCE_FILES) | {
        'midtraining/data/recover_corpus.py', 'midtraining/configs/recovery_mix67.yaml'}
    if fault == 'snapshot_hash':
        review['origin_snapshot']['midtraining/longrun_native.py']['sha256'] = '0'*64
    elif fault == 'snapshot_bytes':
        Path(snapshots['midtraining/longrun_native.py']['path']).write_text('changed')
    elif fault == 'relocated_history':
        snapshots['midtraining/training/longrun_native.py'] = snapshots.pop('midtraining/longrun_native.py')
    elif fault == 'missing_snapshot':
        snapshots.pop('midtraining/longrun_native.py')
    elif fault in ('producer_pin', 'historical_skip'):
        if fault == 'producer_pin':
            failure['source_code_sha256'] = '0'*64
        else:
            failure['counts']['oversized_documents'] = 1
        review['origin_failure'] = pin('failure.json',failure)
    elif fault == 'parts_binding':
        parts['origin_failure_sha256'] = '0'*64
        review['origin_parts'] = pin('parts.json',parts)
    before = {path: path.read_bytes() for path in tmp_path.rglob('*') if path.is_file()}
    if fault is None:
        origin,old,receipt = m.verify_history(review,lambda:None)
        assert origin['phase'] == 'failed'
        assert old == source_review
        assert receipt == parts
    else:
        with pytest.raises(ValueError):
            m.verify_history(review,lambda:None)
    assert {path: path.read_bytes() for path in before} == before


def test_cli_preview_and_denied_execution_are_inert(tmp_path):
    import subprocess
    import sys
    m = module()
    for flag in ('--help','--preview'):
        result = subprocess.run([sys.executable,str(SCRIPT),flag],cwd=tmp_path,text=True,capture_output=True)
        assert result.returncode == 0,result.stderr
    assert not list(tmp_path.iterdir())
    with pytest.raises(ValueError,match='parent'):
        m.execute(m.load_config(),tmp_path/'denied',{})
    assert not (tmp_path/'denied').exists()


def test_watchdog_and_storage_bound_leave_failure_receipt(tmp_path,monkeypatch):
    import time
    m = module()
    config = m.load_config()
    config['max_hours'] = .00001
    monkeypatch.setattr(m,'verify_review',lambda *args:time.sleep(1))
    out = ROOT/'datasets/midtraining'/('recovery-watchdog-fixture-'+tmp_path.name)
    import shutil
    try:
        with pytest.raises(TimeoutError,match='signal'):
            m.execute(config,out,dict(approved=True))
        assert json.loads((out/'failure.json').read_text())['phase'] == 'failed'
        assert not (out/'success.json').exists()
        with pytest.raises(FileExistsError):
            m.execute(config,out,dict(approved=True))
    finally:
        if out.exists():
            shutil.rmtree(out)


def test_review_schema_is_exact_before_origin_payload_reads():
    m = module()
    files = {name:m.producer().file_digest(ROOT/name) for name in m.REVIEW_FILES}
    with pytest.raises(ValueError,match='review schema'):
        m.verify_review(m.load_config(),dict(approved=True,files=files,schema_version=2),lambda:None)


@pytest.mark.parametrize('fault', [None, 'historical_paths', 'missing', 'extra', 'tampered'])
def test_relocated_active_dependencies_are_independently_fail_closed(tmp_path,monkeypatch,fault):
    m = module()
    p = m.producer()
    # Handwritten dependency bytes exercise pinning only, never native SDK execution.
    for name in m.REVIEW_FILES:
        path = tmp_path/name
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text('# handwritten active dependency: '+name+'\n')
    monkeypatch.setattr(p,'ROOT',tmp_path)
    files = {name:p.file_digest(tmp_path/name) for name in m.REVIEW_FILES}
    if fault == 'historical_paths':
        files = {name:'a'*64 for name in m.HISTORY_FILES}
    elif fault == 'missing':
        files.pop('midtraining/data/production_corpus.py')
    elif fault == 'extra':
        files['midtraining/production_corpus.py'] = 'a'*64
    elif fault == 'tampered':
        files['midtraining/data/production_corpus.py'] = '0'*64
    review = dict(schema_version=1,approved=True,files=files,
        **{key:{} for key in ('origin_failure','origin_catalog','origin_review',
            'origin_parts','origin_snapshot','full_reservations','raw_verification','calibration')})
    def stop_before_payloads(config):
        raise RuntimeError('active pins accepted; stop before historical/payload reads')
    monkeypatch.setattr(m,'validate_config',stop_before_payloads)
    if fault is None:
        with pytest.raises(RuntimeError,match='active pins accepted'):
            m.verify_review({},review,lambda:None)
    else:
        with pytest.raises(ValueError,match='SHA256 mismatch' if fault == 'tampered' else 'allowlist'):
            m.verify_review({},review,lambda:None)


def test_review_requires_exact_fresh_dependencies_before_any_payload_reads():
    m = module()
    with pytest.raises(ValueError,match='parent'):
        m.verify_review(m.load_config(),{},lambda:None)
    with pytest.raises(ValueError,match='allowlist'):
        m.verify_review(m.load_config(),dict(approved=True,files={}),lambda:None)
