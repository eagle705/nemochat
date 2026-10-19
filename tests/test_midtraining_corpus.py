"""CPU fixtures, not corpus-scale or native training evidence."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'midtraining/data/prepare_corpus.py'


def module():
    spec = importlib.util.spec_from_file_location('midtraining_corpus', SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_preview_is_inert_and_pins_allowlist(tmp_path):
    result = subprocess.run([sys.executable, str(SCRIPT), '--preview'], cwd=ROOT,
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan['acquisition_bytes'] == 759378296
    assert plan['training_token_quotas'] == {'code_concepts': 6000000, 'algorithmic': 1000000,
                                            'formal_logic': 1000000, 'climbmix_replay': 2000000}
    assert plan['execute'] is False
    assert plan['cycling'] is False
    assert len(plan['sources']) == 3
    assert plan['production_training_budget'] is None
    help_result = subprocess.run([sys.executable, str(SCRIPT), '--help'], cwd=tmp_path,
                                 text=True, capture_output=True)
    assert help_result.returncode == 0
    assert list(tmp_path.iterdir()) == []


def test_config_rejects_unapproved_sources_and_budgets():
    import copy
    import pytest
    mod = module()
    config = mod.load_config()
    mod.validate_config(config)
    for field, value in [('path', '../Economics.parquet'), ('size', 10**9),
                         ('sha256', '0'*64), ('train_tokens', 1)]:
        changed = copy.deepcopy(config)
        changed['sources'][0][field] = value
        with pytest.raises(ValueError):
            mod.validate_config(changed)
    changed = copy.deepcopy(config)
    changed['sources'].append(changed['sources'][0])
    with pytest.raises(ValueError):
        mod.validate_config(changed)
    changed = copy.deepcopy(config)
    changed['policy']['cycling'] = True
    with pytest.raises(ValueError):
        mod.validate_config(changed)


def test_component_split_reserves_variants_and_benchmark_containment(tmp_path):
    mod = module()
    matcher = mod.BenchmarkMatcher(['What is 2 + 2?', '비밀 문제?'])
    assert matcher.matches('prefix WHAT is 2 + 2? answer suffix')
    assert matcher.matches('인용: 비 밀 문 제? 답')
    assert not matcher.matches('unrelated')
    catalog = mod.Catalog(tmp_path / 'catalog.sqlite', matcher)
    rows = [
        ('code_concepts', {'text': 'solution a', 'metadata': {'problem_prompt': 'shared problem'}}),
        ('code_concepts', {'text': 'solution b', 'metadata': {'problem_prompt': 'shared problem'}}),
        ('code_concepts', {'text': 'solution b', 'metadata': {'problem_prompt': 'shared problem'}}),
        ('code_concepts', {'text': 'prefix What is 2 + 2? suffix', 'metadata': {'problem_prompt': 'blocked problem'}}),
        ('code_concepts', {'text': 'clean variant', 'metadata': {'problem_prompt': 'blocked problem'}}),
        ('algorithmic', {'text': 'algorithm one'}),
    ]
    for source, row in rows:
        catalog.add(source, row)
    catalog.finalize('seed')
    records = list(catalog.records())
    assert len(records) == 5
    assert records[0]['group_id'] == records[1]['group_id']
    assert records[0]['split'] == records[1]['split']
    assert records[2]['split'] == records[3]['split'] == 'excluded'
    assert catalog.counts['duplicate_rows'] == 1
    assert catalog.counts['benchmark_rows'] == 1
    assert catalog.counts['benchmark_component_documents'] == 2
    assert len(list(catalog.reservations())) == sum(r['split'] == 'validation' for r in records)
    catalog.close()


def test_formal_composite_questions_connect_variants(tmp_path):
    mod = module()
    catalog = mod.Catalog(tmp_path/'formal.sqlite', mod.BenchmarkMatcher([]))
    prefix = 'The following are multiple choice questions (with answers) about formal logic.\n'
    q1 = 'Which implication holds?\nA. X\nB. Y\nAnswer: A'
    q2 = 'Translate this formula?\nA. A\nB. B\nAnswer: B'
    catalog.add('formal_logic', {'text': prefix+q1+'\n\n'+q2})
    catalog.add('formal_logic', {'text': prefix+q2.replace('Answer: B', 'Answer: A')})
    catalog.add('formal_logic', {'text': 'unknown formal format'})
    catalog.finalize('seed')
    records = list(catalog.records())
    assert len(records) == 2
    assert records[0]['group_id'] == records[1]['group_id']
    assert catalog.counts['schema_rejections'] == 1
    catalog.close()


def test_native_preparation_whole_documents_and_exhaustion(tmp_path):
    import pytest
    mod = module()
    writer = mod.NativeWriter(tmp_path/'train/corpus', eos_id=11, vocab_size=131072)
    receipt = mod.write_documents([{'tokens': [5, 6, 11], 'id': 'a'},
                                   {'tokens': [7, 8, 9, 11], 'id': 'b'},
                                   {'tokens': [10, 11], 'id': 'c'}], writer, 5)
    assert receipt['tokens'] == 7
    assert receipt['overshoot_tokens'] == 2
    assert receipt['documents'] == 2
    assert receipt['reader_verified']
    reader = mod.native_sdk().IndexedDataset(str(tmp_path/'train/corpus'))
    assert reader[0].tolist() == [5, 6, 11]
    assert reader[1].tolist() == [7, 8, 9, 11]
    with pytest.raises(FileExistsError):
        mod.NativeWriter(tmp_path/'train/corpus', 11, 131072)
    short = mod.NativeWriter(tmp_path/'short/corpus', 11, 131072)
    with pytest.raises(RuntimeError, match='exhausted'):
        mod.write_documents([{'tokens': [1, 11], 'id': 'x'}], short, 10)
    assert not (tmp_path/'short/corpus.idx').exists()
    bad = mod.NativeWriter(tmp_path/'bad/corpus', 11, 131072)
    with pytest.raises(ValueError, match='EOS'):
        mod.write_documents([{'tokens': [1, 2], 'id': 'x'}], bad, 1)


def test_download_is_digest_checked_bounded_and_resumable(tmp_path):
    import hashlib
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import pytest
    mod = module()
    payload = b'PAR1-real-local-http-fixture-PAR1'
    requests = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_GET(self):
            offset = int(self.headers.get('Range', 'bytes=0-').split('=')[1].split('-')[0])
            requests.append(offset)
            self.send_response(206 if offset else 200)
            self.send_header('Content-Length', str(len(payload)-offset))
            if offset:
                self.send_header('Content-Range', f'bytes {offset}-{len(payload)-1}/{len(payload)}')
            self.end_headers()
            self.wfile.write(payload[offset:])
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    source = {'size': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()}
    url = f'http://127.0.0.1:{server.server_port}/file'
    try:
        dest = tmp_path/'raw.parquet'
        Path(str(dest)+'.part').write_bytes(payload[:5])
        budget = {'used': 0, 'cap': len(payload)}
        receipt = mod.download_one(source, dest, url, budget)
        assert dest.read_bytes() == payload
        assert requests == [5]
        assert receipt['sha256'] == source['sha256']
        assert budget['used'] == len(payload)-5
        mod.download_one(source, dest, url, budget)
        assert requests == [5]
        with pytest.raises(ValueError, match='digest'):
            mod.download_one(dict(source, sha256='0'*64), dest, url, budget)
        with pytest.raises(RuntimeError, match='budget'):
            mod.download_one(source, tmp_path/'other', url, {'used': 0, 'cap': 2})
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_full_scan_reservation_survives_selection_quota(tmp_path):
    mod = module()
    import pyarrow as pa
    import pyarrow.parquet as pq
    config = mod.load_config()
    config['sources'] = [dict(config['sources'][0], train_tokens=5)]
    config['policy']['validation_tokens_per_source'] = 3
    rows = []
    for i in range(500):
        rows.append({'text': f'synthetic solution {i}', 'license': 'cc-by-4.0',
                     'metadata': {'problem_prompt': f'problem {i}'}})
    parquet = tmp_path/'fixture.parquet'
    pq.write_table(pa.Table.from_pylist(rows), parquet)
    catalog = mod.Catalog(tmp_path/'cat.sqlite', mod.BenchmarkMatcher([]))
    mod.scan_parquet(catalog, 'code_concepts', parquet, lambda: None)
    catalog.finalize(config['policy']['split_seed'])
    expected = list(catalog.reservations())
    assert len(expected) > 1
    class TokenizerFixture:
        def encode(self, text, **kwargs):
            assert kwargs['truncation'] is False
            return [20, 30, 40]
    receipts = mod.prepare_specialized(catalog, TokenizerFixture(), tmp_path/'out', config, lambda: None)
    assert receipts['code_concepts']['train']['tokens'] == 8
    assert receipts['code_concepts']['validation']['tokens'] == 4
    persisted = [json.loads(line) for line in (tmp_path/'out/heldout-reservations.jsonl').read_text().splitlines()]
    assert persisted == expected
    assert len(persisted) > receipts['code_concepts']['validation']['documents']
    assert catalog.counts['scanned_rows'] == 500
    assert receipts['code_concepts']['quota_skipped_documents'] > 0
    catalog.close()


def test_replay_preserves_existing_split_and_never_cycles(tmp_path):
    import pytest
    mod = module()
    config = mod.load_config()
    config['replay']['train_tokens'] = 5
    config['replay']['omit_upstream_clipped_tails'] = False
    config['policy']['validation_tokens_per_source'] = 2
    catalog = mod.Catalog(tmp_path/'cat.sqlite', mod.BenchmarkMatcher(['banned question']))
    catalog.finalize('seed')
    catalog.db.execute('CREATE TABLE reserved_tokens(hash TEXT PRIMARY KEY)')
    catalog.db.execute('CREATE TABLE selected_tokens(hash TEXT PRIMARY KEY)')
    class TokenizerFixture:
        texts = {20: 'replay train a', 30: 'replay train b', 40: 'heldout', 50: 'banned question'}
        def decode(self, ids, **kwargs):
            return self.texts[int(ids[0])]
        def encode(self, text, **kwargs):
            key = next(k for k, v in self.texts.items() if v == text)
            return [key, key+1]
    train_docs = [[50, 51, 11], [40, 41, 11], [20, 21, 11], [30, 31, 11]]
    val_docs = [[40, 41, 11]]
    for split, docs in [('train', train_docs), ('validation', val_docs)]:
        writer = mod.NativeWriter(tmp_path/split/'corpus', 11, 131072)
        mod.write_documents([{'tokens': d} for d in docs], writer, sum(map(len, docs)))
        config['replay'][split+'_prefix'] = str(tmp_path/split/'corpus')
    catalog.db.execute('INSERT INTO reserved_tokens VALUES(?)', (mod.token_digest(val_docs[0]),))
    receipt = mod.prepare_replay(catalog, TokenizerFixture(), tmp_path/'out', config, lambda: None)
    assert receipt['train']['tokens'] == 6
    assert receipt['validation']['tokens'] == 3
    assert receipt['already_consumed_by_base'] is True
    assert receipt['benchmark_excluded_documents'] == 1
    assert receipt['reserved_excluded_documents'] == 1
    config['policy']['max_document_tokens'] = 2
    with pytest.raises(RuntimeError, match='exhausted'):
        mod.prepare_replay(catalog, TokenizerFixture(), tmp_path/'oversized', config, lambda: None)
    config['policy']['max_document_tokens'] = 1000000
    config['replay']['train_tokens'] = 100
    with pytest.raises(RuntimeError, match='exhausted'):
        mod.prepare_replay(catalog, TokenizerFixture(), tmp_path/'too-much', config, lambda: None)
    catalog.close()


def test_parent_review_and_complete_original_reservation_are_required(tmp_path):
    import pytest
    mod = module()
    config = mod.load_config()
    with pytest.raises(ValueError, match='review'):
        mod.verify_review(config, {'approved': False})
    review = {'approved': True, 'all_intended_benchmarks_registered': True,
              'files': {str(p.relative_to(ROOT)): mod.file_digest(p)
                        for p in [SCRIPT, ROOT/'midtraining/configs/corpus.yaml', mod.SDK_PATH]},
              'benchmark_manifests': [{'path': config['policy']['kmmlu_path'],
                  'sha256': mod.file_digest(ROOT/config['policy']['kmmlu_path']), 'count': 34871}]}
    proof = mod.verify_review(config, review)
    assert len(proof['benchmark_questions']) == 34871
    assert len(proof['original_reserved_hashes']) == 84990
    changed = dict(review, all_intended_benchmarks_registered=False)
    with pytest.raises(ValueError, match='benchmark'):
        mod.verify_review(config, changed)
    review['files']['midtraining/data/prepare_corpus.py'] = '0'*64
    with pytest.raises(ValueError, match='source'):
        mod.verify_review(config, review)
    command = subprocess.run([sys.executable, str(SCRIPT), '--execute', '--output', str(tmp_path/'must-not-exist')],
                             cwd=ROOT, capture_output=True, text=True)
    assert command.returncode != 0
    assert not (tmp_path/'must-not-exist').exists()


def test_final_audit_fails_closed_on_entire_heldout_reservation(tmp_path):
    import pytest
    mod = module()
    catalog = mod.Catalog(tmp_path/'cat.sqlite', mod.BenchmarkMatcher([]))
    catalog.db.execute('CREATE TABLE reserved_tokens(hash TEXT PRIMARY KEY)')
    ids = [20, 21, 11]
    catalog.db.execute('INSERT INTO reserved_tokens VALUES(?)', (mod.token_digest(ids),))
    writer = mod.NativeWriter(tmp_path/'train/corpus', 11, 131072)
    train = mod.write_documents([{'tokens': ids}], writer, 2)
    val_writer = mod.NativeWriter(tmp_path/'validation/corpus', 11, 131072)
    validation = mod.write_documents([{'tokens': [22, 23, 11]}], val_writer, 2)
    receipts = {'fixture': {'train': train, 'validation': validation}}
    with pytest.raises(ValueError, match='heldout'):
        mod.audit_outputs(catalog, receipts, 4096, lambda: None)
    catalog.db.execute('DELETE FROM reserved_tokens')
    audit = mod.audit_outputs(catalog, receipts, 4096, lambda: None)
    assert audit['train_tokens'] == 3
    assert audit['train_documents'] == 1
    assert audit['train_heldout_token_document_overlap'] == 0
    catalog.close()


def test_real_parquet_to_native_fixture_pipeline(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    mod = module()
    config = mod.load_config()
    config['sources'] = [dict(config['sources'][0], train_tokens=100)]
    config['replay']['train_tokens'] = 50
    config['policy']['validation_tokens_per_source'] = 20
    rows = [{'text': f'handwritten fixture solution {i}', 'license': 'cc-by-4.0',
             'metadata': {'problem_prompt': f'fixture problem {i}'}} for i in range(500)]
    parquet = tmp_path/'real-fixture.parquet'
    pq.write_table(pa.Table.from_pylist(rows), parquet)
    class CharacterTokenizerFixture:
        def encode(self, text, **kwargs):
            return [ord(c)+100 for c in text]
        def decode(self, ids, **kwargs):
            return ''.join(chr(i-100) for i in ids)
    tokenizer = CharacterTokenizerFixture()
    for split in ('train', 'validation'):
        docs = [tokenizer.encode(f'replay {split} fixture document {i}')+[11] for i in range(5)]
        writer = mod.NativeWriter(tmp_path/split/'corpus', 11, 131072)
        mod.write_documents([{'tokens': d} for d in docs], writer, sum(map(len, docs)))
        config['replay'][split+'_prefix'] = str(tmp_path/split/'corpus')
    proof = {'benchmark_questions': set(), 'original_reserved_hashes': set()}
    status = {}
    snapshots = []
    def progress():
        snapshots.append(status.get('active_train_tokens', 0))
    result = mod.run_preparation(config, tmp_path/'out', {'code_concepts': parquet},
                                 proof, tokenizer, progress, status)
    assert max(snapshots) > 0
    assert result['counts']['scanned_rows'] == 500
    assert result['audit']['train_tokens'] >= 150
    assert result['audit']['train_heldout_token_document_overlap'] == 0
    assert result['cycling'] is False
    assert result['production_training_budget'] is None
    assert (tmp_path/'out/bridge-dataset-fragment.yaml').exists()
    import yaml
    fragment = yaml.safe_load((tmp_path/'out/bridge-dataset-fragment.yaml').read_text())
    assert fragment['dataset']['eod_mask_loss'] is False
    assert fragment['dataset']['blend_per_split'][2] is None
    assert result['full_bounded_shards_scanned'] is True
    manifest = json.loads((tmp_path/'out/manifest.json').read_text())
    assert manifest['schema_version'] == 1
    assert manifest['status'] == 'prepared_verified'
    assert manifest['sources']['code_concepts']['train']['prefix'] == result['sources']['code_concepts']['train']['prefix']
    assert manifest['sequence_length'] == 4096
    assert manifest['production_training_budget'] is None
    assert result['manifest_sha256'] == mod.file_digest(tmp_path/'out/manifest.json')


def test_shared_raw_directory_preview_is_inert(tmp_path):
    result = subprocess.run([sys.executable, str(SCRIPT), '--preview', '--raw-directory', str(tmp_path/'raw'), '--local-only'],
                            cwd=ROOT, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['raw_directory'] == str(tmp_path/'raw')
    assert list(tmp_path.iterdir()) == []


def test_execute_runtime_gate_precedes_all_writes(tmp_path, monkeypatch):
    import importlib.metadata
    import pytest
    mod = module()
    monkeypatch.setattr(importlib.metadata, 'version', lambda name: '4.57.6')
    with pytest.raises(RuntimeError, match='5.12.1'):
        mod.execute(mod.load_config(), tmp_path/'out', tmp_path/'cache', {})
    assert list(tmp_path.iterdir()) == []
    denied = subprocess.run([sys.executable, str(SCRIPT), '--execute'], cwd=ROOT,
                            capture_output=True, text=True)
    assert denied.returncode == 2
    assert '--reviewed-sources' in denied.stderr


def test_original_heldout_exclusion_propagates_to_problem_variants(tmp_path):
    import hashlib
    mod = module()
    catalog = mod.Catalog(tmp_path/'cat.sqlite', mod.BenchmarkMatcher([]))
    catalog.original_reserved_hashes = {hashlib.sha256(b'reserved original').hexdigest()}
    for text in ('reserved original', 'new answer variant'):
        catalog.add('code_concepts', {'text': text, 'metadata': {'problem_prompt': 'same source problem'}})
    catalog.finalize('seed')
    assert {r['split'] for r in catalog.records()} == {'excluded'}
    assert catalog.counts['original_reserved_rows'] == 1
    assert catalog.counts['original_reserved_component_documents'] == 2
    assert catalog.counts['benchmark_component_documents'] == 0
    catalog.close()


def test_bounded_resource_and_tokenizer_policies_cannot_be_relaxed():
    import copy
    import pytest
    mod = module()
    config = mod.load_config()
    changes = [('max_replay_scan_tokens', 10**12), ('max_document_tokens', 10**12),
               ('sequence_length', 8192), ('max_hours', 12)]
    for name, value in changes:
        altered = copy.deepcopy(config)
        altered['policy'][name] = value
        with pytest.raises(ValueError):
            mod.validate_config(altered)
    altered = copy.deepcopy(config)
    altered['tokenizer']['revision'] = 'main'
    with pytest.raises(ValueError):
        mod.validate_config(altered)
