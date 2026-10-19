"""Handwritten CPU fixtures, never real corpus capacity or GPU evidence."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'midtraining/data/production_corpus.py'


@pytest.fixture
def tmp_path():
    """Keep even synthetic corpus/native artifacts on RAID, independent of TMPDIR."""
    import tempfile
    with tempfile.TemporaryDirectory(prefix='production-corpus-fixture-',
                                     dir=ROOT/'datasets/midtraining') as path:
        yield Path(path)


def module():
    assert SCRIPT.is_file(), 'production corpus entry point is missing'
    spec = importlib.util.spec_from_file_location('production_corpus_test', SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_raw_allowlist_matches_official_metadata_and_never_acquires(tmp_path):
    mod = module()
    p = mod.producer()
    config = mod.load_config()
    # Explicit handwritten metadata/payload fixture, not a real Parquet shard.
    local = tmp_path/'raw.parquet'
    local.write_bytes(b'handwritten schema fixture only')
    entry = dict(key='code_concepts', path='Nemotron-Pretraining-Code-Concepts/part_000001.parquet',
                 local_path=str(local), size=local.stat().st_size, sha256=p.file_digest(local))
    metadata = tmp_path/'metadata.json'
    metadata.write_text(json.dumps(dict(id=p.load_config()['repository'], sha=p.load_config()['revision'],
        siblings=[dict(rfilename=entry['path'], size=entry['size'], lfs=dict(sha256=entry['sha256']))])))
    review = dict(raw_files=[entry], metadata=dict(path=str(metadata), sha256=p.file_digest(metadata)),
                  raw_byte_cap=entry['size'])
    paths = mod.verify_raw(config, review, lambda: None)
    assert paths == [('code_concepts', local)]
    for field, bad in [('path', '../escape.parquet'), ('sha256', '0'*64), ('size', entry['size']+1),
                       ('key', 'formal_logic')]:
        changed = copy.deepcopy(review)
        changed['raw_files'][0][field] = bad
        with pytest.raises(ValueError):
            mod.verify_raw(config, changed, lambda: None)
    changed = copy.deepcopy(review)
    changed['raw_files'].append(entry)
    changed['raw_byte_cap'] = entry['size'] * 2
    with pytest.raises(ValueError, match='duplicate'):
        mod.verify_raw(config, changed, lambda: None)
    changed = copy.deepcopy(review)
    changed['raw_files'][0]['local_path'] = '/home/yoon/not-an-allowed-raw-cache.parquet'
    with pytest.raises(ValueError, match='raw cache'):
        mod.verify_raw(config, changed, lambda: None)
    config['max_raw_bytes'] = 1
    with pytest.raises(ValueError, match='cap'):
        mod.verify_raw(config, review, lambda: None)
    config['max_raw_bytes'] = 1000000000
    local.unlink()
    with pytest.raises(FileNotFoundError):
        mod.verify_raw(config, review, lambda: None)
    invalid = mod.load_config()
    invalid['max_hours'] = 0
    with pytest.raises(ValueError, match='bounds'):
        mod.validate_config(invalid)


class FixtureTokenizer:
    """Handwritten reversible token mapping, NOT the model tokenizer."""
    def __init__(self):
        self.texts = {}

    def encode(self, text, **kwargs):
        assert kwargs.get('truncation') is False
        import hashlib
        value = int(hashlib.sha256(text.encode()).hexdigest()[:7], 16) % 100000 + 100
        self.texts[value] = text
        return [value] * 5000

    def decode(self, ids, **kwargs):
        return self.texts[int(ids[0])]


def fixture_inputs(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    mod = module()
    p = mod.producer()
    config = mod.load_config()
    recipe = p.load_config()
    tok = FixtureTokenizer()
    keys = mod.native().KEYS
    seed_dir = tmp_path/'seed'
    seed_dir.mkdir()
    seed_cat = p.Catalog(seed_dir/'catalog.sqlite', p.BenchmarkMatcher([]))
    old = dict(text='old reserved answer', metadata=dict(problem_prompt='old reserved problem'))
    seed_cat.add('code_concepts', old)
    seed_cat.finalize('seed')
    record = next(seed_cat.records())
    seed_rows = [dict(source='code_concepts', content_hash=record['content_hash'], group_id='a'*64)]
    seed_cat.close()
    paths = []
    for key in keys[:3]:
        rows = []
        for i in range(150):
            text = (f'Question {i}?\nA. yes\nB. no\nAnswer: A' if key == 'formal_logic'
                    else f'{key} handwritten text {i}')
            rows.append(dict(text=text, metadata=dict(problem_prompt=f'{key} problem {i}'), license='cc-by-4.0'))
        if key == 'code_concepts':
            rows.extend([dict(old, license='cc-by-4.0'), dict(text='old reserved variant',
                metadata=old['metadata'], license='cc-by-4.0'), dict(text='forbidden question answer',
                metadata=dict(problem_prompt='benchmark'), license='cc-by-4.0')])
        path = tmp_path/(key+'.parquet')
        pq.write_table(pa.Table.from_pylist(rows), path)
        paths.append((key, path))
    seed = dict(sources={})
    for key in keys:
        writer = p.NativeWriter(seed_dir/key/'validation/corpus', 11, 131072)
        seed['sources'][key] = dict(validation=p.write_documents(
            [dict(tokens=tok.encode('frozen '+key, truncation=False)+[11])], writer, 1))
        seed['sources'][key]['validation'].update(truncated_documents=0)
        for suffix in ('bin', 'idx'):
            seed['sources'][key]['validation'][suffix+'_sha256'] = p.file_digest(
                seed['sources'][key]['validation']['prefix']+'.'+suffix)
    for split in ('validation', 'train'):
        docs = [dict(tokens=tok.encode(f'replay {split} {i}', truncation=False)+[11]) for i in range(100)]
        writer = p.NativeWriter(tmp_path/'replay'/split/'corpus', 11, 131072)
        p.write_documents(docs, writer, sum(len(d['tokens']) for d in docs))
        recipe['replay'][split+'_prefix'] = str(tmp_path/'replay'/split/'corpus')
    proof = dict(benchmark_questions={'forbidden question'}, original_reserved_hashes=set(),
                 seed_rows=seed_rows, seed_catalog=seed_dir/'catalog.sqlite', seed=seed)
    config['max_replay_scan_tokens'] = 2000000
    large = dict(sources={key: dict(train=dict(tokens=10**10), validation=dict(tokens=5001)) for key in keys})
    gate = mod.native().capacity_gate(large, 1, 128, 4096)
    plan = dict(train_iters=1, train_tokens={k:v['required_tokens'] for k,v in gate['sources'].items()})
    return mod, config, recipe, tok, paths, proof, plan


def test_real_parquet_native_expansion_preserves_all_prior_reservations(tmp_path):
    mod, config, recipe, tok, paths, proof, plan = fixture_inputs(tmp_path)
    out = tmp_path/'production'
    status = {}
    result = mod.run_preparation(config, recipe, out, paths, proof, tok, plan, lambda: None, status)
    p = mod.producer()
    p.atomic_json(out/'success.json', dict(phase='prepared_verified', result=result))
    manifest, digest = mod.native().load_prepared_manifest(out/'manifest.json')
    assert digest == result['manifest_sha256']
    mod.native().capacity_gate(manifest, 1, 128, 4096)
    audit = mod.native().audit_reserved_ids(manifest)
    assert set(audit['counts'].values()) == {0}
    reservations = [json.loads(line) for line in (out/'heldout-reservations.jsonl').read_text().splitlines()]
    assert all(row in reservations for row in proof['seed_rows'])
    assert len(reservations) > len(proof['seed_rows'])
    import sqlite3
    with sqlite3.connect(out/'catalog.sqlite') as db:
        splits = dict(db.execute('SELECT text,split FROM docs'))
        assert splits['old reserved answer'] == splits['old reserved variant'] == 'excluded'
        assert splits['forbidden question answer'] == 'excluded'
    for key in mod.native().KEYS:
        assert manifest['sources'][key]['validation']['bin_sha256'] == proof['seed']['sources'][key]['validation']['bin_sha256']
        assert manifest['sources'][key]['train']['tokens'] >= plan['train_tokens'][key]
    assert manifest['source_provenance']['repository'] == recipe['repository']
    assert manifest['source_provenance']['revision'] == recipe['revision']
    assert manifest['source_provenance']['license'] == 'cc-by-4.0'
    assert manifest['replay_already_consumed_by_base'] is True
    assert result['full_bounded_shards_scanned'] is True
    assert result['audit']['training_duplicate_token_documents'] == 0
    assert status['counts']['original_reserved_component_documents'] >= 2
    with pytest.raises(FileExistsError):
        mod.run_preparation(config, recipe, out, paths, proof, tok, plan, lambda: None, {})


@pytest.mark.parametrize('old_catalog_state', ['missing', 'malformed'])
def test_raw_reconstruction_does_not_open_old_catalog(tmp_path, old_catalog_state):
    mod, config, recipe, tok, paths, proof, plan = fixture_inputs(tmp_path)
    old_catalog = proof.pop('seed_catalog')
    if old_catalog_state == 'missing':
        old_catalog.unlink()
    else:
        old_catalog.write_bytes(b'non-authoritative malformed SQLite fixture')
    before = old_catalog.read_bytes() if old_catalog.exists() else None
    out = tmp_path/'raw-reconstructed'
    result = mod.run_preparation(config, recipe, out, paths, proof, tok, plan, lambda: None, {})
    import sqlite3
    with sqlite3.connect(out/'catalog.sqlite') as db:
        prior_tokens = mod.producer().token_digest(tok.encode('old reserved answer', truncation=False)+[11])
        assert db.execute('SELECT 1 FROM reserved_tokens WHERE hash=?', (prior_tokens,)).fetchone()
    assert result['audit']['train_heldout_token_document_overlap'] == 0
    assert (old_catalog.read_bytes() if old_catalog.exists() else None) == before


def test_prior_reservation_missing_from_raw_fails_closed(tmp_path):
    import pyarrow.parquet as pq
    mod, config, recipe, tok, paths, proof, plan = fixture_inputs(tmp_path)
    path = paths[0][1]
    table = pq.read_table(path)
    # The answer variant still exists: identity propagation cannot replace exact text.
    pq.write_table(table.filter([text != 'old reserved answer' for text in table['text'].to_pylist()]), path)
    out = tmp_path/'missing-prior'
    with pytest.raises(ValueError, match='prior reservation missing from rescanned raw catalog'):
        mod.run_preparation(config, recipe, out, paths, proof, tok, plan, lambda: None, {})
    assert not (out/'manifest.json').exists()
    assert not (out/'success.json').exists()
    assert not (out/'code_concepts/train/corpus.bin').exists()


def test_prior_reservation_rehashes_reconstructed_text(tmp_path, monkeypatch):
    mod, config, recipe, tok, paths, proof, plan = fixture_inputs(tmp_path)
    p = mod.producer()
    scan = p.scan_parquet
    def altered_scan(catalog, key, path, progress):
        scan(catalog, key, path, progress)
        catalog.db.execute('UPDATE docs SET text=? WHERE content_hash=?',
                           ('altered text under unchanged hash', proof['seed_rows'][0]['content_hash']))
    monkeypatch.setattr(p, 'scan_parquet', altered_scan)
    out = tmp_path/'hash-mismatch'
    with pytest.raises(ValueError, match='prior reservation reconstructed text SHA256 mismatch'):
        mod.run_preparation(config, recipe, out, paths, proof, tok, plan, lambda: None, {})
    assert not (out/'manifest.json').exists()
    assert not (out/'success.json').exists()
    assert not (out/'code_concepts/train/corpus.bin').exists()


def test_review_gate_exact_dependencies_and_seed_scope(tmp_path):
    mod = module()
    config = mod.load_config()
    with pytest.raises(ValueError, match='parent'):
        mod.verify_review(config, {}, tmp_path/'calibration.json', lambda: None)
    p = mod.producer()
    review = dict(approved=True, files={name:p.file_digest(ROOT/name) for name in mod.REVIEW_FILES})
    review['files'].pop('midtraining/training/longrun_native.py')
    with pytest.raises(ValueError, match='allowlist'):
        mod.verify_review(config, review, tmp_path/'calibration.json', lambda: None)
    review['files']['midtraining/training/longrun_native.py'] = '0'*64
    with pytest.raises(ValueError, match='changed'):
        mod.verify_review(config, review, tmp_path/'calibration.json', lambda: None)
    review['files']['midtraining/training/longrun_native.py'] = p.file_digest(ROOT/'midtraining/training/longrun_native.py')
    # Scope checks precede original corpus reads / expensive payload verification.
    review['raw_files'] = []
    with pytest.raises(ValueError, match='seed shards'):
        mod.verify_review(config, review, tmp_path/'calibration.json', lambda: None)


@pytest.mark.parametrize('reservation_fault', [None, 'digest', 'count'])
def test_review_uses_raw_proof_not_seed_catalog(tmp_path, monkeypatch, reservation_fault):
    mod = module()
    p = mod.producer()
    config = mod.load_config()
    seed_path = tmp_path/'manifest.json'
    seed_path.write_text('{}')
    config['seed_manifest'] = str(seed_path)
    rows = [dict(source='code_concepts', content_hash='a'*64, group_id='b'*64)]
    reservations = seed_path.with_name('heldout-reservations.jsonl')
    reservations.write_text(json.dumps(rows[0])+'\n')
    seed = dict(heldout_reservations_sha256=p.file_digest(reservations))
    if reservation_fault == 'digest':
        seed['heldout_reservations_sha256'] = '0'*64
    seed_path.with_name('success.json').write_text(json.dumps(dict(result=dict(
        heldout_reserved_documents=2 if reservation_fault == 'count' else 1))))
    calibration = tmp_path/'accepted.json'
    calibration.write_text('{}')
    review = dict(approved=True, files={name:p.file_digest(ROOT/name) for name in mod.REVIEW_FILES},
        raw_files=p.load_config()['sources'],
        calibration=dict(path=str(calibration), sha256=p.file_digest(calibration)))
    # Isolate this review layer; benchmark/native/raw verifiers have separate real fixtures.
    monkeypatch.setattr(p, 'verify_review', lambda *args: {})
    monkeypatch.setattr(mod.native(), 'load_prepared_manifest',
                        lambda path: (seed, config['seed_manifest_sha256']))
    monkeypatch.setattr(mod, 'capacity_plan', lambda *args: {})
    monkeypatch.setattr(mod, 'verify_raw', lambda *args: [(key, tmp_path/key) for key in mod.native().KEYS[:3]])
    digest = p.file_digest
    def no_old_catalog(path):
        assert Path(path).name != 'catalog.sqlite', 'old catalog must not even be hashed'
        return digest(path)
    monkeypatch.setattr(p, 'file_digest', no_old_catalog)
    if reservation_fault:
        with pytest.raises(ValueError, match='entire prior heldout reservation '+reservation_fault+' mismatch'):
            mod.verify_review(config, review, calibration, lambda: None)
    else:
        _, proof, _, _ = mod.verify_review(config, review, calibration, lambda: None)
        assert proof['seed_rows'] == rows
        assert 'seed_catalog' not in proof
        assert 'seed_catalog_sha256' not in review


def test_exhaustion_and_reserved_tokens_are_honest_failures(tmp_path):
    mod, config, recipe, tok, paths, proof, plan = fixture_inputs(tmp_path)
    for bad in ([18, 11], [1, 12], [-1, 11], [131072, 11]):
        with pytest.raises(ValueError):
            mod.checked_tokens(bad)
    plan['train_tokens']['code_concepts'] = 10000000
    status = {}
    out = tmp_path/'short'
    with pytest.raises(RuntimeError, match='code_concepts exhausted'):
        mod.run_preparation(config, recipe, out, paths, proof, tok, plan, lambda: None, status)
    assert 0 < status['insufficient_capacity']['actual_unique_train_tokens'] < 10000000
    assert not (out/'manifest.json').exists()
    assert not (out/'success.json').exists()
    config['max_replay_scan_tokens'] = 1
    with pytest.raises(RuntimeError, match='replay scan'):
        mod.run_preparation(config, recipe, tmp_path/'scan-short', paths, proof, tok, plan, lambda: None, {})


def test_execute_rejects_without_writes_and_bounds_leave_failure_receipt(tmp_path):
    mod = module()
    config = mod.load_config()
    with pytest.raises(ValueError, match='parent'):
        mod.execute(config, tmp_path/'denied', tmp_path/'calibration.json', {})
    assert not (tmp_path/'denied').exists()
    # Tiny actual output bound exercises the real supervisor/receipt path before runtime imports.
    config['max_output_bytes'] = 1
    out = tmp_path/'bounded'
    with pytest.raises(RuntimeError, match='storage cap'):
        mod.execute(config, out, tmp_path/'calibration.json', dict(approved=True))
    failure = json.loads((out/'failure.json').read_text())
    assert failure['phase'] == 'failed'
    assert failure['acquisition_bytes'] == 0
    assert not (out/'success.json').exists()
    with pytest.raises(FileExistsError):
        mod.execute(config, out, tmp_path/'calibration.json', dict(approved=True))
    with pytest.raises(ValueError, match='datasets/midtraining'):
        mod.execute(config, Path('/home/yoon/forbidden-production-output'), tmp_path/'calibration.json', dict(approved=True))
    with pytest.raises(ValueError, match='canonical'):
        mod.execute(config, tmp_path/'alternate', tmp_path/'calibration.json', dict(approved=True), tmp_path/'other.yaml')


def test_token_document_duplicates_are_skipped_before_filling_quotas(tmp_path):
    mod, config, recipe, tok, paths, proof, plan = fixture_inputs(tmp_path)
    # Distinct raw documents intentionally collapse to prior heldout / another source tokens.
    original_encode = tok.encode
    frozen_ids = original_encode('frozen code_concepts', truncation=False)
    shared_ids = original_encode('code_concepts handwritten text 0', truncation=False)
    def alias_encode(text, **kwargs):
        if text == 'code_concepts handwritten text 1':
            return frozen_ids.copy()
        if text == 'algorithmic handwritten text 0':
            return shared_ids.copy()
        return original_encode(text, **kwargs)
    tok.encode = alias_encode
    result = mod.run_preparation(config, recipe, tmp_path/'dedup', paths, proof, tok, plan, lambda: None, {})
    assert result['counts']['reserved_token_documents_skipped'] > 0
    assert result['counts']['duplicate_token_documents_skipped'] > 0
    assert result['audit']['train_heldout_token_document_overlap'] == 0
    assert result['audit']['training_duplicate_token_documents'] == 0


def test_watchdog_interrupts_blocking_preflight_and_records_failure(tmp_path, monkeypatch):
    import time
    mod = module()
    config = mod.load_config()
    config['max_hours'] = .00001
    def blocked_review(*args):
        time.sleep(1)  # Deliberately blocked dependency, not simulated timing output.
        pytest.fail('watchdog did not interrupt blocking preflight')
    monkeypatch.setattr(mod, 'verify_review', blocked_review)
    out = tmp_path/'deadline'
    with pytest.raises(TimeoutError, match='signal'):
        mod.execute(config, out, tmp_path/'calibration.json', dict(approved=True))
    assert json.loads((out/'failure.json').read_text())['phase'] == 'failed'
    assert not (out/'success.json').exists()


def test_inert_cli_and_measured_capacity_plan(tmp_path):
    mod = module()
    for flag in ('--help', '--preview'):
        result = subprocess.run([sys.executable, str(SCRIPT), flag], cwd=tmp_path,
                                text=True, capture_output=True)
        assert result.returncode == 0, result.stderr
    assert list(tmp_path.iterdir()) == []
    assert 'transformers' not in mod.__dict__
    config = mod.load_config()
    calibration = dict(status='calibration_save_reload_accepted', all_ranks=8,
                       manifest_sha256=config['seed_manifest_sha256'], calibration=dict(
                           warm_updates=list(range(9, 17)), profiled=False,
                           global_batch_size=128, sequence_length=4096, mean_step_seconds=100.))
    plan = mod.capacity_plan(calibration, config)
    assert plan['train_iters'] == 777
    native = mod.native()
    manifest = {'sources': {key: {'train': {'tokens': value}, 'validation': {'tokens': 4097}}
                            for key, value in plan['train_tokens'].items()}}
    native.capacity_gate(manifest, 777, 128, 4096)
    for key in native.KEYS:
        changed = copy.deepcopy(manifest)
        changed['sources'][key]['train']['tokens'] -= 1
        with pytest.raises(ValueError, match='capacity insufficient'):
            native.capacity_gate(changed, 777, 128, 4096)
    calibration['status'] = 'progress'
    with pytest.raises(ValueError, match='accepted'):
        mod.capacity_plan(calibration, config)
    result = subprocess.run([sys.executable, str(SCRIPT), '--execute'], cwd=tmp_path,
                            text=True, capture_output=True)
    assert result.returncode != 0 and '--reviewed-sources' in result.stderr


def test_marker_benchmark_fast_path_preserves_literal_matching(monkeypatch):
    import re
    mod = module()
    original = mod.producer().BenchmarkMatcher
    questions = ['한국어 벤치마크 질문입니다', 'ASCII exam question 123456789', 'K', 'Straße', '数学記号π']
    baseline = original(questions)
    texts = ['plain source text', 'prefix ASCII EXAM question 123456789 suffix',
             '앞부분 한국어 벤치마크 질문입니다 뒤부분', 'K', 'STRASSE', '다른 질문',
             '• English source — π examples', '数学記号π']
    expected = [baseline.matches(text) for text in texts]
    calls = []
    real_matches = original.matches
    def observed(self, text):
        calls.append(text)
        return real_matches(self, text)
    monkeypatch.setattr(original, 'matches', observed)
    fast = mod.ProductionBenchmarkMatcher(questions)
    assert [fast.matches(text) for text in texts] == expected
    marker = re.compile(r'[\uac00-\ud7a3]')
    assert len(calls) == sum(marker.search(mod.producer().normalize_question(text)) is not None for text in texts)


def test_per_document_progress_does_not_rescan_output_tree(tmp_path, monkeypatch):
    import time
    mod = module()
    config = mod.load_config()
    out = tmp_path/'progress-cost'
    original = Path.rglob
    walks = []
    def counted(path, pattern):
        if path == out:
            walks.append(pattern)
        return original(path, pattern)
    monkeypatch.setattr(Path, 'rglob', counted)
    monkeypatch.setattr(time, 'monotonic', lambda: 1000.0)
    def review(*args):
        progress = args[-1]
        for _ in range(100):
            progress()
        assert len(walks) == 1, 'per-document progress performed repeated filesystem scans'
        raise RuntimeError('end read-only fixture before tokenizer import')
    monkeypatch.setattr(mod, 'verify_review', review)
    with pytest.raises(RuntimeError, match='end read-only fixture'):
        mod.execute(config, out, tmp_path/'calibration.json', dict(approved=True))
    assert len(walks) == 1
    assert (out/'failure.json').exists()
    assert not (out/'success.json').exists()
