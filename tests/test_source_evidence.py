"""Read-only source evidence tests; real saved artifacts, no GPU imports."""
import hashlib
import json
import pytest
from pathlib import Path


def _fixture(tmp_path, data=b'def host():\n    return 1\n', first=1, last=2):
    file_hash = hashlib.sha256(data).hexdigest()
    lines = data.decode('utf-8', errors='replace').splitlines(keepends=True)
    snippet_hash = hashlib.sha256(''.join(lines[first-1:last]).encode()).hexdigest()
    ref = dict(path='host.py', start_line=first, end_line=last,
               file_sha256=file_hash, snippet_sha256=snippet_hash,
               snapshot='sources/' + file_hash + '.py')
    ref['source_id'] = hashlib.sha256(json.dumps({k: ref[k] for k in (
        'path', 'start_line', 'end_line', 'file_sha256', 'snippet_sha256')}, sort_keys=True).encode()).hexdigest()
    (tmp_path / 'sources').mkdir()
    (tmp_path / ref['snapshot']).write_bytes(data)
    record = {'bridge_id': 'actual', 'source': ref}
    (tmp_path / 'operations.jsonl').write_text(json.dumps(record) + '\n')
    return record


def _rejected(source):
    assert source['status'] in ('rejected', 'unavailable')
    assert source['snippet'] == ''
    assert source['path'] is None
    assert source['file_sha256'] is None
    assert 'SECRET' not in source['warning']


@pytest.mark.parametrize('field,value', [
    ('path', '../SECRET.py'), ('path', '/SECRET.py'), ('path', 'a\\\\SECRET.py'),
    ('path', 'a//b.py'), ('snapshot', '../SECRET.py'), ('source_id', '0'*64),
    ('file_sha256', '0'*64), ('snippet_sha256', '0'*64),
    ('start_line', True), ('start_line', 0), ('start_line', 1.0), ('end_line', 99)])
def test_bridge_tampering(tmp_path, field, value):
    from dashboard.source_evidence import verified_bridge_source
    record = _fixture(tmp_path)
    record['source'][field] = value
    (tmp_path / 'operations.jsonl').write_text(json.dumps(record) + '\n')
    _rejected(verified_bridge_source(tmp_path, Path('operations.jsonl'), record))


@pytest.mark.parametrize('data,first,last', [
    (bytes([255, 254]), 1, 1), (b'x=1\n'*61, 1, 61),
    (b'#'+b'x'*12288, 1, 1), (b'#'+b'x'*524288, 1, 1)])
def test_bridge_encoding_and_size(tmp_path, data, first, last):
    from dashboard.source_evidence import verified_bridge_source
    record = _fixture(tmp_path, data, first, last)
    _rejected(verified_bridge_source(tmp_path, Path('operations.jsonl'), record))


def test_missing_manifest_and_snapshot(tmp_path):
    from dashboard.source_evidence import verified_bridge_source
    record = _fixture(tmp_path)
    (tmp_path / 'operations.jsonl').unlink()
    _rejected(verified_bridge_source(tmp_path, Path('operations.jsonl'), record))
    (tmp_path / record['source']['snapshot']).unlink()
    _rejected(verified_bridge_source(tmp_path, Path('operations.jsonl'), record))


def test_source_symlinks(tmp_path):
    from dashboard.source_evidence import verified_bridge_source
    record = _fixture(tmp_path)
    snapshot = tmp_path / record['source']['snapshot']
    saved = tmp_path / 'saved.py'
    snapshot.rename(saved)
    snapshot.symlink_to(saved)
    _rejected(verified_bridge_source(tmp_path, Path('operations.jsonl'), record))
    snapshot.unlink()
    saved.rename(snapshot)
    directory = tmp_path / 'sources'
    directory.rename(tmp_path / 'other')
    directory.symlink_to(tmp_path / 'other', target_is_directory=True)
    _rejected(verified_bridge_source(tmp_path, Path('operations.jsonl'), record))


@pytest.mark.parametrize('text', [
    "# with nvphase('forward', update):\npass\n",
    "with other('forward', 1):\n pass\n",
    "with nvphase(label, 1):\n pass\n",
    "with nvphase('forward', 1):\n pass\nwith nvphase('forward', 2):\n pass\n",
    "bad syntax SECRET\n", "with nvphase('forward', 1):\n" + ' pass\n'*61])
def test_native_requires_exact_unique_marker(tmp_path, text):
    from dashboard.source_evidence import verified_native_phase_source
    path = tmp_path / 'native.py'
    path.write_text(text)
    _rejected(verified_native_phase_source(tmp_path, Path('native.py'), hashlib.sha256(path.read_bytes()).hexdigest(), 'forward'))


def test_native_drift_and_bad_inputs(tmp_path):
    from dashboard.source_evidence import verified_native_phase_source
    path = tmp_path / 'native.py'
    path.write_bytes((ROOT / NATIVE).read_bytes())
    assert verified_native_phase_source(tmp_path, path, NATIVE_HASH, 'forward')['status'] == 'host_scope_verified'
    path.write_bytes(path.read_bytes() + b'# changed SECRET\n')
    _rejected(verified_native_phase_source(tmp_path, path, NATIVE_HASH, 'forward'))
    for name, digest, phase in [('native.py', 'invalid', 'forward'), ('../SECRET.py', NATIVE_HASH, 'forward'),
                                 ('native.py', NATIVE_HASH, 'unknown'), ('missing.py', NATIVE_HASH, 'data')]:
        _rejected(verified_native_phase_source(tmp_path, Path(name), digest, phase))
    path.unlink()
    path.symlink_to(ROOT / NATIVE)
    _rejected(verified_native_phase_source(tmp_path, path, NATIVE_HASH, 'data'))


def test_cache_unchanged_and_changed_snapshot(tmp_path, monkeypatch):
    import dashboard.source_evidence as module
    record = _fixture(tmp_path)
    assert module.verified_bridge_source(tmp_path, Path('operations.jsonl'), record)['status'] == 'snapshot_verified'
    def forbidden(*args, **kwargs):
        raise AssertionError('unchanged evidence should use bounded cache')
    with monkeypatch.context() as context:
        context.setattr(module, '_read_uncached', forbidden)
        assert module.verified_bridge_source(tmp_path, Path('operations.jsonl'), record)['status'] == 'snapshot_verified'
    snapshot = tmp_path / record['source']['snapshot']
    snapshot.write_bytes(b'def host():\n    return 2\n')
    _rejected(module.verified_bridge_source(tmp_path, Path('operations.jsonl'), record))


@pytest.mark.parametrize('name', ['.env', 'settings.txt', '.env/native.py'])
def test_native_refuses_non_source_before_read(tmp_path, monkeypatch, name):
    import dashboard.source_evidence as module
    def forbidden(*args, **kwargs):
        raise AssertionError('must not read non-source files')
    monkeypatch.setattr(module, '_read', forbidden)
    _rejected(module.verified_native_phase_source(tmp_path, Path(name), NATIVE_HASH, 'data'))


@pytest.mark.parametrize('data', [bytes([255, 254]), b'#'+b'x'*524288])
def test_native_invalid_encoding_or_oversize(tmp_path, data):
    from dashboard.source_evidence import verified_native_phase_source
    (tmp_path / 'native.py').write_bytes(data)
    _rejected(verified_native_phase_source(tmp_path, Path('native.py'), hashlib.sha256(data).hexdigest(), 'data'))


ROOT = Path(__file__).resolve().parents[1]
OPERATIONS = Path('results/gpu_validation/20261013-124945/bridge_cuda/operations.jsonl')
NATIVE = Path('results/e2e/20261013-1539/profile_retry/native_profile.py')
NATIVE_HASH = '4a477a375757202812bf26f93ba2c68571928957f30025c8c7bb6126109a4325'
FIELDS = {'status', 'kind', 'path', 'function', 'start_line', 'end_line',
          'file_sha256', 'snippet_sha256', 'source_id', 'snippet', 'warning'}


def test_real_native_phases():
    from dashboard.source_evidence import verified_native_phase_source
    for phase, first, last in [('data', 124, 129), ('forward', 130, 131),
                               ('backward+sync', 132, 133), ('optimizer', 134, 135)]:
        source = verified_native_phase_source(ROOT, NATIVE, NATIVE_HASH, phase)
        assert set(source) == FIELDS
        assert source['status'] == 'host_scope_verified'
        assert source['kind'] == 'host_scope'
        assert (source['start_line'], source['end_line']) == (first, last)
        assert source['file_sha256'] == NATIVE_HASH
        assert source['source_id'] is None
        assert 'kernel' in source['warning']
        lines = (ROOT / NATIVE).read_text().splitlines(keepends=True)
        assert source['snippet'] == ''.join(lines[first-1:last])
        assert hashlib.sha256(source['snippet'].encode()).hexdigest() == source['snippet_sha256']


def test_real_bridge_refs():
    from dashboard.source_evidence import verified_bridge_source
    records = [json.loads(line) for line in (ROOT / OPERATIONS).read_text().splitlines()]
    assert len(records) == 21
    for record in records:
        source = verified_bridge_source(ROOT, OPERATIONS, record)
        assert set(source) == FIELDS
        assert source['status'] == 'snapshot_verified'
        assert source['kind'] == 'explicit_bridge_operation'
        assert source['source_id'] == record['source']['source_id']
        assert source['function']
        assert 'educational' in source['warning']
        assert 'kernel' in source['warning']
        assert hashlib.sha256(source['snippet'].encode()).hexdigest() == source['snippet_sha256']
        snapshot = ROOT / OPERATIONS.parent / record['source']['snapshot']
        lines = snapshot.read_text().splitlines(keepends=True)
        assert source['snippet'] == ''.join(lines[source['start_line']-1:source['end_line']])
