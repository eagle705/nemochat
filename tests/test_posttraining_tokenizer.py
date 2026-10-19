"""Fail-closed, row-preserving tokenizer migration contracts."""
import copy
import importlib
import numpy as np
import pytest


def api():
    try:
        return importlib.import_module('nemochat.tokenization.posttraining')
    except ModuleNotFoundError:
        pytest.fail('posttraining tokenizer migration API is not implemented')


def tiny_tokenizer():
    return {'model': {'vocab': {f't{i}': i for i in range(32)}, 'merges': []},
            'added_tokens': [{'id': i, 'content': f't{i}', 'special': True} for i in range(32)],
            'normalizer': None, 'pre_tokenizer': {}, 'decoder': {}}


def test_compare_all_ids_allows_only_reserved_names():
    old = tiny_tokenizer()
    new = copy.deepcopy(old)
    for i in range(18, 30):
        del new['model']['vocab'][f't{i}']
        new['model']['vocab'][f'new{i}'] = i
        new['added_tokens'][i]['content'] = f'new{i}'
    result = api().compare_tokenizers(old, new, expected_vocab=32)
    assert result['changed_ids'] == list(range(18, 30))
    assert result['control_tokens_unchanged'] is True
    bad = copy.deepcopy(new)
    bad['model']['merges'] = [['x', 'y']]
    with pytest.raises(ValueError, match='structure'):
        api().compare_tokenizers(old, bad, expected_vocab=32)
    bad = copy.deepcopy(new)
    bad['added_tokens'][18]['special'] = False
    with pytest.raises(ValueError, match='structure'):
        api().compare_tokenizers(old, bad, expected_vocab=32)
    with pytest.raises(ValueError, match='changed IDs'):
        api().compare_tokenizers(old, old, expected_vocab=32)


def test_scan_covers_all_tokens_inputs_labels_and_reports_positions():
    tokens = np.array([[11, 18, 42, 29], [19, 42, 18, 11]], dtype=np.int32)
    result = api().scan_reserved_ids(tokens, chunk_tokens=3)
    assert result['stored_counts']['18'] == 2
    assert result['stored_counts']['19'] == 1
    assert result['stored_counts']['29'] == 1
    assert result['input_counts']['29'] == 0
    assert result['label_counts']['19'] == 0
    assert result['first_positions']['18'] == [0, 1]
    assert result['tokens_scanned'] == 8
    assert result['status'] == 'blocked'
    clean = api().scan_reserved_ids(np.array([11, 42, 11]), chunk_tokens=2)
    assert clean['status'] == 'passed'
    assert sum(clean['stored_counts'].values()) == 0


def test_indexed_dtype_is_read_from_header(tmp_path):
    import struct
    prefix = tmp_path / 'corpus'
    prefix.with_suffix('.idx').write_bytes(b'MMIDIDX\x00\x00' + struct.pack('<Q', 1) + bytes([4]))
    np.array([11, 131056, 42], dtype='<i4').tofile(prefix.with_suffix('.bin'))
    array, metadata = api().open_indexed_tokens(prefix)
    assert array.tolist() == [11, 131056, 42]
    assert metadata['dtype_code'] == 4
    assert metadata['dtype'] == 'int32'
    prefix.with_suffix('.idx').write_bytes(b'MMIDIDX\x00\x00' + struct.pack('<Q', 1) + bytes([3]))
    with pytest.raises(ValueError, match='int32'):
        api().open_indexed_tokens(prefix)


def test_manifest_loader_rejects_unpassed_or_nonzero_audits(tmp_path):
    import json
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps({'status': 'pending'}))
    with pytest.raises(ValueError, match='passed'):
        api().load_posttraining_tokenizer(path)
    path.write_text(json.dumps({'status': 'passed', 'vocab_size': 131072,
        'allowed_changed_ids': list(range(18, 30)), 'control_tokens_unchanged': True,
        'datasetReservedCounts': {'train': {'stored_counts': {'18': 1}}}}))
    with pytest.raises(ValueError, match='reserved'):
        api().load_posttraining_tokenizer(path)


def test_render_chat_preserves_tools_and_uses_official_no_thinking():
    class Recorder:
        def apply_chat_template(self, messages, **kwargs):
            assert messages[0]['tool_calls'][0]['id'] == 'call_42'
            assert kwargs['enable_thinking'] is False
            assert kwargs['tokenize'] is False
            assert kwargs['add_generation_prompt'] is False
            assert kwargs['tools'][0]['type'] == 'function'
            return 'official'
    assert api().render_chat(Recorder(), [{'role': 'assistant', 'tool_calls': [{'id': 'call_42'}]}],
        tools=[{'type': 'function'}]) == 'official'


def test_prepare_cli_exposes_explicit_stages():
    import subprocess
    from pathlib import Path
    import sys
    script = Path('sft/data/prepare_tokenizer.py')
    assert script.exists(), 'versioned migration CLI missing'
    result = subprocess.run([sys.executable, str(script), '--help'], capture_output=True, text=True)
    assert result.returncode == 0
    for stage in ['download', 'audit', 'verify-runtime', 'publish']:
        assert stage in result.stdout


def test_old_template_can_be_embedded_in_tokenizer_config(tmp_path):
    import importlib.util
    import json
    spec = importlib.util.spec_from_file_location('prepare_tokenizer', 'sft/data/prepare_tokenizer.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (tmp_path / 'tokenizer_config.json').write_text(json.dumps({'chat_template': 'embedded official'}))
    assert hasattr(module, 'read_chat_template'), 'embedded-template reader missing'
    assert module.read_chat_template(tmp_path) == 'embedded official'
    (tmp_path / 'tokenizer_config.json').write_text('{}')
    assert module.read_chat_template(tmp_path) is None


def test_official_old_download_must_match_cache_before_acceptance(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location('prepare_tokenizer', 'sft/data/prepare_tokenizer.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cache, official = tmp_path / 'cache', tmp_path / 'official'
    cache.mkdir(); official.mkdir()
    for directory in [cache, official]:
        (directory / 'tokenizer.json').write_text('{}')
        (directory / 'tokenizer_config.json').write_text('{}')
    assert hasattr(module, 'verify_old_official'), 'official old-cache provenance gate missing'
    assert module.verify_old_official(cache, official)['cached_backend_identical'] is True
    (official / 'tokenizer.json').write_text('{"unexpected": true}')
    with pytest.raises(ValueError, match='official old tokenizer'):
        module.verify_old_official(cache, official)


def test_loader_rejects_asset_tampering_before_importing_transformers(tmp_path):
    import json
    audits = {name: {'status': 'passed', 'stored_counts': {str(i): 0 for i in range(18, 30)}}
        for name in ['climbmix_train', 'climbmix_validation', 'climbmix_calibration',
                     'existing_sft_train', 'existing_sft_heldout']}
    manifest = {'status': 'passed', 'vocab_size': 131072,
        'allowed_changed_ids': list(range(18, 30)), 'control_tokens_unchanged': True,
        'datasetReservedCounts': audits, 'tokenizer_path': str(tmp_path),
        'backend_comparison': {'new_assets': {'tokenizer.json': {'sha256': 'expected', 'bytes': 3}}}}
    (tmp_path / 'tokenizer.json').write_text('{}')
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='tokenizer asset integrity'):
        api().load_posttraining_tokenizer(path)
    manifest['datasetReservedCounts']['climbmix_train']['stored_counts'] = {}
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='reserved token audit incomplete'):
        api().load_posttraining_tokenizer(path)
