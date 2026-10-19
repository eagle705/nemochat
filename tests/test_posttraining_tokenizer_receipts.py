"""Portable asset-receipt guards; fixtures are not official tokenizer evidence."""
import builtins
import hashlib
import json
import sys
import types

import pytest

from nemochat.tokenization.posttraining import load_posttraining_tokenizer


ASSETS = ('tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja', 'README.md')


@pytest.fixture
def complete_manifest(tmp_path):
    assets = {}
    for name in ASSETS:
        data = ('handwritten ' + name).encode()
        (tmp_path / name).write_bytes(data)
        assets[name] = {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
    manifest = {'status': 'passed', 'vocab_size': 131072,
        'allowed_changed_ids': list(range(18, 30)), 'control_tokens_unchanged': True,
        'datasetReservedCounts': {name: {'status': 'passed',
            'stored_counts': {str(i): 0 for i in range(18, 30)}} for name in
            ['climbmix_train', 'climbmix_validation', 'climbmix_calibration',
             'existing_sft_train', 'existing_sft_heldout']},
        'tokenizer_path': str(tmp_path), 'backend_comparison': {'new_assets': assets}}
    path = tmp_path / 'manifest.json'
    def save():
        path.write_text(json.dumps(manifest))
        return path
    return manifest, save


@pytest.mark.parametrize('omitted', [*ASSETS, 'all_runtime_assets'])
def test_incomplete_receipt_rejected_before_transformers_import(complete_manifest, monkeypatch, omitted):
    manifest, save = complete_manifest
    assets = manifest['backend_comparison']['new_assets']
    if omitted == 'all_runtime_assets':
        manifest['backend_comparison']['new_assets'] = {'README.md': assets['README.md']}
    else:
        del assets[omitted]
    original_import = builtins.__import__
    imports = []
    def guard(name, *args, **kwargs):
        if name == 'transformers' or name.startswith('transformers.'):
            imports.append(name)
            raise AssertionError('Transformers imported before complete asset verification')
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guard)
    with pytest.raises(ValueError, match='tokenizer asset integrity'):
        load_posttraining_tokenizer(save())
    assert imports == []


def test_complete_verified_receipt_preserves_local_load(complete_manifest, monkeypatch):
    manifest, save = complete_manifest
    calls = []
    class Tokenizer:
        eos_token_id = 11
        chat_template = 'handwritten template'
        def __len__(self):
            return 131072
    tokenizer = Tokenizer()
    def load(path, **kwargs):
        calls.append((path, kwargs))
        return tokenizer
    monkeypatch.setitem(sys.modules, 'transformers', types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(from_pretrained=load)))
    assert load_posttraining_tokenizer(save()) is tokenizer
    assert calls == [(manifest['tokenizer_path'], {'local_files_only': True, 'trust_remote_code': False})]


@pytest.mark.parametrize('name', ASSETS)
def test_complete_receipt_tampering_rejected_before_import(complete_manifest, monkeypatch, name):
    manifest, save = complete_manifest
    from pathlib import Path
    (Path(manifest['tokenizer_path']) / name).write_bytes(b'tampered')
    original_import = builtins.__import__
    def guard(module, *args, **kwargs):
        if module == 'transformers' or module.startswith('transformers.'):
            raise AssertionError('Transformers imported before asset hash verification')
        return original_import(module, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guard)
    with pytest.raises(ValueError, match='tokenizer asset integrity failed'):
        load_posttraining_tokenizer(save())
