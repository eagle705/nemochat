"""Fail-closed Super3.5 tokenizer transition; never touches model weights.

All paths in migration manifests are relative to the repository-root CWD.
The data preparation runtime must supply a compatible Transformers version;
no remote Python code or host environment upgrades are permitted.
"""
import copy
import struct
from pathlib import Path

import numpy as np

ALLOWED_CHANGED_IDS = tuple(range(18, 30))


def compare_tokenizers(old, new, expected_vocab=131072):
    """Compare the entire backend, allowing only names at exactly IDs 18–29."""
    maps = []
    for tokenizer in (old, new):
        vocab = tokenizer['model']['vocab']
        inverse = {i: token for token, i in vocab.items()}
        if len(vocab) != expected_vocab or set(inverse) != set(range(expected_vocab)):
            raise ValueError('vocabulary must be bijective and unchanged in size')
        maps.append(inverse)
    changed = [i for i in range(expected_vocab) if maps[0][i] != maps[1][i]]
    if changed != list(ALLOWED_CHANGED_IDS):
        raise ValueError(f'changed IDs must be exactly 18–29, got {changed}')
    normalized = []
    for tokenizer in (old, new):
        item = copy.deepcopy(tokenizer)
        item['model']['vocab'] = {
            f'__reserved_id_{i}__' if i in ALLOWED_CHANGED_IDS else token: i
            for token, i in item['model']['vocab'].items()}
        for added in item['added_tokens']:
            if added['id'] in ALLOWED_CHANGED_IDS:
                added['content'] = f"__reserved_id_{added['id']}__"
        normalized.append(item)
    if normalized[0] != normalized[1]:
        raise ValueError('tokenizer structure differs outside permitted token names')
    return {'vocab_size': expected_vocab, 'changed_ids': changed,
            'control_tokens_unchanged': True, 'backend_structure_unchanged_except_names': True,
            'renames': {str(i): {'old': maps[0][i], 'new': maps[1][i]} for i in changed}}


def open_indexed_tokens(prefix):
    """Read native MCore header; only the actual int32 ClimbMix format is allowed.

    Code 4 = int32 is taken from the executed builder's corpus_sdk.py DType
    enum; the native Writer writes little-endian int32 and code 4.
    """
    prefix = Path(prefix)
    with prefix.with_suffix('.idx').open('rb') as handle:
        if handle.read(9) != b'MMIDIDX\x00\x00':
            raise ValueError('invalid indexed dataset magic')
        if struct.unpack('<Q', handle.read(8))[0] != 1:
            raise ValueError('unsupported indexed dataset version')
        code = struct.unpack('<B', handle.read(1))[0]
    if code != 4:
        raise ValueError(f'expected builder int32 code 4, got {code}')
    binary = prefix.with_suffix('.bin')
    if binary.stat().st_size % 4:
        raise ValueError('partial int32 corpus')
    return np.memmap(binary, dtype='<i4', mode='r'), {
        'dtype': 'int32', 'dtype_code': code, 'bytes': binary.stat().st_size,
        'dtype_evidence': 'results/pretraining/nemochat_baby/climbmix/{prepare.py,corpus_sdk.py}'}


def scan_reserved_ids(tokens, chunk_tokens=4_000_000):
    """Scan every stored ID in bounded chunks, including SFT padding/edges.

    For 2D packed SFT each row shifts independently; 1D corpus views shift
    globally. Stored-ID zero counts are stronger than either shifted view.
    Position examples are row/column for SFT, flat offsets for corpus.
    """
    if tokens.ndim not in (1, 2) or chunk_tokens < 1:
        raise ValueError('expected flat or row-packed tokens and positive chunk size')
    flat = tokens.reshape(-1)
    stored = {str(i): 0 for i in ALLOWED_CHANGED_IDS}
    inputs, labels, first = stored.copy(), stored.copy(), {}
    width = tokens.shape[1] if tokens.ndim == 2 else tokens.size
    for start in range(0, flat.size, chunk_tokens):
        chunk = flat[start:start + chunk_tokens]
        offsets = np.flatnonzero((chunk >= 18) & (chunk <= 29))
        if not offsets.size:
            continue
        positions = offsets + start
        values = chunk[offsets]
        for value in ALLOWED_CHANGED_IDS:
            key = str(value)
            found = positions[values == value]
            stored[key] += int(found.size)
            inputs[key] += int(np.count_nonzero(found % width != width - 1))
            labels[key] += int(np.count_nonzero(found % width != 0))
            if found.size and key not in first:
                pos = int(found[0])
                first[key] = list(divmod(pos, width)) if tokens.ndim == 2 else pos
    return {'status': 'blocked' if sum(stored.values()) else 'passed',
            'tokens_scanned': int(flat.size), 'shape': list(tokens.shape),
            'dtype': str(tokens.dtype), 'stored_counts': stored,
            'input_counts': inputs, 'label_counts': labels, 'first_positions': first,
            'chunk_tokens': chunk_tokens, 'padding_included': True}


def load_posttraining_tokenizer(manifest_path):
    """Load only an accepted migration, locally, without remote code.

    The official Super3.5 TokenizersBackend needs the pinned NeMo image's
    Transformers 5.x; do not upgrade host/training libraries to call this.
    The caller owns assistant span masks and selected-turn loss policy.
    """
    import json
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest.get('status') != 'passed':
        raise ValueError('tokenizer migration must be passed')
    if (manifest.get('vocab_size') != 131072 or
            manifest.get('allowed_changed_ids') != list(ALLOWED_CHANGED_IDS) or
            manifest.get('control_tokens_unchanged') is not True):
        raise ValueError('invalid tokenizer compatibility contract')
    audits = manifest.get('datasetReservedCounts', {})
    if not audits or any(sum(r.get('stored_counts', {}).values()) for r in audits.values()):
        raise ValueError('reserved token audit missing or nonzero')
    required = {'climbmix_train', 'climbmix_validation', 'climbmix_calibration',
                'existing_sft_train', 'existing_sft_heldout'}
    count_keys = {str(i) for i in ALLOWED_CHANGED_IDS}
    if set(audits) != required or any(
            r.get('status') != 'passed' or set(r.get('stored_counts', {})) != count_keys
            for r in audits.values()):
        raise ValueError('reserved token audit incomplete')
    import hashlib
    asset_root = Path(manifest['tokenizer_path'])
    assets = manifest.get('backend_comparison', {}).get('new_assets', {})
    # Match prepare_tokenizer.download's exact receipt contract, including README.
    required_assets = {'tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja', 'README.md'}
    if set(assets) != required_assets:
        raise ValueError('tokenizer asset integrity receipt incomplete or unexpected')
    for name, expected in assets.items():
        if name not in {'tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja', 'README.md'}:
            raise ValueError('tokenizer asset integrity allowlist violated')
        data = (asset_root / name).read_bytes()
        if len(data) != expected['bytes'] or hashlib.sha256(data).hexdigest() != expected['sha256']:
            raise ValueError(f'tokenizer asset integrity failed: {name}')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        manifest['tokenizer_path'], local_files_only=True, trust_remote_code=False)
    if len(tokenizer) != 131072 or tokenizer.eos_token_id != 11:
        raise ValueError('loaded tokenizer vocabulary/EOS mismatch')
    if not tokenizer.chat_template:
        raise ValueError('official chat template missing')
    return tokenizer


def render_chat(tokenizer, messages, *, tools=None, add_generation_prompt=False):
    """Use official Super3.5 Jinja unchanged; preserve all structured fields.

    No legacy role concatenation. No thought generation. Tools and tool-call
    identifiers are passed through, not normalized or discarded. Assistant
    mask construction and train_turns_mask belong to the dataset adapter.
    """
    kwargs = {'tokenize': False, 'enable_thinking': False,
              'add_generation_prompt': add_generation_prompt}
    if tools is not None:
        kwargs['tools'] = tools
    return tokenizer.apply_chat_template(messages, **kwargs)
