"""Bounded CPU corpus preparation. Default/--preview are read-only; opt in with --execute."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / 'midtraining/configs/corpus.yaml'


def load_config(path=DEFAULT_CONFIG):
    import yaml
    return yaml.safe_load(Path(path).read_text())


def validate_config(config):
    expected = [
        ('code_concepts', 'Nemotron-Pretraining-Code-Concepts/part_000060.parquet',
         477960523, 'b3e69c8bb02be54d12e6773a8fde5594af30cd357e539b93326f02cde2ba6559', .6, 6000000),
        ('algorithmic', 'Nemotron-Pretraining-Unconditional-Algorithmic/part_000000.parquet',
         208343534, '706bab9781f6541cb9f40b8af61242e45c6e3dabc0d2f9b3171f08571da37ccb', .1, 1000000),
        ('formal_logic', 'Nemotron-Pretraining-Formal-Logic/part_000000.parquet',
         73074239, 'b8b641309dec27e836db6a2003d3045c6ed060c8d6a23e3a93100e1c8a3a450b', .1, 1000000)]
    fields = ('key', 'path', 'size', 'sha256', 'weight', 'train_tokens')
    actual = [tuple(source.get(k) for k in fields) for source in config['sources']]
    if actual != expected or config['repository'] != 'nvidia/Nemotron-Pretraining-Specialized-v1.1' or config['revision'] != '13fa979be2e7f7e62913eee0ec5e97c8fd6e24af':
        raise ValueError('unapproved source/revision/quota; review a new recipe generation')
    if config['replay']['train_tokens'] != 2000000 or config['replay']['weight'] != .2:
        raise ValueError('replay quota/weight mismatch')
    policy = config['policy']
    if (config['preparation_training_tokens'] != 10000000 or config['production_training_budget'] is not None
            or policy['cycling'] is not False or policy['truncation'] is not False
            or not 759378296 <= policy['acquisition_byte_cap'] <= 1000000000
            or not 0 < policy['max_hours'] <= 6
            or not 0 < policy['validation_tokens_per_source'] <= 100000
            or not 0 < policy['max_output_bytes'] <= 8000000000
            or not 0 < policy['max_replay_scan_tokens'] <= 20000000
            or not 0 < policy['max_document_tokens'] <= 1000000
            or policy['sequence_length'] != 4096
            or policy['heldout_modulus'] != 100 or policy['heldout_bucket'] != 0):
        raise ValueError('bounded pilot policy violation')
    if (config['tokenizer']['revision'] != '4ba52e8ffa79e9693936c1a819ac89f551ceeef6'
            or config['tokenizer']['repository'] != 'nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-Base-BF16'
            or config['tokenizer']['eos_id'] != 11 or config['tokenizer']['vocab_size'] != 131072
            or config['runtime']['transformers'] != '5.12.1'
            or config['runtime']['image'] != 'nvcr.io/nvidia/nemo@sha256:fdd6e9c7929b76c8624ddfea939dda345beaf67695d9c3994994c18f9e79b9bf'):
        raise ValueError('pinned tokenizer/runtime identity mismatch')


def preview(config):
    validate_config(config)
    return dict(execute=False, sources=config['sources'],
                acquisition_bytes=sum(s['size'] for s in config['sources']),
                training_token_quotas={s['key']: s['train_tokens'] for s in
                                      [*config['sources'], config['replay']]},
                production_training_budget=config['production_training_budget'], cycling=False,
                runtime=config['runtime'], preparation_only=True)


def normalize_question(text):
    # Same NFC/whitespace/case policy as src/nemochat/data/nemotron_sft.py.
    import re
    import unicodedata
    return re.sub(r'\s+', '', unicodedata.normalize('NFC', text)).casefold()


class BenchmarkMatcher:
    """Literal normalized substring containment, NOT semantic decontamination."""
    def __init__(self, questions):
        self.prefixes = {}
        for question in questions:
            value = normalize_question(question)
            if not value:
                raise ValueError('empty benchmark question')
            self.prefixes.setdefault(value[:16], set()).add(value)
        self.short = [q for qs in self.prefixes.values() for q in qs if len(q) < 16]

    def matches(self, text):
        value = normalize_question(text)
        if any(q in value for q in self.short):
            return True
        for i in range(max(0, len(value)-15)):
            candidates = self.prefixes.get(value[i:i+16])
            if candidates and any(value.startswith(q, i) for q in candidates):
                return True
        return False


def question_keys(source, row):
    import re
    text = row['text']
    metadata = row.get('metadata') or {}
    if source == 'code_concepts':
        prompt = metadata.get('problem_prompt')
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError('code concept requires metadata.problem_prompt, never uuid')
        questions = [prompt]
    elif source == 'formal_logic':
        # Each source document can concatenate several MCQs. Remove answers/options,
        # not the question. Connect documents sharing ANY stem before choosing splits.
        intro = 'The following are multiple choice questions (with answers) about formal logic.'
        body = text.removeprefix(intro).strip()
        blocks = re.split(r'(?m)^Answer:\s*[^\n]+\s*(?:\n|$)', body)
        questions = []
        for block in blocks:
            if not block.strip():
                continue
            stem = re.split(r'(?m)^A[.)]\s+', block, maxsplit=1)
            if len(stem) != 2 or not stem[0].strip():
                raise ValueError('unrecognized formal question/options format')
            questions.append(stem[0])
        if not questions:
            raise ValueError('no formal question identities')
    else:
        # Unconditional algorithmic has no source problem ID in its metadata.
        # No invented identity from uuid/title: full-text exact duplicate scope only.
        questions = [text]
    namespace = 'problem' if source != 'algorithmic' else 'unconditional_fulltext'
    return [hashlib.sha256((namespace+'\0'+normalize_question(q)).encode()).hexdigest()
            for q in questions]


class Catalog:
    """Disk-backed texts plus in-memory union-find of exact question components."""
    def __init__(self, path, matcher):
        import sqlite3
        from collections import Counter
        self.db = sqlite3.connect(path)
        self.db.execute('CREATE TABLE docs(id INTEGER PRIMARY KEY, source TEXT, text TEXT, content_hash TEXT UNIQUE, group_id TEXT, split TEXT)')
        self.matcher = matcher
        self.counts = Counter()
        self.parent = []
        self.minimum = []
        self.blocked = []
        self.owners = {}
        self.original_reserved_hashes = set()

    def root(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def add(self, source, row):
        self.counts['scanned_rows'] += 1
        text = row.get('text')
        if not isinstance(text, str) or not text.strip():
            self.counts['invalid_rows'] += 1
            return
        try:
            keys = question_keys(source, row)
        except ValueError:
            self.counts['schema_rejections'] += 1
            return
        digest = hashlib.sha256(text.encode()).hexdigest()
        old = self.db.execute('SELECT id FROM docs WHERE content_hash=?', (digest,)).fetchone()
        problem = (row.get('metadata') or {}).get('problem_prompt', '')
        blocked = self.matcher.matches(text+'\n'+problem)
        if blocked:
            self.counts['benchmark_rows'] += 1
        if old:
            i = old[0]
            self.counts['duplicate_rows'] += 1
        else:
            i = len(self.parent)
            self.parent.append(i)
            self.minimum.append(min(keys))
            self.blocked.append(False)
            self.db.execute('INSERT INTO docs VALUES(?,?,?,?,NULL,NULL)', (i, source, text, digest))
        root = self.root(i)
        original_reserved = digest in self.original_reserved_hashes
        self.counts['original_reserved_rows'] += original_reserved
        self.blocked[root] |= int(blocked) | (2 if original_reserved else 0)
        for key in keys:
            if key in self.owners:
                other = self.root(self.owners[key])
                root = self.root(root)
                if root != other:
                    lo, hi = sorted((root, other))
                    self.parent[hi] = lo
                    self.minimum[lo] = min(self.minimum[lo], self.minimum[hi])
                    self.blocked[lo] |= self.blocked[hi]
                    root = lo
            else:
                self.owners[key] = i
            self.minimum[self.root(i)] = min(self.minimum[self.root(i)], key)

    def finalize(self, seed):
        for i in range(len(self.parent)):
            root = self.root(i)
            group = hashlib.sha256((seed+'\0'+self.minimum[root]).encode()).hexdigest()
            split = 'excluded' if self.blocked[root] else ('validation' if int(group, 16) % 100 == 0 else 'train')
            self.counts['benchmark_component_documents'] += bool(self.blocked[root] & 1)
            self.counts['original_reserved_component_documents'] += bool(self.blocked[root] & 2)
            self.db.execute('UPDATE docs SET group_id=?,split=? WHERE id=?', (group, split, i))
        self.db.commit()

    def records(self, source=None):
        sql = 'SELECT id,source,text,content_hash,group_id,split FROM docs'
        args = ()
        if source:
            sql += ' WHERE source=?'
            args = (source,)
        sql += ' ORDER BY id'
        keys = ('id', 'source', 'text', 'content_hash', 'group_id', 'split')
        for row in self.db.execute(sql, args):
            yield dict(zip(keys, row))

    def reservations(self):
        for record in self.records():
            if record['split'] == 'validation':
                yield {k: record[k] for k in ('source', 'content_hash', 'group_id')}

    def close(self):
        self.db.close()


_SDK = None
SDK_PATH = ROOT / 'results/pretraining/nemochat_baby/climbmix/corpus_sdk.py'
SDK_SHA256 = 'add90e4c4ac95df509d89b7e246002daf291eab775038755a01d5e13d5dd9613'


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def native_sdk():
    """Verbatim preserved MCore indexed reader/builder, CPU/cloud import isolation.

    No training loop, format reimplementation, installed edits or torch/CUDA import.
    Matches the existing original corpus preparation's documented AST isolation.
    """
    global _SDK
    if _SDK is None:
        import ast
        import types
        if file_digest(SDK_PATH) != SDK_SHA256:
            raise ValueError('preserved native corpus SDK source digest mismatch')
        tree = ast.parse(SDK_PATH.read_text())
        tree.body = [node for node in tree.body if not (
            isinstance(node, ast.ImportFrom) and (node.module or '').startswith('megatron'))
            and not (isinstance(node, ast.Import) and any(a.name == 'torch' for a in node.names))]
        ns = dict(torch=types.SimpleNamespace(utils=types.SimpleNamespace(data=types.SimpleNamespace(Dataset=object)), Tensor=object),
                  S3Config=object, ObjectStorageConfig=object, maybe_msc=types.SimpleNamespace(open=open),
                  MultiStorageClientFeature=object, log_single_rank=lambda *a, **k: None,
                  is_object_storage_path=lambda path: False)
        exec(compile(tree, str(SDK_PATH), 'exec'), ns)
        _SDK = types.SimpleNamespace(**ns)
    return _SDK


class NativeWriter:
    def __init__(self, prefix, eos_id, vocab_size):
        import numpy as np
        self.prefix = Path(prefix)
        self.eos_id, self.vocab_size = eos_id, vocab_size
        if any(Path(str(prefix)+suffix).exists() for suffix in ('.bin', '.idx', '.partial.bin', '.partial.idx')):
            raise FileExistsError(prefix)
        self.prefix.parent.mkdir(parents=True, exist_ok=True)
        self.builder = native_sdk().IndexedDatasetBuilder(str(prefix)+'.partial.bin', dtype=np.int32)
        self.tokens = self.documents = 0

    def append(self, tokens):
        import numpy as np
        ids = np.asarray(tokens, dtype=np.int64)
        if ids.ndim != 1 or len(ids) == 0 or ids[-1] != self.eos_id or ids.min() < 0 or ids.max() >= self.vocab_size:
            raise ValueError('token range/EOS violation')
        self.builder.add_document(ids, [len(ids)])
        self.tokens += len(ids)
        self.documents += 1

    def close(self):
        if not self.builder.data_file.closed:
            self.builder.data_file.close()

    def publish(self):
        import os
        self.builder.finalize(str(self.prefix)+'.partial.idx')
        os.rename(str(self.prefix)+'.partial.bin', str(self.prefix)+'.bin')
        os.rename(str(self.prefix)+'.partial.idx', str(self.prefix)+'.idx')
        reader = native_sdk().IndexedDataset(str(self.prefix))
        if len(reader) != self.documents or int(reader.sequence_lengths.sum()) != self.tokens:
            raise RuntimeError('native reader count verification failed')
        for i in range(len(reader)):
            ids = reader[i]
            if ids[-1] != self.eos_id or ids.min() < 0 or ids.max() >= self.vocab_size:
                raise ValueError('native readback range/EOS violation')
        return dict(prefix=str(self.prefix.resolve()), tokens=self.tokens, documents=self.documents,
                    reader_verified=True, dtype='int32', bin_bytes=Path(str(self.prefix)+'.bin').stat().st_size,
                    idx_bytes=Path(str(self.prefix)+'.idx').stat().st_size)


def write_documents(documents, writer, target):
    """Stop after the first complete document crossing the target. Never cycle."""
    try:
        for document in documents:
            if writer.tokens >= target:
                break
            writer.append(document['tokens'])
        if writer.tokens < target:
            raise RuntimeError('source exhausted below token quota; no cycling')
        receipt = writer.publish()
        receipt.update(target_tokens=target, overshoot_tokens=writer.tokens-target, truncated_documents=0)
        return receipt
    finally:
        writer.close()


def download_one(source, dest, url, budget, progress=lambda: None):
    """Exact SHA/length, resumable Range, bounded transfer including retry bytes."""
    import os
    import re
    import urllib.request
    dest = Path(dest)
    if dest.exists():
        if dest.stat().st_size != source['size'] or file_digest(dest) != source['sha256']:
            raise ValueError('cached source size/digest mismatch')
        return dict(path=str(dest), sha256=source['sha256'], size=source['size'], cached=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(str(dest)+'.part')
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > source['size']:
        raise ValueError('partial file exceeds pinned length')
    if source['size']-offset > budget['cap']-budget['used']:
        raise RuntimeError('acquisition transfer budget exhausted')
    if offset < source['size']:
        headers = {'Accept-Encoding': 'identity'}
        if offset:
            headers['Range'] = f'bytes={offset}-{source["size"]-1}'
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as response:
            if offset or response.status == 206:
                match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', response.headers.get('Content-Range', ''))
                if response.status != 206 or not match or tuple(map(int, match.groups())) != (offset, source['size']-1, source['size']):
                    raise ValueError('exact pinned Content-Range required for resume')
            elif response.status != 200:
                raise ValueError('HTTP 200/206 required')
            if int(response.headers.get('Content-Length', '-1')) != source['size']-offset:
                raise ValueError('pinned HTTP length mismatch')
            with partial.open('ab') as stream:
                while offset < source['size']:
                    progress()
                    block = response.read(min(65536, source['size']-offset))
                    if not block:
                        raise OSError('interrupted transfer; .part retained for exact Range resume')
                    if budget['used']+len(block) > budget['cap']:
                        raise RuntimeError('acquisition transfer budget exhausted')
                    stream.write(block)
                    offset += len(block)
                    budget['used'] += len(block)
                    progress()
                stream.flush()
                os.fsync(stream.fileno())
    if file_digest(partial) != source['sha256']:
        raise ValueError('source digest mismatch; preserve failed partial, do not silently redownload')
    os.rename(partial, dest)
    return dict(path=str(dest), sha256=source['sha256'], size=source['size'], cached=False)


def scan_parquet(catalog, source, path, progress):
    import pyarrow.parquet as pq
    parquet = pq.ParquetFile(path)
    if not {'text', 'metadata', 'license'} <= set(parquet.schema_arrow.names):
        raise ValueError('reviewed text/metadata/license schema required')
    # Project only identity metadata, not multi-KB generated_solution fields.
    columns = ['text', 'license']
    columns += ['metadata.problem_prompt'] if source == 'code_concepts' else []
    for batch in parquet.iter_batches(batch_size=128, columns=columns, use_threads=False):
        for row in batch.to_pylist():
            if row.get('license') != 'cc-by-4.0':
                catalog.counts['license_rejections'] += 1
                catalog.counts['scanned_rows'] += 1
                continue
            catalog.add(source, row)
        catalog.db.commit()
        progress()
    catalog.counts['files_scanned_complete'] += 1


def token_digest(tokens):
    import numpy as np
    return hashlib.sha256(np.asarray(tokens, dtype='<i4').tobytes()).hexdigest()


def prepare_specialized(catalog, tokenizer, output, config, progress, status=None):
    status = status if status is not None else {}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    catalog.db.execute('CREATE TABLE reserved_tokens(hash TEXT PRIMARY KEY)')
    catalog.db.execute('CREATE TABLE selected_tokens(hash TEXT PRIMARY KEY)')
    with (output/'heldout-reservations.jsonl').open('x') as stream:
        for reservation in catalog.reservations():
            stream.write(json.dumps(reservation)+'\n')
    result = {}
    for source in config['sources']:
        key = source['key']
        writers = {split: NativeWriter(output/key/split/'corpus', config['tokenizer']['eos_id'], config['tokenizer']['vocab_size'])
                   for split in ('train', 'validation')}
        quotas = dict(train=source['train_tokens'], validation=config['policy']['validation_tokens_per_source'])
        counts = dict(quota_skipped_documents=0, oversized_documents=0, reserved_documents=0)
        try:
            for record in catalog.records(key):
                status.update(current_source=key, active_train_tokens=writers['train'].tokens,
                              active_validation_tokens=writers['validation'].tokens,
                              active_train_documents=writers['train'].documents,
                              active_validation_documents=writers['validation'].documents)
                progress()
                split = record['split']
                if split == 'excluded':
                    continue
                if split == 'validation':
                    counts['reserved_documents'] += 1
                if split == 'train' and writers[split].tokens >= quotas[split]:
                    counts['quota_skipped_documents'] += 1
                    continue
                ids = tokenizer.encode(record['text'], add_special_tokens=False, truncation=False)
                ids.append(config['tokenizer']['eos_id'])
                if len(ids) > config['policy']['max_document_tokens']:
                    counts['oversized_documents'] += 1
                    continue
                digest = token_digest(ids)
                if split == 'validation':
                    # Reserve every full heldout document, including beyond val quota.
                    catalog.db.execute('INSERT OR IGNORE INTO reserved_tokens VALUES(?)', (digest,))
                if writers[split].tokens >= quotas[split]:
                    counts['quota_skipped_documents'] += 1
                    continue
                writers[split].append(ids)
                catalog.db.execute('INSERT OR IGNORE INTO selected_tokens VALUES(?)', (digest,))
            for split, writer in writers.items():
                if writer.tokens < quotas[split]:
                    raise RuntimeError(f'{key}/{split} exhausted: {writer.tokens} < {quotas[split]}; no cycling or substitute')
                counts[split] = writer.publish()
                counts[split].update(target_tokens=quotas[split], overshoot_tokens=writer.tokens-quotas[split],
                                     truncated_documents=0)
            result[key] = counts
            catalog.db.commit()
        finally:
            for writer in writers.values():
                writer.close()
    return result


def prepare_replay(catalog, tokenizer, output, config, progress):
    import numpy as np
    replay = config['replay']
    readers = {split: native_sdk().IndexedDataset(str(resolve_path(replay[split+'_prefix'])))
               for split in ('validation', 'train')}
    if any(reader.index.dtype != np.int32 for reader in readers.values()):
        raise ValueError('reviewed ClimbMix int32 native source required')
    counts = dict(already_consumed_by_base=True, novel_unique_token_claim=False,
                  scanned_tokens=0, scanned_documents=0, benchmark_excluded_documents=0,
                  reserved_excluded_documents=0, duplicate_documents=0, oversized_documents=0,
                  upstream_clipped_tail_omitted_documents=0)
    selected = set()
    for split in ('validation', 'train'):
        reader = readers[split]
        writer = NativeWriter(Path(output)/'climbmix_replay'/split/'corpus', config['tokenizer']['eos_id'], config['tokenizer']['vocab_size'])
        quota = replay['train_tokens'] if split == 'train' else config['policy']['validation_tokens_per_source']
        try:
            for i in range(len(reader)):
                progress()
                ids = reader[i]
                counts['scanned_tokens'] += len(ids)
                counts['scanned_documents'] += 1
                if counts['scanned_tokens'] > config['policy']['max_replay_scan_tokens']:
                    raise RuntimeError('bounded replay scan exhausted; no cycling')
                if not len(ids) or ids[-1] != config['tokenizer']['eos_id']:
                    raise ValueError('upstream replay EOS mismatch')
                digest = token_digest(ids)
                if split == 'validation':
                    # ALL existing validation tokens, not merely the selected prefix.
                    catalog.db.execute('INSERT OR IGNORE INTO reserved_tokens VALUES(?)', (digest,))
                if len(ids) > config['policy']['max_document_tokens']:
                    counts['oversized_documents'] += 1
                    continue
                if replay.get('omit_upstream_clipped_tails', True) and i == len(reader)-1:
                    counts['upstream_clipped_tail_omitted_documents'] += 1
                    continue
                if split == 'validation' and writer.tokens >= quota:
                    continue
                if split == 'train' and catalog.db.execute('SELECT 1 FROM reserved_tokens WHERE hash=?', (digest,)).fetchone():
                    counts['reserved_excluded_documents'] += 1
                    continue
                if digest in selected or catalog.db.execute('SELECT 1 FROM selected_tokens WHERE hash=?', (digest,)).fetchone():
                    counts['duplicate_documents'] += 1
                    continue
                text = tokenizer.decode(ids[:-1].tolist(), skip_special_tokens=False, clean_up_tokenization_spaces=False)
                roundtrip = tokenizer.encode(text, add_special_tokens=False, truncation=False)
                if roundtrip != ids[:-1].tolist():
                    raise ValueError('replay token IDs not roundtrip-compatible with pinned target tokenizer')
                if catalog.matcher.matches(text):
                    counts['benchmark_excluded_documents'] += 1
                    continue
                writer.append(ids)
                selected.add(digest)
                if split == 'train' and writer.tokens >= quota:
                    break
            if writer.tokens < quota:
                raise RuntimeError(f'replay {split} exhausted below quota; no cycling')
            counts[split] = writer.publish()
            counts[split].update(target_tokens=quota, overshoot_tokens=writer.tokens-quota, truncated_documents=0)
        finally:
            writer.close()
    catalog.db.commit()
    return counts


def resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT/path


def verify_review(config, review):
    import re
    if review.get('approved') is not True:
        raise ValueError('explicit parent source review required')
    expected_files = ['midtraining/data/prepare_corpus.py', 'midtraining/configs/corpus.yaml', str(SDK_PATH.relative_to(ROOT))]
    if set(review.get('files', {})) != set(expected_files):
        raise ValueError('exact parent source allowlist required')
    for name, digest in review['files'].items():
        if file_digest(ROOT/name) != digest:
            raise ValueError('source changed after review: '+name)
    if review.get('all_intended_benchmarks_registered') is not True:
        raise ValueError('parent benchmark inventory review is incomplete')
    expected_paths = [config['policy']['kmmlu_path'], *config['policy']['additional_benchmark_manifests']]
    entries = review.get('benchmark_manifests', [])
    if sorted(e['path'] for e in entries) != sorted(expected_paths):
        raise ValueError('complete benchmark inventory allowlist mismatch')
    questions = set()
    for entry in entries:
        path = resolve_path(entry['path'])
        if file_digest(path) != entry['sha256']:
            raise ValueError('benchmark manifest digest mismatch')
        document = json.loads(path.read_text())
        values = document['questions']
        if (not isinstance(values, list) or any(not isinstance(q, str) or not q for q in values)
                or len(values) != entry['count'] or len(set(values)) != len(values)):
            raise ValueError('benchmark question reservation count/schema mismatch')
        if entry['path'] == config['policy']['kmmlu_path'] and (
                len(values) != 34871 or document['revision'] != config['policy']['kmmlu_revision']
                or entry['sha256'] != '61a804e227bd8aedbb4f1818c2559391f122627398e6f80085cd58c2525b5f0b'):
            raise ValueError('complete original KMMLU manifest required')
        questions.update(values)
    replay = config['replay']
    hashes_path = resolve_path(replay['original_heldout_hashes'])
    if file_digest(hashes_path) != replay['original_heldout_sha256']:
        raise ValueError('whole original ClimbMix heldout digest mismatch')
    hashes = json.loads(hashes_path.read_text())
    if (len(hashes) != 84990 or len(set(hashes)) != 84990
            or any(not isinstance(h, str) or not re.fullmatch('[0-9a-f]{64}', h) for h in hashes)):
        raise ValueError('whole original ClimbMix heldout count/schema mismatch')
    receipt_path = resolve_path(replay['preparation_receipt'])
    if file_digest(receipt_path) != replay['preparation_receipt_sha256']:
        raise ValueError('original native replay preparation receipt mismatch')
    original = json.loads(receipt_path.read_text())
    if original['phase'] != 'full_ready' or not original['validation_last_document_truncated'] or not original['last_document_truncated']:
        raise ValueError('review upstream replay clipping/ready contract')
    for split in ('train', 'validation'):
        prefix = resolve_path(replay[split+'_prefix'])
        if str(prefix) != original[split]['prefix']:
            raise ValueError('original replay prefix mismatch')
        for suffix in ('.bin', '.idx'):
            if Path(str(prefix)+suffix).stat().st_size != original[split][suffix[1:]+'_bytes']:
                raise ValueError('original native replay file length mismatch')
    for name, digest in config['tokenizer']['asset_sha256'].items():
        if file_digest(resolve_path(config['tokenizer']['path'])/name) != digest:
            raise ValueError('pinned tokenizer asset mismatch')
    return dict(benchmark_questions=questions, original_reserved_hashes=set(hashes),
                benchmark_manifests=entries, original_replay_receipt=original)


def audit_outputs(catalog, receipts, sequence_length, progress):
    reserved = {row[0] for row in catalog.db.execute('SELECT hash FROM reserved_tokens')}
    for receipt in receipts.values():
        validation = native_sdk().IndexedDataset(receipt['validation']['prefix'])
        reserved.update(token_digest(validation[i]) for i in range(len(validation)))
    seen = set()
    totals = dict(train_tokens=0, train_documents=0, validation_tokens=0,
                  validation_documents=0, train_heldout_token_document_overlap=0,
                  training_duplicate_token_documents=0, capacity_by_source={})
    for source, receipt in receipts.items():
        for split in ('train', 'validation'):
            reader = native_sdk().IndexedDataset(receipt[split]['prefix'])
            tokens = int(reader.sequence_lengths.sum())
            if tokens != receipt[split]['tokens'] or len(reader) != receipt[split]['documents']:
                raise ValueError('published native prefix/receipt count mismatch')
            totals[split+'_tokens'] += tokens
            totals[split+'_documents'] += len(reader)
            for suffix in ('.bin', '.idx'):
                receipt[split][suffix[1:]+'_sha256'] = file_digest(receipt[split]['prefix']+suffix)
            if split == 'train':
                totals['capacity_by_source'][source] = max(0, (tokens-1)//sequence_length)
                for i in range(len(reader)):
                    progress()
                    digest = token_digest(reader[i])
                    if digest in reserved:
                        raise ValueError('training document overlaps full heldout token reservation')
                    if digest in seen:
                        raise ValueError('duplicate training token document across output sources')
                    seen.add(digest)
    return totals


def run_preparation(config, output, source_paths, proof, tokenizer, progress, status):
    """Real Parquet -> full grouping/reservation -> native prefixes -> readback audit."""
    import yaml
    from collections import Counter
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    catalog = Catalog(output/'catalog.sqlite', BenchmarkMatcher(proof['benchmark_questions']))
    catalog.original_reserved_hashes = proof['original_reserved_hashes']
    status['counts'] = catalog.counts
    try:
        status['phase'] = 'scanning_entire_bounded_shards'
        for source in config['sources']:
            status['current_source'] = source['key']
            scan_parquet(catalog, source['key'], source_paths[source['key']], progress)
        status['phase'] = 'freezing_grouped_heldout'
        catalog.finalize(config['policy']['split_seed'])
        components = Counter(catalog.root(i) for i in range(len(catalog.parent)))
        group_statistics = dict(components=len(components), largest_component_documents=max(components.values(), default=0))
        status['phase'] = 'specialized_tokenizing'
        receipts = prepare_specialized(catalog, tokenizer, output, config, progress, status)
        status['prepared_sources'] = receipts
        status['phase'] = 'replay_selecting'
        receipts['climbmix_replay'] = prepare_replay(catalog, tokenizer, output, config, progress)
        status['phase'] = 'final_native_readback_audit'
        audit = audit_outputs(catalog, receipts, config['policy']['sequence_length'], progress)
        weights = [s['weight'] for s in [*config['sources'], config['replay']]]
        keys = [s['key'] for s in [*config['sources'], config['replay']]]
        blend_per_split = [[[receipts[key][split]['prefix'] for key in keys], weights]
                           for split in ('train', 'validation')]+[None]
        fragment = dict(dataset=dict(
            _target_='megatron.bridge.training.config.GPTDatasetConfig',
            seq_length=config['policy']['sequence_length'], random_seed=1234,
            blend=None, blend_per_split=blend_per_split, split=None,
            num_dataset_builder_threads=1, num_workers=0,
            eod_mask_loss=False, reset_position_ids=False, reset_attention_mask=False,
            skip_getting_attention_mask_from_dataset=True))
        with (output/'bridge-dataset-fragment.yaml').open('x') as stream:
            yaml.safe_dump(fragment, stream, sort_keys=False)
        result = dict(sources=receipts, audit=audit, counts=dict(catalog.counts), grouped_identity_statistics=group_statistics,
                    full_bounded_shards_scanned=catalog.counts['files_scanned_complete'] == len(config['sources']),
                    heldout_reservations_sha256=file_digest(output/'heldout-reservations.jsonl'),
                    heldout_reserved_documents=sum(1 for _ in catalog.reservations()),
                    cycling=False, production_training_budget=None, preparation_only=True,
                    native_training_gate='pending_Bridge_config_load_sample_index_no_repeat_and_native_update',
                    replay_already_consumed_by_base=True, assistant_only_loss_mask=False)
        manifest = dict(schema_version=1, status='prepared_verified', sources=receipts,
                        source_order=keys, weights=dict(zip(keys, weights)),
                        sequence_length=config['policy']['sequence_length'], tokenizer=config['tokenizer'],
                        capacity_by_source=audit['capacity_by_source'], audit=audit,
                        heldout_reservations_sha256=result['heldout_reservations_sha256'],
                        cycling=False, production_training_budget=None,
                        companion_success_receipt=str((output/'success.json').resolve()),
                        native_training_gate=result['native_training_gate'])
        atomic_json(output/'manifest.json', manifest)
        result.update(manifest_path=str((output/'manifest.json').resolve()),
                      manifest_sha256=file_digest(output/'manifest.json'))
        return result
    finally:
        catalog.close()


def atomic_json(path, value):
    import os
    path = Path(path)
    temp = Path(str(path)+'.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    os.replace(temp, path)


def execute(config, output, cache, review, local_only=False):
    """Parent-reviewed CPU execution; no model weights, GPUs, telemetry or training."""
    import fcntl
    import importlib.metadata
    import os
    import signal
    import time
    from datetime import datetime, timezone
    validate_config(config)
    if importlib.metadata.version('transformers') != '5.12.1':
        raise RuntimeError('execute requires pinned CPU image with Transformers 5.12.1; do not upgrade host')
    if config != load_config(DEFAULT_CONFIG):
        raise ValueError('execution must consume the exact parent-reviewed canonical corpus.yaml')
    proof = verify_review(config, review)
    os.environ.update(CUDA_VISIBLE_DEVICES='', TOKENIZERS_PARALLELISM='false',
                      HF_HUB_OFFLINE='1', HF_HUB_DISABLE_IMPLICIT_TOKEN='1',
                      HF_HOME=str(ROOT/'datasets/midtraining/specialized-v1.1-13fa979/.hf-home'),
                      OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(resolve_path(config['tokenizer']['path'])),
                    local_files_only=True, trust_remote_code=False, use_fast=True)
    values = tokenizer.get_vocab().values()
    if tokenizer.eos_token_id != 11 or min(values) < 0 or max(values) >= 131072:
        raise ValueError('pinned tokenizer vocabulary/EOS mismatch')
    native_sdk()
    output, cache = Path(output).resolve(), Path(cache).resolve()
    allowed = (ROOT/'datasets/midtraining').resolve()
    if not output.is_relative_to(allowed) or not cache.is_relative_to(allowed):
        raise ValueError('execution writes restricted to repository datasets/midtraining')
    if output.is_relative_to(cache) or cache.is_relative_to(output):
        raise ValueError('fresh output and resumable acquisition cache must be disjoint')
    if output.exists():
        raise FileExistsError('fresh preparation output required; published/failed outputs are immutable')
    cache.mkdir(parents=True, exist_ok=True)
    with (cache/'acquisition.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        output.mkdir(parents=True, exist_ok=False)
        ledger_path = cache/'transfer-ledger.json'
        retained = sum(Path(str(cache/s['path'])+suffix).stat().st_size
                       for s in config['sources'] for suffix in ('', '.part')
                       if Path(str(cache/s['path'])+suffix).exists())
        previous = json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
        budget = dict(used=max(previous.get('used', 0), retained), cap=config['policy']['acquisition_byte_cap'])
        start = time.monotonic()
        status = dict(phase='preflight_complete', pid=os.getpid(), started_at=datetime.now(timezone.utc).isoformat(),
                      preparation_only=True, execute=True, sources=[], failed_attempts=[],
                      runtime_versions={name: importlib.metadata.version(name) for name in ('transformers', 'pyarrow', 'numpy', 'pyyaml')},
                      configured_runtime_image=config['runtime']['image'], image_identity_verified_by_this_process=False,
                      source_review=review, config=config, source_code_sha256=file_digest(Path(__file__)),
                      benchmark_manifest_count=len(proof['benchmark_manifests']),
                      benchmark_questions=len(proof['benchmark_questions']), original_climbmix_reserved_documents=84990,
                      checkpoint_weight_payloads_read=0)
        last_status = [0.0]
        last_budget_written = [None]
        def persist():
            status.update(updated_at=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic()-start,
                          acquisition_payload_bytes=budget['used'], acquisition_payload_byte_cap=budget['cap'])
            atomic_json(output/'progress.json', status)
        def progress():
            now = time.monotonic()
            if now-start > config['policy']['max_hours']*3600:
                raise TimeoutError('bounded preparation wall-time exhausted')
            # Persist every charged chunk, without rewriting an unchanged ledger per row.
            if last_budget_written[0] != budget['used']:
                atomic_json(ledger_path, budget)
                last_budget_written[0] = budget['used']
            if now-last_status[0] >= 5:
                size = sum(p.stat().st_size for root in (output, cache) for p in root.rglob('*') if p.is_file())
                if size > config['policy']['max_output_bytes']:
                    raise RuntimeError('bounded corpus/cache storage cap exceeded')
                persist()
                last_status[0] = now
        def interrupted(signum, frame):
            raise RuntimeError(f'preparation interrupted by signal {signum}; no success receipt')
        old_handlers = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT)}
        signal.setitimer(signal.ITIMER_REAL, config['policy']['max_hours']*3600)
        try:
            persist()
            paths = {}
            for source in config['sources']:
                status.update(phase='acquiring_pinned_parquet', current_source=source['key'])
                dest = cache/source['path']
                if local_only and not dest.is_file():
                    raise FileNotFoundError('parent acquisition not complete; --local-only refuses duplicate download: '+str(dest))
                url = f'https://huggingface.co/datasets/{config["repository"]}/resolve/{config["revision"]}/{source["path"]}'
                for attempt in range(3):
                    try:
                        downloaded = download_one(source, dest, url, budget, progress)
                        break
                    except OSError as error:
                        status['failed_attempts'].append(dict(source=source['key'], attempt=attempt+1,
                                                              error=repr(error), acquisition_payload_bytes=budget['used']))
                        persist()
                        if attempt == 2:
                            raise
                        time.sleep(2**attempt)
                status['sources'].append(downloaded)
                paths[source['key']] = dest
                persist()
            result = run_preparation(config, output, paths, proof, tokenizer, progress, status)
            status.update(phase='prepared_verified', result=result)
            progress()
            persist()
            atomic_json(output/'success.json', status)
            print(json.dumps(dict(status='prepared_verified', receipt=str(output/'success.json'), audit=result['audit']), indent=2))
            return status
        except BaseException as error:
            status.update(phase='failed', error=repr(error))
            atomic_json(ledger_path, budget)
            persist()
            atomic_json(output/'failure.json', status)
            raise
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--preview', action='store_true')
    mode.add_argument('--execute', action='store_true')
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'datasets/midtraining/super35-kd-01/pilot-10m-v1')
    parser.add_argument('--cache', '--raw-directory', dest='cache', type=Path,
                        default=ROOT / 'datasets/midtraining/specialized-v1.1-13fa979/raw')
    parser.add_argument('--local-only', action='store_true', help='reuse completed parent acquisition; never download missing files')
    parser.add_argument('--reviewed-sources', type=Path, help='parent-approved source/benchmark review JSON, execution only')
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if not args.execute:
        print(json.dumps(dict(preview(config), raw_directory=str(args.cache), local_only=args.local_only), indent=2))
        return 0
    if args.reviewed_sources is None:
        parser.error('--execute requires --reviewed-sources; no implicit approval')
    if args.config.resolve() != DEFAULT_CONFIG.resolve():
        parser.error('execution requires the reviewed canonical midtraining/configs/corpus.yaml')
    execute(config, args.output, args.cache, json.loads(args.reviewed_sources.read_text()), args.local_only)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
