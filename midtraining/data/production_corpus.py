#!/usr/bin/env python3
"""Local-only bounded production corpus expansion; preview is inert."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / 'midtraining/configs/production_corpus.yaml'


@lru_cache(maxsize=None)
def helper(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def producer():
    return helper('production_original_producer', 'midtraining/data/prepare_corpus.py')


def native():
    return helper('production_native_consumer', 'midtraining/training/longrun_native.py')


class ProductionBenchmarkMatcher:
    """Identical literal policy with a necessary-character fast rejection.

    Text without any marker character cannot contain a question containing one.
    Non-Hangul Unicode (math, punctuation, CJK) still receives exact matching.
    """
    def __init__(self, questions):
        import re
        self.original = producer().BenchmarkMatcher(questions)
        self.marker = re.compile(r'[\uac00-\ud7a3]')
        self.without_marker_questions = tuple(q for values in self.original.prefixes.values()
                                             for q in values if self.marker.search(q) is None)

    def matches(self, text):
        value = producer().normalize_question(text)
        if self.marker.search(value) is None:
            return any(question in value for question in self.without_marker_questions)
        return self.original.matches(text)


def load_config(path=DEFAULT_CONFIG):
    import yaml
    return yaml.safe_load(Path(path).read_text())


def capacity_plan(calibration, config):
    if (calibration.get('status') != 'calibration_save_reload_accepted'
            or calibration.get('all_ranks') != 8
            or calibration.get('manifest_sha256') != config['seed_manifest_sha256']):
        raise ValueError('accepted all-rank seed calibration required')
    n = native()
    plan = n.derive_horizon(calibration['calibration'])
    samples = plan['train_iters'] * 128
    plan['train_tokens'] = {key: (math.ceil(math.ceil(samples * weight) * 1.005) + 2) * 4096 + 1
                            for key, weight in zip(n.KEYS, n.WEIGHTS)}
    return plan


def validate_config(config):
    expected = load_config()
    if set(config) != set(expected) or config['schema_version'] != 1 or any(
            config[k] != expected[k] for k in ('seed_manifest', 'seed_manifest_sha256')):
        raise ValueError('immutable seed configuration required')
    limits = dict(max_hours=24, max_output_bytes=256000000000, max_raw_bytes=64000000000,
                  max_replay_scan_tokens=100000000000, max_document_tokens=1000000)
    for key, maximum in limits.items():
        if type(config[key]) not in (int, float) or not 0 < config[key] <= maximum:
            raise ValueError('production bounds violation: '+key)


def verify_raw(config, review, progress):
    """No URL opening or cache writes: exact local, officially pinned files only."""
    import re
    p = producer()
    base = p.load_config()
    metadata_entry = review['metadata']
    metadata_path = p.resolve_path(metadata_entry['path'])
    if p.file_digest(metadata_path) != metadata_entry['sha256']:
        raise ValueError('official metadata digest mismatch')
    metadata = json.loads(metadata_path.read_text())
    if metadata.get('id') != base['repository'] or metadata.get('sha') != base['revision']:
        raise ValueError('official metadata repository/revision mismatch')
    official = {e['rfilename']: e for e in metadata['siblings']}
    directories = dict(code_concepts='Nemotron-Pretraining-Code-Concepts',
                       algorithmic='Nemotron-Pretraining-Unconditional-Algorithmic',
                       formal_logic='Nemotron-Pretraining-Formal-Logic')
    seen, local_seen, paths = set(), set(), []
    raw = review['raw_files']
    if not raw:
        raise ValueError('explicit nonempty raw allowlist required')
    total = sum(e['size'] for e in raw)
    if type(review['raw_byte_cap']) is not int or not 0 < total <= review['raw_byte_cap'] <= config['max_raw_bytes']:
        raise ValueError('parent raw byte cap exceeded; no acquisition authorized here')
    for entry in raw:
        progress()
        name = entry['path']
        local = p.resolve_path(entry['local_path']).resolve()
        if not local.is_relative_to((ROOT/'datasets/midtraining').resolve()) or not str(local).startswith('/raid/'):
            raise ValueError('local raw cache must be under /raid repository datasets/midtraining')
        if name in seen or local in local_seen:
            raise ValueError('duplicate raw allowlist path')
        seen.add(name)
        local_seen.add(local)
        directory = directories.get(entry['key'])
        if not directory or not re.fullmatch(re.escape(directory)+r'/part_[0-9]{6}\.parquet', name):
            raise ValueError('ineligible source path')
        pin = official.get(name, {})
        if (type(entry['size']) is not int or entry['size'] <= 0
                or not re.fullmatch('[0-9a-f]{64}', entry['sha256'])
                or pin.get('size') != entry['size']
                or pin.get('lfs', {}).get('sha256') != entry['sha256']):
            raise ValueError('official raw size/SHA256 mismatch')
        if local.stat().st_size != entry['size'] or p.file_digest(local) != entry['sha256']:
            raise ValueError('local raw size/SHA256 mismatch')
        paths.append((entry['key'], local))
    return paths


REVIEW_FILES = (
    'midtraining/data/production_corpus.py', 'midtraining/configs/production_corpus.yaml',
    'midtraining/data/prepare_corpus.py', 'midtraining/configs/corpus.yaml',
    'midtraining/training/longrun_native.py', 'results/pretraining/nemochat_baby/climbmix/corpus_sdk.py',
)


def verify_review(config, review, calibration_path, progress):
    import re
    p = producer()
    if review.get('approved') is not True:
        raise ValueError('explicit parent production review required')
    if set(review.get('files', {})) != set(REVIEW_FILES):
        raise ValueError('exact parent source allowlist required')
    for name, digest in review['files'].items():
        if p.file_digest(ROOT/name) != digest:
            raise ValueError('source changed after review: '+name)
    recipe = p.load_config()
    p.validate_config(recipe)
    seed_paths = {e['path'] for e in recipe['sources']}
    if not seed_paths <= {e['path'] for e in review.get('raw_files', [])}:
        raise ValueError('all original seed shards must be rescanned for variant propagation')
    # Reuse the original, strict full KMMLU/ClimbMix/tokenizer/lineage verifier.
    original_review = dict(review, files={name:review['files'][name] for name in (
        'midtraining/data/prepare_corpus.py', 'midtraining/configs/corpus.yaml', str(p.SDK_PATH.relative_to(ROOT)))})
    proof = p.verify_review(recipe, original_review)
    progress()
    seed_path = p.resolve_path(config['seed_manifest'])
    seed, digest = native().load_prepared_manifest(seed_path)
    if digest != config['seed_manifest_sha256']:
        raise ValueError('immutable seed manifest SHA mismatch')
    reservations = seed_path.with_name('heldout-reservations.jsonl')
    if p.file_digest(reservations) != seed['heldout_reservations_sha256']:
        raise ValueError('entire prior heldout reservation digest mismatch')
    rows = [json.loads(line) for line in reservations.read_text().splitlines()]
    success = json.loads(seed_path.with_name('success.json').read_text())
    if len(rows) != success['result']['heldout_reserved_documents']:
        raise ValueError('entire prior heldout reservation count mismatch')
    for row in rows:
        if (row.get('source') not in native().KEYS or any(
                not re.fullmatch('[0-9a-f]{64}', row.get(k, '')) for k in ('content_hash', 'group_id'))):
            raise ValueError('prior heldout row schema mismatch')

    calibration_path = Path(calibration_path).resolve()
    approved_calibration = review['calibration']
    if (p.resolve_path(approved_calibration['path']).resolve() != calibration_path
            or p.file_digest(calibration_path) != approved_calibration['sha256']):
        raise ValueError('reviewed calibration path/SHA mismatch')
    calibration = json.loads(calibration_path.read_text())
    plan = capacity_plan(calibration, config)
    paths = verify_raw(config, review, progress)
    if {key for key, path in paths} != set(native().KEYS[:3]):
        raise ValueError('all three specialized sources required')
    proof.update(seed=seed, seed_rows=rows, raw_files=review['raw_files'])
    return recipe, proof, paths, plan


def checked_tokens(ids):
    import numpy as np
    values = np.asarray(ids, dtype=np.int64)
    if (values.ndim != 1 or not len(values) or values[-1] != 11
            or values.min() < 0 or values.max() >= 131072):
        raise ValueError('token range/EOS violation')
    if np.any((values >= 18) & (values <= 29)):
        raise ValueError('reserved IDs 18..29 require a separate alias review')
    return values


def run_preparation(config, recipe, output, paths, proof, tokenizer, plan, progress, status):
    """Entire local allowlist -> inherited catalog policy -> native SDK -> audits."""
    import shutil
    from collections import Counter
    p, n = producer(), native()
    output = Path(output)
    # execute owns the directory; the core also supports direct fixture execution.
    output.mkdir(parents=True, exist_ok=True)
    if (output/'catalog.sqlite').exists():
        raise FileExistsError('fresh production catalog required')
    catalog = p.Catalog(output/'catalog.sqlite', ProductionBenchmarkMatcher(proof['benchmark_questions']))
    status['counts'] = catalog.counts
    catalog.original_reserved_hashes = set(proof['original_reserved_hashes'])
    catalog.original_reserved_hashes.update(row['content_hash'] for row in proof['seed_rows'])
    receipts = {}
    scanned_replay = [0]

    def encode(text):
        return checked_tokens(tokenizer.encode(text, add_special_tokens=False, truncation=False)+[11])

    def reserve(ids):
        catalog.db.execute('INSERT OR IGNORE INTO reserved_tokens VALUES(?)', (p.token_digest(ids),))

    def replay_document(reader, i):
        progress()
        ids = checked_tokens(reader[i])
        scanned_replay[0] += len(ids)
        status['replay_scanned_tokens'] = scanned_replay[0]
        if scanned_replay[0] > config['max_replay_scan_tokens']:
            raise RuntimeError('bounded replay scan exhausted; no cycling')
        return ids

    def selected(ids):
        digest = p.token_digest(ids)
        if catalog.db.execute('SELECT 1 FROM reserved_tokens WHERE hash=?', (digest,)).fetchone():
            catalog.counts['reserved_token_documents_skipped'] += 1
            return False
        if catalog.db.execute('SELECT 1 FROM selected_tokens WHERE hash=?', (digest,)).fetchone():
            catalog.counts['duplicate_token_documents_skipped'] += 1
            return False
        catalog.db.execute('INSERT INTO selected_tokens VALUES(?)', (digest,))
        return True

    try:
        status['phase'] = 'scanning_entire_parent_allowlist'
        for key, path in paths:
            status['current_source'] = key
            p.scan_parquet(catalog, key, path, progress)
        catalog.finalize(recipe['policy']['split_seed'])
        components = Counter(catalog.root(i) for i in range(len(catalog.parent)))
        catalog.db.execute('CREATE TABLE reserved_tokens(hash TEXT PRIMARY KEY)')
        catalog.db.execute('CREATE TABLE selected_tokens(hash TEXT PRIMARY KEY)')
        status['phase'] = 'reserving_all_prior_and_additional_heldout'
        with (output/'heldout-reservations.jsonl').open('x') as stream:
            for row in proof['seed_rows']:
                stream.write(json.dumps(row)+'\n')
            previous = {r['content_hash'] for r in proof['seed_rows']}
            for row in catalog.reservations():
                if row['content_hash'] not in previous:
                    stream.write(json.dumps(row)+'\n')
        # Recover ALL prior heldout text from the fully rescanned verified raw files.
        # The non-authoritative old SQLite cache is neither read nor required.
        for row in proof['seed_rows']:
            progress()
            record = catalog.db.execute('SELECT text FROM docs WHERE content_hash=?',
                                        (row['content_hash'],)).fetchone()
            if record is None:
                raise ValueError('prior reservation missing from rescanned raw catalog: '+row['content_hash'])
            if p.hashlib.sha256(record[0].encode()).hexdigest() != row['content_hash']:
                raise ValueError('prior reservation reconstructed text SHA256 mismatch: '+row['content_hash'])
            reserve(encode(record[0]))
        for record in catalog.records():
            progress()
            if record['split'] == 'validation':
                reserve(encode(record['text']))
        # Copy frozen seed validation bytes; never repartition or use for training.
        for key in n.KEYS:
            previous = proof['seed']['sources'][key]['validation']
            prefix = output/key/'validation/corpus'
            prefix.parent.mkdir(parents=True, exist_ok=True)
            for suffix in ('bin', 'idx'):
                progress()
                src, dest = Path(previous['prefix']+'.'+suffix), Path(str(prefix)+'.'+suffix)
                with src.open('rb') as source, dest.open('xb') as target:
                    shutil.copyfileobj(source, target, 1024*1024)
                if p.file_digest(dest) != previous[suffix+'_sha256']:
                    raise ValueError('frozen validation SHA mismatch')
            entry = dict(previous, prefix=str(prefix.resolve()), frozen_seed_heldout_reuse=True)
            reader = p.native_sdk().IndexedDataset(entry['prefix'])
            if len(reader) != entry['documents'] or int(reader.sequence_lengths.sum()) != entry['tokens']:
                raise ValueError('frozen validation native readback mismatch')
            for i in range(len(reader)):
                progress()
                reserve(checked_tokens(reader[i]))
            receipts[key] = dict(validation=entry)
        replay_val = p.native_sdk().IndexedDataset(str(p.resolve_path(recipe['replay']['validation_prefix'])))
        for i in range(len(replay_val)):
            reserve(replay_document(replay_val, i))
        catalog.db.commit()
        status['phase'] = 'selecting_unique_complete_training_documents'
        status['prepared_sources'] = receipts
        for key in n.KEYS:
            status['current_source'] = key
            writer = p.NativeWriter(output/key/'train/corpus', 11, 131072)
            target = plan['train_tokens'][key]
            try:
                if key == 'climbmix_replay':
                    reader = p.native_sdk().IndexedDataset(str(p.resolve_path(recipe['replay']['train_prefix'])))
                    records = range(max(0, len(reader)-1))  # omit known upstream clipped tail
                else:
                    records = catalog.records(key)
                for record in records:
                    progress()
                    if writer.tokens >= target:
                        break
                    if key == 'climbmix_replay':
                        ids = replay_document(reader, record)
                        if len(ids) > config['max_document_tokens']:
                            catalog.counts['oversized_documents'] += 1
                            continue
                        text = tokenizer.decode(ids[:-1].tolist(), skip_special_tokens=False,
                                                clean_up_tokenization_spaces=False)
                        if tokenizer.encode(text, add_special_tokens=False, truncation=False) != ids[:-1].tolist():
                            raise ValueError('replay tokenizer roundtrip mismatch')
                        if catalog.matcher.matches(text):
                            catalog.counts['replay_benchmark_documents'] += 1
                            continue
                        if p.hashlib.sha256(text.encode()).hexdigest() in catalog.original_reserved_hashes:
                            catalog.counts['replay_prior_reserved_documents'] += 1
                            continue
                    else:
                        if record['split'] != 'train':
                            continue
                        ids = encode(record['text'])
                        if len(ids) > config['max_document_tokens']:
                            catalog.counts['oversized_documents'] += 1
                            continue
                    if not selected(ids):
                        continue
                    writer.append(ids)
                    status.update(active_train_tokens=writer.tokens, active_train_documents=writer.documents)
                if writer.tokens < target:
                    status['insufficient_capacity'] = dict(source=key, actual_unique_train_tokens=writer.tokens,
                                                           required_train_tokens=target, documents=writer.documents)
                    raise RuntimeError(f'{key} exhausted: {writer.tokens} < {target}; no cycling or substitution')
                receipts[key]['train'] = writer.publish()
                receipts[key]['train'].update(target_tokens=target, overshoot_tokens=writer.tokens-target,
                                               truncated_documents=0)
                catalog.db.commit()
            finally:
                writer.close()
        status['phase'] = 'final_native_readback_and_capacity_audit'
        audit = p.audit_outputs(catalog, receipts, 4096, progress)
        manifest = dict(schema_version=1, status='prepared_verified', sources=receipts,
            source_order=list(n.KEYS), weights=dict(zip(n.KEYS, n.WEIGHTS)), sequence_length=4096,
            source_provenance=dict(repository=recipe['repository'], revision=recipe['revision'],
                                   license=recipe['license'], raw_files=proof.get('raw_files', [])),
            replay_lineage={k:v for k,v in recipe['replay'].items() if k not in ('train_tokens', 'weight')},
            replay_already_consumed_by_base=True,
            tokenizer=recipe['tokenizer'], capacity_by_source=audit['capacity_by_source'], audit=audit,
            cycling=False, production_training_budget=plan, full_bounded_shards_scanned=True,
            heldout_reservations_sha256=p.file_digest(output/'heldout-reservations.jsonl'),
            companion_success_receipt=str((output/'success.json').resolve()),
            native_training_gate='pending_native_sample_index_and_training',
            seed_manifest_sha256=config['seed_manifest_sha256'], frozen_seed_heldout_reuse=True)
        manifest['capacity_gate'] = n.capacity_gate(manifest, plan['train_iters'], 128, 4096)
        reserved_ids = n.audit_reserved_ids(manifest)
        n.corpus_tokenizer_gate(manifest, reserved_ids)
        manifest['reserved_id_audit'] = reserved_ids
        p.atomic_json(output/'manifest.json', manifest)
        return dict(manifest_path=str((output/'manifest.json').resolve()),
                    manifest_sha256=p.file_digest(output/'manifest.json'), sources=receipts,
                    audit=audit, counts=dict(catalog.counts), full_bounded_shards_scanned=True,
                    cycling=False, production_training_budget=plan, frozen_seed_heldout_reuse=True,
                    grouped_identity_statistics=dict(components=len(components),
                        largest_component_documents=max(components.values(), default=0)))
    finally:
        catalog.close()


def execute(config, output, calibration_path, review, config_path=DEFAULT_CONFIG):
    """Explicit parent review, no acquisition, no GPU, immutable fresh generation."""
    import importlib.metadata
    import os
    import signal
    import time
    from datetime import datetime, timezone
    if review.get('approved') is not True:
        raise ValueError('explicit parent production review required')
    if Path(config_path).resolve() != DEFAULT_CONFIG.resolve():
        raise ValueError('execution requires canonical production_corpus.yaml')
    validate_config(config)
    output = Path(output).resolve()
    allowed = (ROOT/'datasets/midtraining').resolve()
    if output == allowed or not output.is_relative_to(allowed) or not str(output).startswith('/raid/'):
        raise ValueError('writes restricted to /raid repository datasets/midtraining')
    if output.exists():
        raise FileExistsError('fresh immutable production output required')
    output.mkdir(parents=True, exist_ok=False)
    p = producer()
    start = time.monotonic()
    status = dict(phase='preflight', pid=os.getpid(), execute=True, local_only=True,
        acquisition_bytes=0, gpu_used=False, checkpoint_weight_payloads_read=0,
        remote_tracking_run_created=False, started_at=datetime.now(timezone.utc).isoformat(),
        source_review=review, config=config, source_code_sha256=p.file_digest(Path(__file__)),
        image_identity_verified_by_this_process=False)
    last_persist = [0.]

    def persist():
        status.update(elapsed_seconds=time.monotonic()-start,
                      updated_at=datetime.now(timezone.utc).isoformat())
        p.atomic_json(output/'progress.json', status)

    def progress(force=False):
        elapsed = time.monotonic()-start
        if elapsed > config['max_hours']*3600:
            raise TimeoutError('bounded production preparation wall-time exhausted')
        if force or elapsed-last_persist[0] >= 5:
            size = sum(path.stat().st_size for path in output.rglob('*') if path.is_file())
            status['output_bytes_observed'] = size
            if size > config['max_output_bytes']:
                raise RuntimeError('bounded production storage cap exceeded')
            persist()
            last_persist[0] = elapsed

    def interrupted(signum, frame):
        raise TimeoutError(f'production preparation signal {signum}; no success')

    handlers = {sig:signal.signal(sig, interrupted) for sig in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT)}
    signal.setitimer(signal.ITIMER_REAL, config['max_hours']*3600)
    try:
        persist()
        progress(force=True)
        recipe, proof, paths, plan = verify_review(config, review, calibration_path, progress)
        status['plan'] = plan
        status['benchmark_manifests'] = proof['benchmark_manifests']
        status['benchmark_questions'] = len(proof['benchmark_questions'])
        status['original_climbmix_reserved_documents'] = len(proof['original_reserved_hashes'])
        status['prior_seed_reserved_documents'] = len(proof['seed_rows'])
        if importlib.metadata.version('transformers') != recipe['runtime']['transformers']:
            raise RuntimeError('requires pinned CPU image with Transformers 5.12.1; do not upgrade host')
        os.environ.update(CUDA_VISIBLE_DEVICES='', USE_TORCH='0', USE_TF='0', USE_FLAX='0',
            TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
            HF_HUB_DISABLE_IMPLICIT_TOKEN='1', HF_HUB_DISABLE_TELEMETRY='1', WANDB_MODE='disabled',
            HF_HOME=str(output/'cache/hf'), XDG_CACHE_HOME=str(output/'cache'),
            OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(p.resolve_path(recipe['tokenizer']['path'])),
                         local_files_only=True, trust_remote_code=False, use_fast=True)
        values = tokenizer.get_vocab().values()
        if tokenizer.eos_token_id != 11 or min(values) < 0 or max(values) >= 131072:
            raise ValueError('pinned tokenizer vocabulary/EOS mismatch')
        status['runtime_versions'] = {name:importlib.metadata.version(name)
                                     for name in ('transformers', 'numpy', 'pyarrow', 'pyyaml')}
        status['configured_runtime_image'] = recipe['runtime']['image']
        result = run_preparation(config, recipe, output, paths, proof, tokenizer, plan, progress, status)
        status.update(phase='prepared_verified', result=result)
        progress(force=True)
        persist()
        progress(force=True)
        p.atomic_json(output/'success.json', status)
        # Read back the exact published target and its companion before returning.
        native().load_prepared_manifest(output/'manifest.json')
        print(json.dumps(dict(status='prepared_verified', output=str(output),
                              manifest_sha256=result['manifest_sha256']), indent=2))
        return status
    except BaseException as error:
        status.update(phase='failed', error=repr(error))
        # A late receipt/companion failure is never advertised as success.
        if (output/'success.json').exists():
            (output/'success.json').unlink()
        persist()
        p.atomic_json(output/'failure.json', status)
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--preview', action='store_true')
    modes.add_argument('--execute', action='store_true')
    parser.add_argument('--reviewed-sources', type=Path)
    parser.add_argument('--calibration', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if args.execute:
        if args.reviewed_sources is None:
            parser.error('--execute requires --reviewed-sources')
        if args.calibration is None or args.output is None:
            parser.error('--execute requires --calibration and fresh --output')
        execute(load_config(args.config), args.output, args.calibration,
                json.loads(args.reviewed_sources.read_text()), args.config)
    else:
        config = load_config(args.config)
        result = dict(execute=False, local_only=True, acquisition_bytes=0, config=config,
                      calibration_required=True, training_executed=False)
        if args.calibration:
            result['plan'] = capacity_plan(json.loads(args.calibration.read_text()), config)
        print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
