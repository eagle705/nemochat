#!/usr/bin/env python3
"""Local-only, fail-closed 67/10/3/20 unique-data recovery. Preview is inert."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from functools import lru_cache
import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path(__file__).with_name('recovery_mix67.yaml')
KEYS = ('code_concepts', 'algorithmic', 'formal_logic', 'climbmix_replay')
WEIGHTS = dict(zip(KEYS, (.67, .1, .03, .2)))
SKIPS = ('oversized_documents', 'reserved_token_documents_skipped', 'duplicate_token_documents_skipped')
HISTORY_FILES = (
    'midtraining/production_corpus.py', 'midtraining/production_corpus.yaml',
    'midtraining/prepare_corpus.py', 'midtraining/corpus.yaml', 'midtraining/longrun_native.py',
    'results/pretraining/nemochat_baby/climbmix/corpus_sdk.py',
)
REVIEW_FILES = HISTORY_FILES + ('midtraining/recover_corpus.py', 'midtraining/recovery_mix67.yaml')
ORIGIN_PRODUCER_SHA256 = '5bd5afc188402bd0908364b3c6d00731a48fa50bef5a27953d2c0813960adcb8'
SEED_SHA256 = '52d0d44fce30ea1737396fad9b2d95c3040835769eaa089df1d9bfa1f1d1e308'



@lru_cache(maxsize=None)
def helper(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT/relative)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def load_config(path=DEFAULT_CONFIG):
    import yaml
    return yaml.safe_load(Path(path).read_text())


def producer():
    return helper('recovery_indexed_producer', 'midtraining/prepare_corpus.py')


def native():
    return helper('recovery_native_consumer', 'midtraining/longrun_native.py')


def checked_tokens(ids):
    import numpy as np
    a = np.asarray(ids)
    if (a.ndim != 1 or not len(a) or a.dtype.kind not in 'iu'
            or a[-1] != 11 or a.min() < 0 or a.max() >= 131072):
        raise ValueError('native range/EOS/integer violation')
    if np.any((a >= 18) & (a <= 29)):
        raise ValueError('reserved IDs 18..29 forbidden')
    return a


@contextmanager
def open_catalog(path, sha256):
    """Never create, repair, journal, or change the failed production catalog."""
    path = Path(path).resolve()
    if producer().file_digest(path) != sha256:
        raise ValueError('origin catalog SHA mismatch')
    if any(Path(str(path)+s).exists() for s in ('-wal', '-journal')):
        raise ValueError('origin catalog has unreviewed transaction sidecars')
    db = sqlite3.connect(path.as_uri()+'?mode=ro&immutable=1', uri=True)
    db.row_factory = sqlite3.Row
    try:
        db.execute('PRAGMA query_only=ON')
        if [r[0] for r in db.execute('PRAGMA integrity_check')] != ['ok']:
            raise ValueError('origin catalog integrity check failed')
        required = dict(docs={'id','source','text','content_hash','group_id','split'},
                        reserved_tokens={'hash'}, selected_tokens={'hash'})
        for table, columns in required.items():
            actual = {r[1] for r in db.execute('PRAGMA table_info('+table+')')}
            if not columns <= actual:
                raise ValueError('origin catalog missing required table/columns: '+table)
        yield db
    finally:
        db.close()


def read_native(entry, progress=lambda: None):
    """Fresh actual reader and measured bytes; never invent an old authoritative SHA."""
    import numpy as np
    p = producer()
    reader = p.native_sdk().IndexedDataset(entry['prefix'])
    if (entry.get('reader_verified') is not True or entry.get('dtype') != 'int32'
            or entry.get('truncated_documents') != 0 or reader.index.dtype != np.int32
            or len(reader) != entry['documents']
            or int(reader.sequence_lengths.sum()) != entry['tokens']
            or len(reader.document_indices) != len(reader)+1
            or not np.array_equal(reader.document_indices, np.arange(len(reader)+1))
            or np.any(reader.sequence_lengths <= 0)):
        raise ValueError('old native prefix count/dtype/document shape mismatch')
    measured = dict(entry)
    for suffix in ('bin','idx'):
        progress()
        path = Path(entry['prefix']+'.'+suffix)
        if path.stat().st_size != entry[suffix+'_bytes']:
            raise ValueError('old native prefix byte count mismatch')
        digest = p.file_digest(path)
        if suffix+'_sha256' in entry and digest != entry[suffix+'_sha256']:
            raise ValueError('old native prefix SHA mismatch')
        measured[suffix+'_sha256'] = digest
    if measured['bin_bytes'] != measured['tokens']*4:
        raise ValueError('native int32 payload size mismatch')
    return reader, measured


def infer_cursor(db, entry, counts, tokenizer, expected_cursor=None, progress=lambda: None):
    """First N eligible TRAIN records, justified ONLY by reviewed zero-skip history."""
    import numpy as np
    if any(type(counts.get(k,0)) is not int or counts.get(k,0) != 0 for k in SKIPS):
        raise ValueError('cursor inference requires zero-skip historical counters')
    reader, measured = read_native(entry, progress)
    n = len(reader)
    if n <= 0:
        raise ValueError('empty old Code prefix')
    sql = "SELECT id,text,content_hash FROM docs WHERE source='code_concepts' AND split='train' ORDER BY id LIMIT 1 OFFSET ?"
    last = db.execute(sql,(n-1,)).fetchone()
    if last is None or (expected_cursor is not None and last['id'] != expected_cursor):
        raise ValueError('Code cursor mismatch')
    samples = []
    for i in sorted(i for i in {0,1,n//4,n//2,3*n//4,n-2,n-1} if 0 <= i < n):
        progress()
        row = db.execute(sql,(i,)).fetchone()
        if hashlib.sha256(row['text'].encode('utf-8')).hexdigest() != row['content_hash']:
            raise ValueError('cursor sampled text content SHA mismatch')
        ids = checked_tokens(tokenizer.encode(row['text'],add_special_tokens=False,truncation=False)+[11])
        if not np.array_equal(ids,checked_tokens(reader[i])):
            raise ValueError('cursor sample differs from native prefix')
        samples.append(dict(native_document=i,catalog_id=row['id'],content_hash=row['content_hash'],token_sha256=producer().token_digest(ids)))
    return dict(last_catalog_id=last['id'], documents=n, samples=samples, native=measured,
                zero_skip_precondition={k:counts.get(k,0) for k in SKIPS},
                inference='Nth eligible Code TRAIN record in catalog id order; no BPE on remaining prefix')


def validate_config(config):
    if set(config) != set(load_config()):
        raise ValueError('exact recovery configuration fields required')
    required = dict(schema_version=1,seed_manifest='datasets/midtraining/super35-kd-01/pilot-10m-v1/manifest.json',
        seed_manifest_sha256=SEED_SHA256,weights=WEIGHTS,train_iters=3200,global_batch_size=128,sequence_length=4096,
        train_tokens=dict(zip(KEYS,(1129705473,168620033,50593793,337231873))))
    if any(config.get(k) != v for k,v in required.items()):
        raise ValueError('approved 67/10/3/20 recovery budget or immutable seed changed')
    for key,maximum in dict(max_hours=12,max_output_bytes=64000000000,max_raw_bytes=7000000000,
                            max_replay_scan_tokens=2000000000,max_document_tokens=1000000).items():
        if type(config[key]) not in (int,float) or not 0 < config[key] <= maximum:
            raise ValueError('recovery bounds violation: '+key)


def pinned(entry, label):
    import re
    if not re.fullmatch('[0-9a-f]{64}',entry.get('sha256','')):
        raise ValueError(label+' malformed SHA256')
    path = producer().resolve_path(entry['path']).resolve()
    if producer().file_digest(path) != entry['sha256']:
        raise ValueError(label+' SHA256 mismatch')
    return path


def verify_history(review, progress):
    """History is authenticated from frozen sources, NEVER today's live native module."""
    origin = json.loads(pinned(review['origin_failure'],'origin failure').read_text())
    old = json.loads(pinned(review['origin_review'],'origin review').read_text())
    if (origin.get('phase') != 'failed' or origin.get('insufficient_capacity',{}).get('source') != 'formal_logic'
            or origin.get('source_review') != old or old.get('approved') is not True
            or origin.get('source_code_sha256') != ORIGIN_PRODUCER_SHA256
            or old.get('files',{}).get('midtraining/production_corpus.py') != ORIGIN_PRODUCER_SHA256):
        raise ValueError('origin must be the reviewed Formal quota failure, not a successful whole corpus')
    if any(type(origin['counts'].get(k,0)) is not int or origin['counts'].get(k,0) != 0 for k in SKIPS):
        raise ValueError('historical zero-skip precondition failed')
    if set(old['files']) != set(HISTORY_FILES) or set(review['origin_snapshot']) != set(HISTORY_FILES):
        raise ValueError('exact six frozen history snapshot files required')
    for name,entry in review['origin_snapshot'].items():
        progress()
        if entry['sha256'] != old['files'][name]:
            raise ValueError('history snapshot differs from origin source review: '+name)
        pinned(entry,'history snapshot '+name)
    receipt = json.loads(pinned(review['origin_parts'],'origin parts').read_text())
    if (receipt.get('status') != 'published_parts_readback_and_current_digests_recorded'
            or receipt.get('origin_failure_sha256') != review['origin_failure']['sha256']
            or receipt.get('origin_catalog_sha256') != review['origin_catalog']['sha256']
            or set(receipt.get('parts',{})) != set(KEYS[:2])):
        raise ValueError('origin parts receipt does not bind failed origin/catalog')
    for key in KEYS[:2]:
        previous = origin['prepared_sources'][key]['train']
        measured = receipt['parts'][key]
        if any(previous[k] != measured[k] for k in ('prefix','tokens','documents','bin_bytes','idx_bytes')):
            raise ValueError('origin native part receipt identity mismatch')
        # These are explicitly NEW current measurements, not an invented prior digest.
        previous.update(bin_sha256=measured['bin_sha256'],idx_sha256=measured['idx_sha256'])
    return origin,old,receipt


def verify_review(config, review, progress):
    """Exact source allowlist + full inherited scan + complete frozen exclusion lineage."""
    import re
    if review.get('approved') is not True:
        raise ValueError('explicit parent recovery review required')
    if set(review.get('files',{})) != set(REVIEW_FILES):
        raise ValueError('exact fresh parent source allowlist required')
    fields = {'schema_version','approved','files','origin_failure','origin_catalog','origin_review',
              'origin_parts','origin_snapshot','full_reservations','raw_verification','calibration'}
    if review.get('schema_version') != 1 or set(review) not in (fields,fields|{'expected_code_cursor'}):
        raise ValueError('exact version 1 recovery review schema required')
    for name,digest in review['files'].items():
        pinned(dict(path=name,sha256=digest),'active dependency '+name)
    validate_config(config)
    p,n = producer(),native()
    recipe = p.load_config()
    p.validate_config(recipe)
    origin,old,parts = verify_history(review,progress)
    failure_path = p.resolve_path(review['origin_failure']['path']).resolve()
    catalog = p.resolve_path(review['origin_catalog']['path']).resolve()
    if catalog != failure_path.with_name('catalog.sqlite'):
        raise ValueError('only the NEW failed production catalog is eligible; never the seed SQLite')
    full_reservations = pinned(review['full_reservations'],'full old+new reservations')
    if full_reservations != failure_path.with_name('heldout-reservations.jsonl'):
        raise ValueError('failed production full heldout reservation path required')
    # Keep benchmark, full original ClimbMix heldout, tokenizer assets and native
    # receipt checks from the reviewed producer. No model weight file is opened.
    active_review = dict(old,files={name:review['files'][name] for name in (
        'midtraining/prepare_corpus.py','midtraining/corpus.yaml',str(p.SDK_PATH.relative_to(ROOT)))})
    proof = p.verify_review(recipe,active_review)
    seed_path = p.resolve_path(config['seed_manifest'])
    seed,digest = n.load_prepared_manifest(seed_path)
    if digest != SEED_SHA256:
        raise ValueError('immutable whole seed manifest SHA mismatch')
    seed_reservations = seed_path.with_name('heldout-reservations.jsonl')
    if p.file_digest(seed_reservations) != seed['heldout_reservations_sha256']:
        raise ValueError('whole seed heldout manifest SHA mismatch')
    seed_rows = [json.loads(line) for line in seed_reservations.read_text().splitlines()]
    success = json.loads(seed_path.with_name('success.json').read_text())
    if len(seed_rows) != success['result']['heldout_reserved_documents']:
        raise ValueError('whole seed heldout reservation count mismatch')
    rows = [json.loads(line) for line in full_reservations.read_text().splitlines()]
    def identity(row):
        if row.get('source') not in KEYS or any(not re.fullmatch('[0-9a-f]{64}',row.get(k,'')) for k in ('content_hash','group_id')):
            raise ValueError('heldout reservation schema mismatch')
        return tuple(row[k] for k in ('source','content_hash','group_id'))
    reserved = {identity(row) for row in rows}
    if not {identity(row) for row in seed_rows} <= reserved:
        raise ValueError('full old heldout reservations were not preserved')
    with open_catalog(catalog,review['origin_catalog']['sha256']) as db:
        # The exact full scan catalog includes rejected raw rows in failure counts.
        counts = origin['counts']
        expected = counts['scanned_rows'] - sum(counts.get(k,0) for k in ('invalid_rows','schema_rejections','duplicate_rows'))
        if db.execute('SELECT COUNT(*) FROM docs').fetchone()[0] != expected:
            raise ValueError('inherited full scan catalog document count mismatch')
        if db.execute('SELECT COUNT(*) FROM selected_tokens').fetchone()[0] != sum(origin['prepared_sources'][k]['train']['documents'] for k in KEYS[:2]):
            raise ValueError('committed old selected set must contain exactly Code+Algo')
        if db.execute("SELECT 1 FROM docs WHERE source NOT IN ('code_concepts','algorithmic','formal_logic') OR split IS NULL OR split NOT IN ('train','validation','excluded') OR length(content_hash)!=64 OR length(group_id)!=64 LIMIT 1").fetchone():
            raise ValueError('inherited catalog identity/split schema invalid')
        for row in db.execute("SELECT source,content_hash,group_id FROM docs WHERE split='validation'"):
            if tuple(row) not in reserved:
                raise ValueError('additional catalog heldout missing from full reservation file')
    calibration = json.loads(pinned(review['calibration'],'historical calibration').read_text())
    if review['calibration'] != old['calibration']:
        raise ValueError('historical accepted calibration pin changed')
    if (calibration.get('status') != 'calibration_save_reload_accepted' or calibration.get('all_ranks') != 8
            or calibration.get('manifest_sha256') != SEED_SHA256):
        raise ValueError('accepted historical calibration required')
    plan = n.derive_horizon(calibration['calibration'])
    if plan['train_iters'] != config['train_iters']:
        raise ValueError('historical projected horizon differs from approved 3200 updates')
    plan.update(train_tokens=config['train_tokens'],weights=WEIGHTS,
        calibration_scope='historical legacy mix; NOT a new-mix benchmark')
    raw_report = json.loads(pinned(review['raw_verification'],'official raw verification').read_text())
    if (raw_report.get('status') != 'approved_raw_pool_verified' or raw_report.get('repository') != recipe['repository']
            or raw_report.get('revision') != recipe['revision']):
        raise ValueError('official raw verification identity mismatch')
    fields = ('key','path','local_path','size','sha256')
    raw = old['raw_files']
    normalize = lambda entries: sorted(tuple(e[k] for k in fields) for e in entries)
    if normalize(raw_report['raw_files']) != normalize(raw) or len(raw) != 13 or sum(e['size'] for e in raw) != 6032365862:
        raise ValueError('exact 13 reviewed official raw shards required')
    prod = helper('recovery_production_helpers','midtraining/production_corpus.py')
    paths = prod.verify_raw(config,old,progress)
    import pyarrow.parquet as pq
    total_rows = 0
    for key,path in paths:
        progress()
        parquet = pq.ParquetFile(path)
        if not {'text','metadata','license'} <= set(parquet.schema_arrow.names):
            raise ValueError('raw Parquet schema changed')
        total_rows += parquet.metadata.num_rows
    if total_rows != origin['counts']['scanned_rows'] or origin['counts']['files_scanned_complete'] != 13:
        raise ValueError('origin full raw scan coverage mismatch')
    proof.update(catalog=catalog,catalog_sha256=review['origin_catalog']['sha256'],origin=origin,
        active_source_files=dict(review['files']),
        full_reservations=full_reservations,seed=seed,raw_files=raw,
        expected_code_cursor=review.get('expected_code_cursor'),
        inherited_full_scan_provenance=dict(origin_phase='failed',old_whole_corpus_accepted=False,
            full_raw_scan_completed=True,recovery_did_rescan_raw=False,
            scanned_rows=total_rows,raw_files=raw,origin_failure=review['origin_failure'],
            origin_catalog=review['origin_catalog'],origin_review=review['origin_review'],
            origin_parts=review['origin_parts'],origin_snapshot=review['origin_snapshot'],
            raw_verification=review['raw_verification'],full_reservations=review['full_reservations'],
            seed_heldout_sha256=seed['heldout_reservations_sha256'],calibration=review['calibration']))
    return recipe,proof,plan


def run_preparation(config, recipe, output, proof, tokenizer, plan, progress, status):
    """Rebuild fresh native indices; tokenize only Code tail, Formal, and Replay checks."""
    from collections import Counter
    import shutil
    from types import SimpleNamespace
    p,n = producer(),native()
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    if (output/'selection.sqlite').exists():
        raise FileExistsError('fresh recovery selection required')
    counts = Counter()
    status['counts'] = counts
    receipts,reused = {},{}
    selected = sqlite3.connect(output/'selection.sqlite')
    selected.execute('CREATE TABLE selected_tokens(hash TEXT PRIMARY KEY)')
    selected.execute('CREATE TABLE reserved_tokens(hash TEXT PRIMARY KEY)')
    try:
        with open_catalog(proof['catalog'],proof['catalog_sha256']) as db:
            status['phase'] = 'verifying_cursor_and_frozen_validation'
            cursor = infer_cursor(db,proof['origin']['prepared_sources']['code_concepts']['train'],
                proof['origin']['counts'],tokenizer,proof.get('expected_code_cursor'),progress)
            cursor.update(catalog_sha256=proof['catalog_sha256'],recovery_source_sha256=p.file_digest(__file__),
                active_recovery_source_files=proof.get('active_source_files',{}),
                inherited_full_scan_provenance=proof['inherited_full_scan_provenance'])
            p.atomic_json(output/'code-cursor-proof.json',cursor)
            selected.executemany('INSERT INTO reserved_tokens VALUES(?)',
                ((row[0],) for row in db.execute('SELECT hash FROM reserved_tokens')))
            with Path(proof['full_reservations']).open('rb') as src, (output/'heldout-reservations.jsonl').open('xb') as dst:
                shutil.copyfileobj(src,dst)
            reserved_text = set(proof['original_reserved_hashes'])
            for line in Path(proof['full_reservations']).read_text().splitlines():
                reserved_text.add(json.loads(line)['content_hash'])
            matcher = helper('recovery_production_helpers','midtraining/production_corpus.py').ProductionBenchmarkMatcher(proof['benchmark_questions'])
            for key in KEYS:
                previous = proof['origin']['prepared_sources'][key]['validation']
                reader,entry = read_native(previous,progress)
                seed_entry = proof['seed']['sources'][key]['validation']
                prefix = output/key/'validation/corpus'
                prefix.parent.mkdir(parents=True,exist_ok=True)
                for suffix in ('bin','idx'):
                    if entry[suffix+'_sha256'] != seed_entry[suffix+'_sha256']:
                        raise ValueError('frozen whole seed validation changed')
                    with Path(previous['prefix']+'.'+suffix).open('rb') as src, Path(str(prefix)+'.'+suffix).open('xb') as dst:
                        shutil.copyfileobj(src,dst,1024*1024)
                    if p.file_digest(str(prefix)+'.'+suffix) != entry[suffix+'_sha256']:
                        raise ValueError('frozen validation copy SHA mismatch')
                entry.update(prefix=str(prefix.resolve()),frozen_seed_heldout_reuse=True)
                copied,_ = read_native(entry,progress)
                for i in range(len(copied)):
                    progress()
                    selected.execute('INSERT OR IGNORE INTO reserved_tokens VALUES(?)',(p.token_digest(checked_tokens(copied[i])),))
                receipts[key] = dict(validation=entry)
            # The catalog already reserves the ENTIRE original native Replay validation,
            # including its upstream clipped tail; never re-decode/trust that tail.
            selected.commit()

            def accept(ids, historical_exclude=False, required_old=False):
                digest = p.token_digest(checked_tokens(ids))
                if selected.execute('SELECT 1 FROM reserved_tokens WHERE hash=?',(digest,)).fetchone():
                    if required_old:
                        raise ValueError('reused prefix overlaps heldout')
                    counts['reserved_token_documents_skipped'] += 1
                    return False
                if selected.execute('SELECT 1 FROM selected_tokens WHERE hash=?',(digest,)).fetchone():
                    if required_old:
                        raise ValueError('duplicate reused training document')
                    counts['duplicate_token_documents_skipped'] += 1
                    return False
                old = db.execute('SELECT 1 FROM selected_tokens WHERE hash=?',(digest,)).fetchone()
                if required_old and not old:
                    raise ValueError('reused native document absent from old selected hashes')
                if historical_exclude and old:
                    counts['historical_selected_documents_skipped'] += 1
                    return False
                selected.execute('INSERT INTO selected_tokens VALUES(?)',(digest,))
                return True

            status['prepared_sources'] = receipts
            status['phase'] = 'copying_native_prefixes_and_selecting_unique_tail'
            replay_scanned = 0
            for key in KEYS:
                status['current_source'] = key
                writer = p.NativeWriter(output/key/'train/corpus',11,131072)
                target = plan['train_tokens'][key]
                try:
                    if key in KEYS[:2]:
                        reader,entry = read_native(proof['origin']['prepared_sources'][key]['train'],progress)
                        for i in range(len(reader)):
                            progress()
                            ids = checked_tokens(reader[i])
                            accept(ids,required_old=True)
                            writer.append(ids)
                        reused[key] = dict(origin_phase='failed',old_whole_corpus_accepted=False,
                            fresh_native_readback=True,entry=entry)
                    if key in ('code_concepts','formal_logic'):
                        sql = "SELECT id,text,content_hash FROM docs WHERE source=? AND split='train' AND id>? ORDER BY id"
                        after = cursor['last_catalog_id'] if key == 'code_concepts' else -1
                        for row in db.execute(sql,(key,after)):
                            progress()
                            if writer.tokens >= target:
                                break
                            if hashlib.sha256(row['text'].encode('utf-8')).hexdigest() != row['content_hash']:
                                raise ValueError('candidate raw text content SHA mismatch')
                            if row['content_hash'] in reserved_text:
                                raise ValueError('TRAIN catalog candidate belongs to full heldout')
                            ids = checked_tokens(tokenizer.encode(row['text'],add_special_tokens=False,truncation=False)+[11])
                            if len(ids) > config['max_document_tokens']:
                                counts['oversized_documents'] += 1
                                continue
                            # Formal may be the exact old incomplete subset: its own old
                            # selected hashes are not duplicates of new output documents.
                            if accept(ids,historical_exclude=key == 'code_concepts'):
                                writer.append(ids)
                            status.update(active_train_tokens=writer.tokens,active_train_documents=writer.documents)
                    elif key == 'climbmix_replay':
                        reader = p.native_sdk().IndexedDataset(str(p.resolve_path(recipe['replay']['train_prefix'])))
                        for i in range(max(0,len(reader)-1)):
                            progress()
                            if writer.tokens >= target:
                                break
                            ids = checked_tokens(reader[i])
                            replay_scanned += len(ids)
                            status['replay_scanned_tokens'] = replay_scanned
                            if replay_scanned > config['max_replay_scan_tokens']:
                                raise RuntimeError('bounded replay scan exhausted; no cycling')
                            if len(ids) > config['max_document_tokens']:
                                counts['oversized_documents'] += 1
                                continue
                            text = tokenizer.decode(ids[:-1].tolist(),skip_special_tokens=False,clean_up_tokenization_spaces=False)
                            if tokenizer.encode(text,add_special_tokens=False,truncation=False) != ids[:-1].tolist():
                                raise ValueError('replay pinned tokenizer roundtrip mismatch')
                            if matcher.matches(text) or hashlib.sha256(text.encode()).hexdigest() in reserved_text:
                                counts['replay_benchmark_or_heldout_documents_skipped'] += 1
                                continue
                            if accept(ids):
                                writer.append(ids)
                            status.update(active_train_tokens=writer.tokens,active_train_documents=writer.documents)
                    if writer.tokens < target:
                        status['insufficient_capacity'] = dict(source=key,actual_unique_train_tokens=writer.tokens,required_train_tokens=target)
                        raise RuntimeError(f'{key} exhausted: {writer.tokens} < {target}; no cycling or substitution')
                    receipts[key]['train'] = writer.publish()
                    receipts[key]['train'].update(target_tokens=target,overshoot_tokens=writer.tokens-target,truncated_documents=0)
                    selected.commit()
                finally:
                    writer.close()
            status['phase'] = 'fresh_all_eight_native_byte_and_capacity_audit'
            audit = p.audit_outputs(SimpleNamespace(db=selected),receipts,4096,progress)
            manifest = dict(schema_version=1,status='prepared_verified',source_order=list(KEYS),weights=WEIGHTS,
                sources=receipts,sequence_length=4096,cycling=False,tokenizer=recipe['tokenizer'],
                capacity_by_source=audit['capacity_by_source'],audit=audit,
                production_training_budget=plan,seed_manifest_sha256=load_config()['seed_manifest_sha256'],
                frozen_seed_heldout_reuse=True,heldout_reservations_sha256=p.file_digest(output/'heldout-reservations.jsonl'),
                companion_success_receipt=str((output/'success.json').resolve()),
                native_training_gate='pending_native_sample_index_and_training',
                inherited_full_scan_provenance=proof['inherited_full_scan_provenance'],recovery_did_rescan_raw=False,
                reused_parts_provenance=reused,code_cursor_proof_sha256=p.file_digest(output/'code-cursor-proof.json'),
                source_provenance=dict(repository=recipe['repository'],revision=recipe['revision'],license=recipe['license'],raw_files=proof.get('raw_files',[])),
                replay_lineage={k:v for k,v in recipe['replay'].items() if k not in ('train_tokens','weight')},
                replay_already_consumed_by_base=True,checkpoint_weights_integrity='not_checked_explicit_user_waiver')
            manifest['capacity_gate'] = n.capacity_gate(manifest,plan['train_iters'],128,4096)
            reserved_ids = n.audit_reserved_ids(manifest)
            n.corpus_tokenizer_gate(manifest,reserved_ids)
            manifest['reserved_id_audit'] = reserved_ids
            p.atomic_json(output/'manifest.json',manifest)
            return dict(manifest_path=str((output/'manifest.json').resolve()),manifest_sha256=p.file_digest(output/'manifest.json'),
                sources=receipts,audit=audit,counts=dict(counts),cycling=False,recovery_did_rescan_raw=False,
                inherited_full_scan_provenance=proof['inherited_full_scan_provenance'],production_training_budget=plan,
                frozen_seed_heldout_reuse=True)
    finally:
        selected.commit()
        selected.close()


def execute(config, output, review, config_path=DEFAULT_CONFIG):
    """CPU-only fresh generation. The parent owns the 64 GiB cgroup/image pin."""
    import importlib.metadata
    import os
    import signal
    import time
    from datetime import datetime,timezone
    if review.get('approved') is not True:
        raise ValueError('explicit parent recovery review required')
    if Path(config_path).resolve() != DEFAULT_CONFIG.resolve():
        raise ValueError('canonical recovery_mix67.yaml required')
    validate_config(config)
    output = Path(output).resolve()
    allowed = (ROOT/'datasets/midtraining').resolve()
    if output == allowed or not output.is_relative_to(allowed) or not str(output).startswith('/raid/'):
        raise ValueError('writes restricted to /raid repository datasets/midtraining')
    if output.exists():
        raise FileExistsError('fresh immutable recovery output required')
    output.mkdir(parents=True,exist_ok=False)
    p = producer()
    start = time.monotonic()
    status = dict(phase='preflight',execute=True,pid=os.getpid(),source_review=review,config=config,
        local_only=True,acquisition_bytes=0,gpu_used=False,checkpoint_weight_payloads_read=0,
        checkpoint_weights_integrity='not_checked_explicit_user_waiver',
        remote_tracking_run_created=False,image_identity_verified_by_this_process=False,
        parent_required_memory_limit_bytes=68719476736,started_at=datetime.now(timezone.utc).isoformat())
    last = [-10.]
    def persist():
        status.update(elapsed_seconds=time.monotonic()-start,updated_at=datetime.now(timezone.utc).isoformat())
        p.atomic_json(output/'progress.json',status)
    def progress(force=False):
        elapsed = time.monotonic()-start
        if elapsed > config['max_hours']*3600:
            raise TimeoutError('recovery wall-time exhausted')
        if force or elapsed-last[0] >= 5:
            size = sum(path.stat().st_size for path in output.rglob('*') if path.is_file())
            status['output_bytes_observed'] = size
            if size > config['max_output_bytes']:
                raise RuntimeError('recovery storage cap exceeded')
            persist()
            last[0] = elapsed
    def interrupted(signum,frame):
        raise TimeoutError(f'recovery signal {signum}; no success')
    handlers = {sig:signal.signal(sig,interrupted) for sig in (signal.SIGALRM,signal.SIGTERM,signal.SIGINT)}
    signal.setitimer(signal.ITIMER_REAL,config['max_hours']*3600)
    try:
        os.environ.update(CUDA_VISIBLE_DEVICES='',USE_TORCH='0',USE_TF='0',USE_FLAX='0',
            TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
            HF_HUB_DISABLE_IMPLICIT_TOKEN='1',HF_HUB_DISABLE_TELEMETRY='1',WANDB_MODE='disabled',
            HF_HOME=str(output/'cache/hf'),XDG_CACHE_HOME=str(output/'cache'),
            OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        persist()
        progress(force=True)
        recipe,proof,plan = verify_review(config,review,progress)
        if importlib.metadata.version('transformers') != recipe['runtime']['transformers']:
            raise RuntimeError('requires pinned production CPU image Transformers 5.12.1; do not upgrade host')
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(p.resolve_path(recipe['tokenizer']['path'])),
            local_files_only=True,trust_remote_code=False,use_fast=True)
        values = tokenizer.get_vocab().values()
        if tokenizer.eos_token_id != 11 or min(values) < 0 or max(values) >= 131072:
            raise ValueError('pinned tokenizer vocabulary/EOS mismatch')
        status.update(plan=plan,configured_runtime_image=recipe['runtime']['image'],
            runtime_versions={name:importlib.metadata.version(name) for name in ('transformers','numpy','pyarrow','pyyaml')})
        result = run_preparation(config,recipe,output,proof,tokenizer,plan,progress,status)
        # Concurrent source changes revoke acceptance, including deferred helper loads.
        for name,digest in review['files'].items():
            pinned(dict(path=name,sha256=digest),'active dependency '+name)
        status.update(phase='prepared_verified',result=result)
        progress(force=True)
        p.atomic_json(output/'success.json',status)
        native().load_prepared_manifest(output/'manifest.json')
        print(json.dumps(dict(status='prepared_verified',output=str(output),manifest_sha256=result['manifest_sha256']),indent=2))
        return status
    except BaseException as error:
        status.update(phase='failed',error=repr(error))
        if (output/'success.json').exists():
            (output/'success.json').unlink()
        persist()
        p.atomic_json(output/'failure.json',status)
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        for sig,handler in handlers.items():
            signal.signal(sig,handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--preview',action='store_true')
    modes.add_argument('--execute',action='store_true')
    parser.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    parser.add_argument('--reviewed-sources',type=Path)
    parser.add_argument('--output',type=Path)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    validate_config(config)
    if not args.execute:
        print(json.dumps(dict(execute=False,acquisition_bytes=0,gpu_used=False,config=config,
            required_source_files=list(REVIEW_FILES),required_history_snapshot_files=list(HISTORY_FILES),
            no_implicit_epochs=True,raw_rescan=False,parent_review_required=True),indent=2))
        return 0
    if args.reviewed_sources is None or args.output is None:
        parser.error('--execute requires --reviewed-sources and fresh --output')
    execute(config,args.output,json.loads(args.reviewed_sources.read_text()),args.config)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

