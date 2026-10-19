#!/usr/bin/env python3
"""Fresh native corpus for the explicitly approved 3662-call budget."""
from pathlib import Path
import math
ROOT = Path(__file__).resolve().parents[3]
KEYS = ('code_concepts','algorithmic','formal_logic','climbmix_replay')
WEIGHTS = dict(zip(KEYS,(.67,.10,.03,.20)))
CAPACITY_SHA = '467608152ca9288b561d5111d9f39d33cd26d546ea05d876024755e13f9593db'
OUTPUT = ROOT/'datasets/midtraining/super35-kd-01/offline1920-v1/data'
RESULTS = ROOT/'results/midtraining/offline1920-corpus-01'

def budget():
    return dict(train_iters=3662,global_batch_size=128,sequence_length=4096,
        processed_tokens=3662*128*4096,cycling=False,weights=WEIGHTS,
        train_tokens={k:(math.ceil(math.ceil(3662*128*w)*1.005)+2)*4096+1 for k,w in WEIGHTS.items()})

def capacity_check(capacity):
    margin = {k:capacity['sources'][k]['tokens']-t for k,t in budget()['train_tokens'].items()}
    if any(v<0 for v in margin.values()): raise ValueError('insufficient approved capacity')
    return margin


def helpers():
    import importlib.util
    from functools import lru_cache
    # The reviewed recovery module already caches each native helper.
    global _REC
    if '_REC' not in globals():
        s = importlib.util.spec_from_file_location('offline1920_recovery',ROOT/'midtraining/data/recover_corpus.py')
        _REC = importlib.util.module_from_spec(s); s.loader.exec_module(_REC)
    return _REC


def write_training(output, db, readers, cursors, replay_reader, tokenizer, targets,
                   reserved_tokens, reserved_text, benchmark_matches, excluded,
                   max_document_tokens, max_replay_scan_tokens, progress):
    """Copy native documents, BPE only specialized tails; no truncation/repetition."""
    import hashlib
    from collections import Counter
    rec = helpers(); p = rec.producer()
    prep = rec.helper('offline1920_census_helpers','midtraining/data/offline_kd/prepare.py')
    counts = Counter(); seen = set(); receipts = {}
    def accept(ids, required=False):
        digest = p.token_digest(rec.checked_tokens(ids))
        if digest in reserved_tokens or digest in seen:
            if required: raise ValueError('authenticated native prefix duplicate/heldout violation')
            counts['heldout_or_duplicate_documents_skipped'] += 1
            return False
        seen.add(digest)
        return True
    for key in KEYS:
        writer = p.NativeWriter(Path(output)/key/'train/corpus',11,131072)
        target = targets[key]
        try:
            if key in KEYS[:3]:
                progress(phase='copying_authenticated_native_prefix',current_source=key)
                for i in range(len(readers[key])):
                    ids = rec.checked_tokens(readers[key][i])
                    accept(ids,True); writer.append(ids)
                    if i % 4096 == 0: progress(active_train_tokens=writer.tokens,active_train_documents=writer.documents)
                progress(phase='tokenizing_only_unique_tail',current_source=key)
                for row in db.execute("SELECT id,text,content_hash FROM docs WHERE source=? AND split='train' AND id>? ORDER BY id",(key,cursors[key])):
                    if writer.tokens >= target: break
                    if hashlib.sha256(row['text'].encode()).hexdigest() != row['content_hash']:
                        raise ValueError('tail content SHA mismatch')
                    if row['id'] in excluded:
                        pin = excluded[row['id']]
                        # Authenticate each whole-document exclusion; never encode replacement IDs.
                        ids = tokenizer.encode(row['text'],add_special_tokens=False,truncation=False)+[11]
                        if row['content_hash'] != pin['content_hash'] or p.token_digest(ids) != pin['token_sha256']:
                            raise ValueError('authorized exclusion identity mismatch')
                        if not any(18<=x<=29 for x in ids): raise ValueError('exclusion no longer contains forbidden IDs')
                        counts['reserved_id_documents_excluded'] += 1
                        counts['reserved_id_document_tokens_excluded'] += len(ids)
                        continue
                    if row['content_hash'] in reserved_text:
                        counts['heldout_text_documents_skipped'] += 1; continue
                    ids = rec.checked_tokens(tokenizer.encode(row['text'],add_special_tokens=False,truncation=False)+[11])
                    counts['tail_tokenized_documents'] += 1
                    if len(ids)>max_document_tokens:
                        counts['oversized_documents_skipped'] += 1; continue
                    if accept(ids): writer.append(ids)
                    progress(active_train_tokens=writer.tokens,active_train_documents=writer.documents)
            else:
                progress(phase='selecting_whole_roundtrip_replay_documents',current_source=key)
                # Upstream's known clipped final document is never offered by caller.
                for i,raw in enumerate(replay_reader):
                    if writer.tokens >= target: break
                    ids = rec.checked_tokens(raw)
                    counts['replay_scanned_tokens'] += len(ids)
                    if counts['replay_scanned_tokens']>max_replay_scan_tokens: raise RuntimeError('replay scan bound exhausted')
                    if len(ids)>max_document_tokens:
                        counts['oversized_documents_skipped'] += 1; continue
                    candidate = prep.replay_candidate(ids,tokenizer)
                    if not candidate['accepted']:
                        counts['non_roundtrip_documents_excluded'] += 1; continue
                    text = candidate['text']
                    if benchmark_matches(text) or hashlib.sha256(text.encode()).hexdigest() in reserved_text:
                        counts['replay_benchmark_or_heldout_documents_skipped'] += 1; continue
                    if accept(ids): writer.append(ids)
                    progress(active_train_tokens=writer.tokens,active_train_documents=writer.documents,
                             replay_scanned_tokens=counts['replay_scanned_tokens'])
            if writer.tokens < target: raise RuntimeError(key+' unique source capacity exhausted')
            progress(phase='publishing_fresh_native_pair',current_source=key,active_train_tokens=writer.tokens)
            entry = writer.publish()
            entry.update(target_tokens=target,overshoot_tokens=writer.tokens-target,truncated_documents=0)
            receipts[key] = entry
        finally: writer.close()
    return receipts,dict(counts)


def fresh_train_audit(receipts,reserved_tokens,progress):
    """Fresh indices, counts, every complete document, bytes and global uniqueness."""
    rec = helpers(); p = rec.producer(); seen = set()
    result = dict(train_tokens=0,train_documents=0,train_heldout_token_document_overlap=0,
                  training_duplicate_token_documents=0,capacity_by_source={})
    for key,entry in receipts.items():
        progress(phase='fresh_full_native_train_audit',current_source=key)
        # read_native freshly checks complete index/document shape plus byte SHA.
        reader,measured = rec.read_native(entry)
        entry.update(measured)
        for i in range(len(reader)):
            ids = rec.checked_tokens(reader[i]); digest = p.token_digest(ids)
            if digest in reserved_tokens: raise ValueError('fresh audit heldout overlap')
            if digest in seen: raise ValueError('fresh audit duplicate document')
            seen.add(digest)
            if i%8192 == 0: progress(audited_documents=len(seen))
        result['train_tokens'] += entry['tokens']; result['train_documents'] += len(reader)
        result['capacity_by_source'][key] = (entry['tokens']-1)//4096
    return result


def validate_request(request):
    if request.get('approved') is not True: raise ValueError('explicit user approval required')
    if request.get('budget') != budget(): raise ValueError('immutable approved budget required')
    if request.get('capacity_sha256') != CAPACITY_SHA: raise ValueError('approved capacity SHA required')
    if request.get('whole_document_exclude_algorithmic_reserved_ids') is not True:
        raise ValueError('explicit whole-document exclusion approval required')
    if request.get('remap_ids') is not False: raise ValueError('no ID remapping permitted')


def execute(request):
    validate_request(request)
    import json, os, time, signal, shutil, sqlite3, hashlib, importlib.metadata
    from datetime import datetime,timezone
    import numpy as np
    rec = helpers(); p = rec.producer(); n = rec.native()
    import yaml
    config = yaml.safe_load((ROOT/'midtraining/configs/offline_kd/materialize.yaml').read_text())
    if config['budget'] != budget() or config['max_hours'] != 2 or config['max_output_bytes'] != 16000000000:
        raise ValueError('canonical bounded materializer config required')
    for name,sha in request['files'].items(): rec.pinned(dict(path=name,sha256=sha),'materializer frozen source')
    cap_path = ROOT/'results/midtraining/offline2b-corpus-01/capacity.json'
    if p.file_digest(cap_path) != CAPACITY_SHA: raise ValueError('capacity SHA mismatch')
    cap = json.loads(cap_path.read_text()); margins = capacity_check(cap)
    if OUTPUT.exists(): raise FileExistsError('fresh immutable materialization data required')
    OUTPUT.mkdir(parents=True,exist_ok=False)
    start = time.monotonic(); last = [0.]
    status = dict(phase='authenticating_frozen_history',approved_request=request,budget=budget(),
        capacity_sha256=CAPACITY_SHA,capacity_headroom_tokens=margins,pid=os.getpid(),
        started_at=datetime.now(timezone.utc).isoformat(),acquisition_bytes=0,gpu_used=False,
        checkpoint_weight_payloads_read=0,training_release=False)
    def progress(**fields):
        changed = fields.get('phase') != status.get('phase') if 'phase' in fields else False
        status.update(fields); now = time.monotonic()
        if now-start > config['max_hours']*3600: raise TimeoutError('materializer wall bound exhausted')
        if changed or now-last[0]>=5:
            size = sum(x.stat().st_size for x in OUTPUT.rglob('*') if x.is_file())
            if size>config['max_output_bytes']: raise RuntimeError('materializer disk bound exceeded')
            status.update(output_bytes_observed=size,elapsed_seconds=now-start,updated_at=datetime.now(timezone.utc).isoformat())
            p.atomic_json(OUTPUT/'progress.json',status); last[0] = now
    def interrupted(sig,frame): raise TimeoutError('materializer interrupted '+str(sig))
    handlers = {s:signal.signal(s,interrupted) for s in (signal.SIGTERM,signal.SIGINT,signal.SIGALRM)}
    signal.setitimer(signal.ITIMER_REAL,config['max_hours']*3600)
    try:
        progress()
        if importlib.metadata.version('transformers') != '5.12.1': raise RuntimeError('pinned CPU image Transformers required')
        review = cap['source_review']
        failure = json.loads(rec.pinned(review['recovery_failure'],'historical failed generation').read_text())
        if failure.get('phase') != 'failed' or failure.get('error') != "ValueError('replay pinned tokenizer roundtrip mismatch')":
            raise ValueError('historical failure must remain failed')
        if any(failure.get('counts',{}).get(k,0) != 0 for k in rec.SKIPS): raise ValueError('zero-skip historical cursor required')
        for pin in review['recovery_snapshot'].values(): rec.pinned(pin,'frozen historical source')
        lineage = cap['inherited_full_scan_provenance']
        for key in ('origin_failure','origin_review','origin_parts','raw_verification','calibration','full_reservations'):
            rec.pinned(lineage[key],'inherited '+key)
        for pin in lineage['origin_snapshot'].values(): rec.pinned(pin,'original selector snapshot')
        old = json.loads(rec.pinned(lineage['origin_review'],'original review').read_text())
        recipe = p.load_config()
        active = dict(old,files={name:request['files'][name] for name in ('midtraining/data/prepare_corpus.py','midtraining/configs/corpus.yaml',str(p.SDK_PATH.relative_to(ROOT)))})
        proof = p.verify_review(recipe,active)
        # Raw pool inherited complete-scan proof; verify bytes, never rescan/re-tokenize it.
        for raw in lineage['raw_files']:
            path = p.resolve_path(raw['local_path'])
            if path.stat().st_size != raw['size'] or p.file_digest(path) != raw['sha256']:
                raise ValueError('official raw byte pin mismatch')
        seed_pin = dict(path='datasets/midtraining/super35-kd-01/pilot-10m-v1/manifest.json',sha256=rec.SEED_SHA256)
        seed_path = rec.pinned(seed_pin,'immutable seed manifest')
        seed,seed_sha = n.load_prepared_manifest(seed_path)
        full_path = rec.pinned(lineage['full_reservations'],'complete heldout reservation')
        rows = [json.loads(x) for x in full_path.read_text().splitlines()]
        if len(rows) != 32397 or p.file_digest(full_path) != cap['full_heldout_reservations']['sha256']:
            raise ValueError('full heldout count/SHA mismatch')
        seed_rows = [json.loads(x) for x in seed_path.with_name('heldout-reservations.jsonl').read_text().splitlines()]
        if p.file_digest(seed_path.with_name('heldout-reservations.jsonl')) != seed['heldout_reservations_sha256']:
            raise ValueError('seed heldout SHA mismatch')
        identity = lambda r:tuple(r[k] for k in ('source','content_hash','group_id'))
        if not {identity(r) for r in seed_rows} <= {identity(r) for r in rows}: raise ValueError('missing original heldout reservation')
        shutil.copyfile(full_path,OUTPUT/'heldout-reservations.jsonl')
        reserved_text = set(proof['original_reserved_hashes'])|{r['content_hash'] for r in rows}
        if len(proof['original_reserved_hashes']) != 84990: raise ValueError('full original ClimbMix heldout reservation required')
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(p.resolve_path(recipe['tokenizer']['path'])),local_files_only=True,trust_remote_code=False,use_fast=True)
        if tok.eos_token_id != 11: raise ValueError('pinned EOS mismatch')
        validation = {}; reserved_tokens = set()
        progress(phase='authenticating_and_copying_all_frozen_validation')
        for key in KEYS:
            entry = cap['frozen_validation'][key]
            reader,measured = rec.read_native(entry)
            for suffix in ('bin','idx'):
                if measured[suffix+'_sha256'] != seed['sources'][key]['validation'][suffix+'_sha256']:
                    raise ValueError('frozen seed validation mismatch')
            prefix = OUTPUT/key/'validation/corpus'; prefix.parent.mkdir(parents=True,exist_ok=False)
            for suffix in ('bin','idx'): shutil.copyfile(entry['prefix']+'.'+suffix,str(prefix)+'.'+suffix)
            copied,measured = rec.read_native(dict(measured,prefix=str(prefix)))
            validation[key] = measured
            for i in range(len(copied)): reserved_tokens.add(p.token_digest(rec.checked_tokens(copied[i])))
        replay_validation = p.native_sdk().IndexedDataset(str(p.resolve_path(recipe['replay']['validation_prefix'])))
        # Upstream raw validation is reserved in its entirety, including clipped tail.
        for i in range(len(replay_validation)): reserved_tokens.add(p.token_digest(replay_validation[i]))
        catalog = p.resolve_path(lineage['origin_catalog']['path'])
        readers = {}; cursors = {}
        excluded_rows = cap['sources']['algorithmic']['rejected_reserved_id_examples']
        if len(excluded_rows) != 7: raise ValueError('exact seven authorized raw Algo exclusions required')
        excluded = {r['catalog_id']:r for r in excluded_rows}
        progress(phase='authenticating_catalog_and_native_prefix_cursors')
        with rec.open_catalog(catalog,cap['full_catalog_sha256']) as db:
            reserved_tokens.update(r[0] for r in db.execute('SELECT hash FROM reserved_tokens'))
            for key in KEYS[:3]:
                entry = cap['native_reusable_parts'][key]['entry']
                reader,measured = rec.read_native(entry); readers[key] = reader
                measured_cursor = db.execute("SELECT id FROM docs WHERE source=? AND split='train' ORDER BY id LIMIT 1 OFFSET ?",(key,len(reader)-1)).fetchone()[0]
                if measured_cursor != cap['sources'][key]['last_native_catalog_id']: raise ValueError('native cursor differs from census')
                cursors[key] = measured_cursor
                for sample in cap['sources'][key]['cursor_samples']:
                    row = db.execute('SELECT text,content_hash FROM docs WHERE id=?',(sample['catalog_id'],)).fetchone()
                    ids = tok.encode(row['text'],add_special_tokens=False,truncation=False)+[11]
                    if row['content_hash'] != sample['content_hash'] or hashlib.sha256(row['text'].encode()).hexdigest()!=sample['content_hash'] or p.token_digest(ids)!=sample['token_sha256'] or not np.array_equal(ids,reader[sample['native_document']]):
                        raise ValueError('frozen native cursor sampled content mismatch')
            # Authenticate all seven excluded records even if quota stops before one.
            for rid,pin in excluded.items():
                row = db.execute('SELECT source,text,content_hash FROM docs WHERE id=?',(rid,)).fetchone()
                ids = tok.encode(row['text'],add_special_tokens=False,truncation=False)+[11]
                if row['source']!='algorithmic' or hashlib.sha256(row['text'].encode()).hexdigest()!=pin['content_hash'] or p.token_digest(ids)!=pin['token_sha256']:
                    raise ValueError('authorized raw exclusion authentication failed')
            replay_entry = cap['sources']['climbmix_replay']['native_source_measured_bytes']
            replay,replay_measured = rec.read_native(replay_entry)
            if proof['original_replay_receipt'].get('last_document_truncated') is not True:
                raise ValueError('explicit upstream clipped tail provenance required')
            replay_iter = (replay[i] for i in range(len(replay)-1))
            matcher = rec.helper('offline1920_benchmark','midtraining/data/production_corpus.py').ProductionBenchmarkMatcher(proof['benchmark_questions'])
            training,counts = write_training(OUTPUT,db,readers,cursors,replay_iter,tok,budget()['train_tokens'],reserved_tokens,reserved_text,matcher.matches,excluded,config['max_document_tokens'],config['max_replay_scan_tokens'],progress)
        # Native full readback uses a new global set, not the selector's acceptance claims.
        audit = fresh_train_audit(training,reserved_tokens,progress)
        audit.update(validation_tokens=sum(v['tokens'] for v in validation.values()),validation_documents=sum(v['documents'] for v in validation.values()))
        sources = {k:dict(train=training[k],validation=validation[k]) for k in KEYS}
        manifest = dict(schema_version=1,status='prepared_verified',source_order=list(KEYS),weights=WEIGHTS,
            sources=sources,sequence_length=4096,cycling=False,tokenizer=recipe['tokenizer'],
            audit=audit,capacity_by_source=audit['capacity_by_source'],production_training_budget=budget(),
            approved_request_sha256=p.file_digest(RESULTS/'approved-request.json'),capacity_report_sha256=CAPACITY_SHA,
            seed_manifest_sha256=seed_sha,frozen_seed_heldout_reuse=True,
            heldout_reservations_sha256=p.file_digest(OUTPUT/'heldout-reservations.jsonl'),
            full_heldout_reservations=cap['full_heldout_reservations'],inherited_full_scan_provenance=lineage,
            reused_parts_provenance=cap['native_reusable_parts'],recovery_did_rescan_raw=False,
            authorized_raw_reserved_id_exclusions=excluded_rows,raw_source_exclusions_authorized=True,
            replay_upstream_last_document_truncated=True,replay_upstream_clipped_tail_always_excluded=True,
            replay_roundtrip_policy='exclude_whole_document_without_ID_rewrite',selection_counts=counts,
            source_provenance=dict(repository=recipe['repository'],revision=recipe['revision'],raw_files=lineage['raw_files']),
            companion_success_receipt=str(OUTPUT/'success.json'),native_training_gate='parent_owned',
            checkpoint_weights_integrity='not_checked_explicit_user_waiver')
        manifest['capacity_gate'] = n.capacity_gate(manifest,3662,128,4096)
        progress(phase='fresh_all_eight_full_byte_reserved_ID_audit')
        manifest['reserved_id_audit'] = n.audit_reserved_ids(manifest)
        n.corpus_tokenizer_gate(manifest,manifest['reserved_id_audit'])
        for name,sha in request['files'].items(): rec.pinned(dict(path=name,sha256=sha),'final frozen source')
        p.atomic_json(OUTPUT/'manifest.json',manifest)
        result = dict(status='prepared_verified',manifest_path=str(OUTPUT/'manifest.json'),manifest_sha256=p.file_digest(OUTPUT/'manifest.json'),sources=sources,audit=audit,counts=counts,cycling=False,full_bounded_shards_scanned=True)
        status.update(phase='prepared_verified',result=result,elapsed_seconds=time.monotonic()-start)
        p.atomic_json(OUTPUT/'progress.json',status); p.atomic_json(OUTPUT/'success.json',status)
        n.load_prepared_manifest(OUTPUT/'manifest.json')
        print(json.dumps(dict(status='prepared_verified',manifest_path=result['manifest_path'],manifest_sha256=result['manifest_sha256'])),flush=True)
        return result
    except BaseException as error:
        status.update(phase='failed',error=repr(error),elapsed_seconds=time.monotonic()-start)
        (OUTPUT/'success.json').unlink(missing_ok=True)
        p.atomic_json(OUTPUT/'failure.json',status); p.atomic_json(OUTPUT/'progress.json',status)
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        for s,h in handlers.items(): signal.signal(s,h)


def main():
    import argparse,json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--request',type=Path)
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps(dict(execute=False,budget=budget(),output=str(OUTPUT))))
        return
    if args.request is None: parser.error('--execute requires --request')
    execute(json.loads(args.request.read_text()))

if __name__ == '__main__': main()
