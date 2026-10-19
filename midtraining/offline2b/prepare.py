#!/usr/bin/env python3
"""Independent local-only offline2B corpus capacity and preparation gates."""
from __future__ import annotations
import math
import hashlib
import importlib.util
from collections import Counter
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KEYS = ('code_concepts', 'algorithmic', 'formal_logic', 'climbmix_replay')
WEIGHTS = dict(zip(KEYS, (.67, .10, .03, .20)))


def budget():
    steps = math.ceil(2_000_000_000 / (4096 * 128))
    return dict(train_iters=steps, global_batch_size=128, sequence_length=4096,
                processed_tokens=steps*128*4096, cycling=False, weights=WEIGHTS,
                train_tokens={k: (math.ceil(math.ceil(steps*128*w)*1.005)+2)*4096+1
                              for k,w in WEIGHTS.items()})


@lru_cache(maxsize=None)
def helper(relative):
    spec = importlib.util.spec_from_file_location('offline2b_'+Path(relative).stem, ROOT/relative)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def census_source(db, source, reader, tokenizer, reserved_tokens, reserved_text, seen,
                  max_document_tokens, progress=lambda: None, *, diagnose_forbidden=False):
    """Measure entire eligible source once; authenticate sampled first-N native cursor.

    Caller authenticates frozen zero-skip provenance, native bytes/counts and full
    reservations. Prefix records are not tokenized again except seven samples.
    Every token document is range/EOS/reserved-ID/heldout/duplicate checked.
    """
    import numpy as np
    rec = helper('midtraining/recover_corpus.py')
    p = rec.producer()
    counts = Counter(tokens=0, documents=0, native_reused_documents=len(reader),
                     tail_tokenized_documents=0, duplicate_token_documents_skipped=0,
                     heldout_text_documents_skipped=0)
    sql = "SELECT id,text,content_hash FROM docs WHERE source=? AND split='train' ORDER BY id LIMIT 1 OFFSET ?"
    n = len(reader)
    rejected_ids = Counter()
    rejected_examples = []
    after = -1
    samples = []
    if n:
        row = db.execute(sql,(source,n-1)).fetchone()
        if row is None:
            raise ValueError('native prefix exceeds eligible catalog documents')
        after = row['id']
        for i in sorted(i for i in {0,1,n//4,n//2,3*n//4,n-2,n-1} if 0 <= i < n):
            row = db.execute(sql,(source,i)).fetchone()
            if hashlib.sha256(row['text'].encode()).hexdigest() != row['content_hash']:
                raise ValueError('sampled catalog text SHA mismatch')
            ids = tokenizer.encode(row['text'],add_special_tokens=False,truncation=False)+[11]
            if not np.array_equal(ids,reader[i]):
                raise ValueError('native first-N sampled cursor mismatch')
            samples.append(dict(native_document=i,catalog_id=row['id'],content_hash=row['content_hash'],
                                token_sha256=p.token_digest(ids)))
    for i in range(n):
        progress()
        ids = rec.checked_tokens(reader[i])
        digest = p.token_digest(ids)
        if digest in reserved_tokens or digest in seen:
            raise ValueError('reused native prefix duplicate or heldout violation')
        seen.add(digest)
        counts['tokens'] += len(ids)
        counts['documents'] += 1
    for row in db.execute("SELECT id,text,content_hash FROM docs WHERE source=? AND split='train' AND id>? ORDER BY id",(source,after)):
        progress()
        if hashlib.sha256(row['text'].encode()).hexdigest() != row['content_hash']:
            raise ValueError('tail catalog text SHA mismatch')
        counts['tail_tokenized_documents'] += 1
        if row['content_hash'] in reserved_text:
            counts['heldout_text_documents_skipped'] += 1
            continue
        encoded = tokenizer.encode(row['text'],add_special_tokens=False,truncation=False)+[11]
        forbidden = Counter(str(int(x)) for x in encoded if 18 <= x <= 29)
        if forbidden and diagnose_forbidden:
            counts['forbidden_id_documents_rejected'] += 1
            counts['forbidden_id_document_tokens_rejected'] += len(encoded)
            rejected_ids.update(forbidden)
            if len(rejected_examples)<10:
                rejected_examples.append(dict(catalog_id=row['id'],content_hash=row['content_hash'],
                                              token_sha256=p.token_digest(encoded),occurrences=dict(forbidden)))
            continue
        ids = rec.checked_tokens(encoded)
        if len(ids)>max_document_tokens:
            counts['oversized_documents_skipped'] += 1
            continue
        digest = p.token_digest(ids)
        if digest in reserved_tokens:
            counts['heldout_token_documents_skipped'] += 1
            continue
        if digest in seen:
            counts['duplicate_token_documents_skipped'] += 1
            continue
        seen.add(digest)
        counts['tokens'] += len(ids)
        counts['documents'] += 1
    return dict(counts, last_native_catalog_id=after, cursor_samples=samples,
                source_compatibility_gate=('blocked_forbidden_source_ids_require_explicit_review'
                                           if rejected_ids else 'source_ids_compatible'),
                rejected_reserved_id_occurrences=dict(rejected_ids),rejected_reserved_id_examples=rejected_examples,
                reserved_ids_18_through_29=0, full_eligible_source_exhausted=True,
                cycling=False, native_capacity_samples=max(0,(counts['tokens']-1)//4096))


def replay_candidate(ids, tokenizer):
    """Never change authenticated native IDs to make lossy decode pass.

    Byte-level native documents need not survive decode/encode. Exclude those
    documents and keep an explicit counter; no normalization or quota shortcut.
    """
    rec = helper('midtraining/recover_corpus.py')
    values = rec.checked_tokens(ids)[:-1].tolist()
    text = tokenizer.decode(values,skip_special_tokens=False,clean_up_tokenization_spaces=False)
    encoded = tokenizer.encode(text,add_special_tokens=False,truncation=False)
    if values != encoded:
        first = next((i for i,(a,b) in enumerate(zip(values,encoded)) if a != b),min(len(values),len(encoded)))
        return dict(accepted=False,reason='non_roundtrip_token_document_excluded',
                    first_mismatch_position=first, original_tokens=len(values), reencoded_tokens=len(encoded),
                    replacement_characters=text.count('\ufffd'),
                    native_token_sha256=rec.producer().token_digest(ids),
                    decoded_text_sha256=hashlib.sha256(text.encode()).hexdigest())
    return dict(accepted=True,text=text)


def capacity_report(sources):
    plan = budget()
    shortfalls = {k:plan['train_tokens'][k]-v['tokens'] for k,v in sources.items()
                  if v['tokens']<plan['train_tokens'][k]}
    compatibility = [k for k,v in sources.items() if v.get('source_compatibility_gate','').startswith('blocked_')]
    status = ('blocked_insufficient_unique_source_capacity' if shortfalls else
              'blocked_source_tokenizer_compatibility' if compatibility else 'capacity_measured_not_materialized')
    return dict(schema_version=1,status=status,source_compatibility_blockers=compatibility,
                measured_sources_complete=set(sources)==set(KEYS),
                budget=plan,sources=sources,shortfalls=shortfalls,
                training_release=False,manifest_published=False,cycling=False,
                acquisition_bytes=0,gpu_used=False,
                checkpoint_weights_integrity='not_checked_explicit_user_waiver')


def read_original_replay(original, progress=lambda: None):
    """Read the known clipped upstream source without claiming it was untruncated.

    Legacy read_native requires a zero *new writer* truncation counter for shape
    validation. That adapter input is not upstream provenance: restore and expose
    the authoritative source clipping flag explicitly in the measured receipt.
    The selector always omits the final document; no clipped tokens are accepted.
    """
    if original.get('last_document_truncated') is not True:
        raise ValueError('reviewed original Replay clipped-tail receipt required')
    rec = helper('midtraining/recover_corpus.py')
    reader, measured = rec.read_native(dict(original['train'],dtype='int32',truncated_documents=0),progress)
    measured.update(truncated_documents=1,upstream_last_document_truncated=True,
                    new_generation_truncated_documents=0)
    return reader, measured


def execute_census(output, review):
    """Authenticate inherited provenance, then measure all unique specialized tails.

    This is a real capacity run, not a success manifest producer. A short pool
    returns an explicit blocked receipt; it never promotes a failed whole run.
    """
    import json
    import os
    import signal
    import time
    import importlib.metadata
    from datetime import datetime,timezone
    if review.get('approved') is not True:
        raise ValueError('explicit independent source review required')
    output = Path(output).resolve()
    allowed = (ROOT/'results/midtraining/offline2b-corpus-01').resolve()
    if not output.is_relative_to(allowed) or output == allowed:
        raise ValueError('census writes restricted to fresh owned results child')
    if output.exists():
        raise FileExistsError('immutable fresh census output required')
    rec = helper('midtraining/recover_corpus.py')
    p = rec.producer()
    if set(review) != {'schema_version','approved','files','recovery_failure','recovery_snapshot'} or review['schema_version'] != 1:
        raise ValueError('exact independent source review schema required')
    required = set(rec.REVIEW_FILES)|{'midtraining/offline2b/prepare.py','midtraining/offline2b/corpus.yaml'}
    if set(review['files']) != required:
        raise ValueError('exact independent source review allowlist required')
    for name,sha in review['files'].items():
        rec.pinned(dict(path=name,sha256=sha),'independent source '+name)
    failure_path = rec.pinned(review['recovery_failure'],'historical recovery failure')
    failure = json.loads(failure_path.read_text())
    if failure.get('phase') != 'failed' or failure.get('error') != "ValueError('replay pinned tokenizer roundtrip mismatch')":
        raise ValueError('exact failed recovery generation required')
    if set(review['recovery_snapshot']) != set(rec.REVIEW_FILES):
        raise ValueError('all eight frozen recovery sources required')
    for name,pin in review['recovery_snapshot'].items():
        if pin['sha256'] != failure['source_review']['files'][name]:
            raise ValueError('frozen recovery source lineage mismatch')
        rec.pinned(pin,'frozen recovery '+name)
    if any(failure.get('counts',{}).get(k,0) != 0 for k in rec.SKIPS):
        raise ValueError('zero-skip recovery first-N precondition required')
    import yaml
    config = yaml.safe_load((ROOT/'midtraining/offline2b/corpus.yaml').read_text())
    if config['budget'] != budget() or config['cycling'] is not False:
        raise ValueError('offline2B budget is immutable')
    output.mkdir(parents=True,exist_ok=False)
    start = time.monotonic()
    state = dict(phase='authenticating_history',started_at=datetime.now(timezone.utc).isoformat(),
                 budget=budget(),source_review=review,acquisition_bytes=0,gpu_used=False,
                 checkpoint_weight_payloads_read=0,training_release=False,manifest_published=False)
    last = [0.]
    def progress(force=False):
        now = time.monotonic()
        if now-start > config['max_hours']*3600:
            raise TimeoutError('bounded offline2B census exhausted')
        if force or now-last[0] > 5:
            state['elapsed_seconds'] = now-start
            state['updated_at'] = datetime.now(timezone.utc).isoformat()
            p.atomic_json(output/'progress.json',state)
            last[0] = now
    def interrupted(signum,frame):
        raise TimeoutError('offline2B census interrupted '+str(signum))
    handlers = {sig:signal.signal(sig,interrupted) for sig in (signal.SIGALRM,signal.SIGTERM,signal.SIGINT)}
    signal.setitimer(signal.ITIMER_REAL,config['max_hours']*3600)
    try:
        os.environ.update(CUDA_VISIBLE_DEVICES='',USE_TORCH='0',USE_TF='0',USE_FLAX='0',
                          TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                          HF_HUB_DISABLE_IMPLICIT_TOKEN='1',HF_HUB_DISABLE_TELEMETRY='1',WANDB_MODE='disabled')
        progress(True)
        recipe,proof,_historical_plan = rec.verify_review(rec.load_config(),failure['source_review'],progress)
        if importlib.metadata.version('transformers') != '5.12.1':
            raise RuntimeError('pinned image Transformers 5.12.1 required')
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(p.resolve_path(recipe['tokenizer']['path'])),
                       local_files_only=True,trust_remote_code=False,use_fast=True)
        if tokenizer.eos_token_id != 11 or min(tokenizer.get_vocab().values()) < 0 or max(tokenizer.get_vocab().values()) >= 131072:
            raise ValueError('pinned tokenizer vocabulary/EOS mismatch')
        state.update(runtime_versions={n:importlib.metadata.version(n) for n in ('transformers','numpy','pyarrow','pyyaml')},
                     tokenizer=recipe['tokenizer'],inherited_full_scan_provenance=proof['inherited_full_scan_provenance'])
        reserved_text = set(proof['original_reserved_hashes'])
        rows = [json.loads(line) for line in Path(proof['full_reservations']).read_text().splitlines()]
        reserved_text.update(row['content_hash'] for row in rows)
        state['full_heldout_reservations'] = dict(documents=len(rows),sha256=p.file_digest(proof['full_reservations']),
                                                original_climbmix_documents=len(proof['original_reserved_hashes']))
        sources = {}
        seen = set()
        with rec.open_catalog(proof['catalog'],proof['catalog_sha256']) as db:
            reserved_tokens = {r[0] for r in db.execute('SELECT hash FROM reserved_tokens')}
            state['phase'] = 'authenticating_all_frozen_validation'
            validation = {}
            for key in KEYS:
                reader,entry = rec.read_native(failure['prepared_sources'][key]['validation'],progress)
                seed = proof['seed']['sources'][key]['validation']
                if any(entry[s+'_sha256'] != seed[s+'_sha256'] for s in ('bin','idx')):
                    raise ValueError('frozen seed validation mismatch')
                for i in range(len(reader)):
                    reserved_tokens.add(p.token_digest(rec.checked_tokens(reader[i])))
                validation[key] = entry
            replay_val = p.native_sdk().IndexedDataset(str(p.resolve_path(recipe['replay']['validation_prefix'])))
            for i in range(len(replay_val)):
                reserved_tokens.add(p.token_digest(rec.checked_tokens(replay_val[i])))
            state.update(frozen_validation=validation,reserved_token_document_hashes=len(reserved_tokens))
            native_parts = {}
            for key in KEYS[:3]:
                state.update(phase='measuring_entire_unique_specialized_source',current_source=key)
                progress(True)
                reader,entry = rec.read_native(failure['prepared_sources'][key]['train'],progress)
                result = census_source(db,key,reader,tokenizer,reserved_tokens,reserved_text,seen,
                                       config['max_document_tokens'],progress,diagnose_forbidden=True)
                result.update(required_tokens=budget()['train_tokens'][key],
                              deficit_tokens=max(0,budget()['train_tokens'][key]-result['tokens']))
                native_parts[key] = dict(entry=entry,origin_phase='failed',whole_manifest_accepted=False,
                                         byte_digests='fresh_current_measurement_not_historical_claim')
                sources[key] = result
                state['sources'] = sources
                report = capacity_report(sources)
                p.atomic_json(output/'capacity-specialized-progress.json',report)
                p.atomic_json(output.parent/'capacity.json',report)
                progress(True)
        state['native_reusable_parts'] = native_parts
        # Enough Replay is measured only to the fixed target; no whole-10B claim.
        state.update(phase='measuring_unique_replay_to_required_capacity',current_source=KEYS[-1])
        progress(True)
        reader,replay_bytes = read_original_replay(proof['original_replay_receipt'],progress)
        counts = Counter(tokens=0,documents=0,scanned_tokens=0,scanned_documents=0,
                         non_roundtrip_documents_skipped=0,upstream_clipped_tail_omitted_documents=1)
        matcher = helper('midtraining/production_corpus.py').ProductionBenchmarkMatcher(proof['benchmark_questions'])
        mismatches = []
        for i in range(len(reader)-1):
            progress()
            if counts['tokens'] >= budget()['train_tokens']['climbmix_replay']:
                break
            ids = rec.checked_tokens(reader[i])
            counts['scanned_tokens'] += len(ids)
            counts['scanned_documents'] += 1
            if counts['scanned_tokens'] > config['max_replay_scan_tokens']:
                raise RuntimeError('bounded replay scan exhausted; no cycling')
            if len(ids)>config['max_document_tokens']:
                counts['oversized_documents_skipped'] += 1
                continue
            candidate = replay_candidate(ids,tokenizer)
            if not candidate['accepted']:
                counts['non_roundtrip_documents_skipped'] += 1
                if len(mismatches)<10:
                    mismatches.append(dict(candidate,native_document=i))
                continue
            text = candidate['text']
            if matcher.matches(text) or hashlib.sha256(text.encode()).hexdigest() in reserved_text:
                counts['benchmark_or_heldout_text_documents_skipped'] += 1
                continue
            digest = p.token_digest(ids)
            if digest in reserved_tokens or digest in seen:
                counts['heldout_or_duplicate_token_documents_skipped'] += 1
                continue
            seen.add(digest)
            counts['tokens'] += len(ids)
            counts['documents'] += 1
        sources['climbmix_replay'] = dict(counts,required_tokens=budget()['train_tokens']['climbmix_replay'],
            native_source_measured_bytes=replay_bytes,roundtrip_mismatch_examples=mismatches,
            capacity_is_lower_bound=True,full_eligible_source_exhausted=False,reserved_ids_18_through_29=0,
            native_capacity_samples=max(0,(counts['tokens']-1)//4096))
        report = capacity_report(sources)
        report.update(full_heldout_reservations=state['full_heldout_reservations'],
                      frozen_validation=validation,native_reusable_parts=native_parts,
                      tokenizer=recipe['tokenizer'],source_review=review,
                      inherited_full_scan_provenance=proof['inherited_full_scan_provenance'],
                      full_catalog_sha256=proof['catalog_sha256'],
                      historical_recovery_failure=review['recovery_failure'],
                      historical_recovery_error=failure['error'],
                      checkpoint_weight_payloads_read=0,actual_unique_token_document_hashes=len(seen))
        for name,sha in review['files'].items():
            rec.pinned(dict(path=name,sha256=sha),'final independent source '+name)
        p.atomic_json(output/'capacity.json',report)
        p.atomic_json(output.parent/'capacity.json',report)
        state.update(phase='capacity_census_completed',result=report)
        progress(True)
        p.atomic_json(output/'census-completed.json',state)
        print(json.dumps(dict(status=report['status'],shortfalls=report['shortfalls'],output=str(output)),indent=2))
        return report
    except BaseException as error:
        state.update(phase='failed',error=repr(error))
        p.atomic_json(output/'failure.json',state)
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        for sig,handler in handlers.items():
            signal.signal(sig,handler)


def main(argv=None):
    import argparse
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--preview',action='store_true')
    modes.add_argument('--census',action='store_true')
    parser.add_argument('--reviewed-sources',type=Path)
    parser.add_argument('--output',type=Path)
    args = parser.parse_args(argv)
    if not args.census:
        print(json.dumps(dict(execute=False,budget=budget(),training_release=False,acquisition_bytes=0),indent=2))
        return 0
    if args.reviewed_sources is None or args.output is None:
        parser.error('--census requires --reviewed-sources and fresh --output')
    execute_census(args.output,json.loads(args.reviewed_sources.read_text()))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
