"""Pinned, bounded Nemotron acquisition and official-template SFT preparation.

No model/checkpoint bytes are read by this module. Prefix samples are biased;
RL seed scenarios are reserved, never converted to fabricated assistant answers.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

PINS = {
    'Nemotron-SFT-Agentic-v4': 'a4593a26ea83a700035bbe0ec1edc9ef3810669d',
    'Nemotron-SFT-Instruction-Following-Chat-v4': '1cd362945d7a00b136b95b96892bbfea8798686e',
    'Nemotron-SFT-Long-Context-Transduction-v1': '90fa9816dd6e9a6bdb123a73fc1a2ed6408783e0',
    'Nemotron-RL-Agentic-Conversational-Tool-Use-Seed-v1': '9715a833ae2de4863e321067211f25efe42f6ebe',
    'Nemotron-SFT-BBQ-v1': '4efdb045cda11918f29065a2e0fd5000b73b8c22',
}


def pinned_sources(discovery):
    sources = []
    for name, revision in PINS.items():
        card = json.loads((Path(discovery) / (name + '.json')).read_text())
        if card['id'] != 'nvidia/' + name or card['sha'] != revision:
            raise ValueError('pinned discovery identity mismatch: ' + name)
        license_value = card['cardData']['license']
        licenses = [license_value] if isinstance(license_value, str) else license_value
        for file in card['files']:
            if file['path'].endswith('.jsonl'):
                if not file['path'].startswith('data/') or '..' in Path(file['path']).parts:
                    raise ValueError('invalid pinned path')
                sources.append(dict(dataset=card['id'], revision=revision, path=file['path'],
                    expected_bytes=file['size'], license=licenses,
                    purpose='reserved_rl_seed' if 'Seed' in name else 'sft',
                    key=name + '--' + Path(file['path']).stem))
    return sources


def collect_jsonl(chunks, destination, byte_cap, row_cap, *, append=False):
    """Retain complete JSONL records only; cap includes partial/unretained bytes."""
    transferred = retained = rows = 0
    pending = b''
    digest = hashlib.sha256()
    if append:
        with Path(destination).open('rb') as prior:
            for line in prior:
                if not line.endswith(b'\n'):
                    raise ValueError('resume requires complete retained lines')
                json.loads(line)
                digest.update(line)
                retained += len(line)
                rows += 1
    with Path(destination).open('ab' if append else 'xb') as output:
        for chunk in chunks:
            transferred += len(chunk)
            if transferred > byte_cap:
                raise ValueError('transfer cap exceeded by transport')
            pending += chunk
            while b'\n' in pending and rows < row_cap:
                line, pending = pending.split(b'\n', 1)
                json.loads(line)
                record = line + b'\n'
                output.write(record)
                digest.update(record)
                retained += len(record)
                rows += 1
            if transferred >= byte_cap or rows >= row_cap:
                break
    return dict(rows=rows, retained_bytes=retained, transferred_bytes=transferred,
                prefix_sha256=digest.hexdigest(), source_byte_end=retained,
                sampling='bounded_prefix_not_representative', full_file_digest_verified=False)


def acquire_source(source, output, byte_cap, row_cap, *, base_url='https://huggingface.co', resume=False):
    """One bounded pinned prefix. Fresh paths only; HTTP retries cannot duplicate rows."""
    import time
    import urllib.request
    import urllib.error
    if not 0 < byte_cap <= 1024**3 or row_cap < 1:
        raise ValueError('positive row budget and at most 1 GiB per file required')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    dest = output / (source['key'] + '.jsonl')
    receipt_path = dest.with_suffix('.receipt.json')
    if receipt_path.exists() or (dest.exists() and not resume):
        raise FileExistsError('fresh acquisition only; existing receipt must be explicitly reused')
    offset = dest.stat().st_size if resume and dest.exists() else 0
    if offset >= source['expected_bytes']:
        raise ValueError('source already fully retained; explicit readback/reuse required')
    cap = min(byte_cap, source['expected_bytes'])
    url = f"{base_url}/datasets/{source['dataset']}/resolve/{source['revision']}/{source['path']}"
    request = urllib.request.Request(url, headers={'Range': f"bytes={offset}-{min(source['expected_bytes']-1,offset+cap-1)}", 'Accept-Encoding': 'identity'})
    response = None
    # Retry only the request before bytes are consumed; interrupted prefixes remain failures.
    for attempt in range(3):
        try:
            response = urllib.request.urlopen(request, timeout=90)
            break
        except (urllib.error.URLError, TimeoutError):
            if attempt == 2:
                raise
            time.sleep(2**attempt)
    with response:
        status = response.status
        if status == 206:
            content_range = response.headers.get('Content-Range', '')
            import re
            match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', content_range)
            if not match or int(match[1]) != offset or int(match[2]) >= offset+cap or int(match[3]) != source['expected_bytes']:
                raise ValueError('pinned HTTP range/size mismatch')
        elif status != 200:
            raise ValueError('HTTP 200/206 required')
        remaining = [cap]
        skipped = 0
        # A server ignoring Range is safely replayed from byte zero, never appended
        # blindly. Replayed transport bytes consume the SAME transfer budget.
        if status == 200 and offset:
            if offset >= cap:
                raise ValueError('200 fallback replay would exhaust transfer budget')
            while skipped < offset:
                chunk = response.read(min(65536,offset-skipped,remaining[0]))
                if not chunk:
                    raise ValueError('short 200 fallback before retained prefix')
                skipped += len(chunk)
                remaining[0] -= len(chunk)
        def chunks():
            while remaining[0]:
                chunk = response.read(min(65536, remaining[0]))
                if not chunk:
                    expected_read = min(cap, source['expected_bytes'] - (offset if status == 206 else 0))
                    if cap - remaining[0] < expected_read:
                        raise OSError('short pinned stream; resume retained complete lines explicitly')
                    break
                remaining[0] -= len(chunk)
                yield chunk
        try:
            result = collect_jsonl(chunks(), dest, cap-skipped, row_cap, append=bool(offset))
        except BaseException as error:
            with (output/'failed-transfers.jsonl').open('a') as ledger:
                ledger.write(json.dumps(dict(key=source['key'],transferred_bytes=cap-remaining[0],
                    retained_bytes=dest.stat().st_size if dest.exists() else 0,error=type(error).__name__))+'\n')
            raise
        result['transferred_bytes'] += skipped
    receipt = dict(source, **result, http_status=status, byte_cap=cap, row_cap=row_cap,
                   resume_policy='exact_206_offset_or_budgeted_200_replay_no_duplicate_append',
                   resumed_source_offset=offset, raw_file=dest.name)
    receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
    return json.loads(receipt_path.read_text())


class SourceSchemaRejection(ValueError):
    """An explicit unsupported source contract, never an instrumentation failure.

    Positional constructor arguments intentionally round-trip through process pools.
    Details contain indexes/types only, not source conversation contents.
    """
    def __init__(self, reason_code, details):
        self.reason_code = reason_code
        self.details = details
        super().__init__(reason_code, details)


def normalize_record(record, dataset, *, decode_tool_arguments=False):
    """Preserve tool identity/order; IFC masks index ALL messages, not assistant turns."""
    import copy
    if 'Seed' in dataset:
        raise ValueError('RL seed is not an SFT target dataset')
    messages = copy.deepcopy(record['messages'])
    if not messages:
        raise ValueError('empty conversation')
    selected = record.get('metadata', {}).get('train_turns_mask')
    if 'Instruction-Following' in dataset and selected is None:
        raise ValueError('IFC requires message-aligned train_turns_mask')
    if selected is None:
        selected = [m['role'] == 'assistant' for m in messages]
    if len(selected) != len(messages) or any(type(v) is not bool for v in selected):
        raise ValueError('boolean message-aligned train_turns_mask required')
    calls = set()
    decoded_arguments = []
    for i, message in enumerate(messages):
        role = message['role']
        if role not in ('system', 'user', 'assistant', 'tool'):
            raise ValueError('unsupported message role')
        if selected[i] and role != 'assistant':
            raise ValueError('only assistant turns can be supervised')
        message.pop('reasoning_content', None)
        if message.get('content') is None:
            message['content'] = ''
        if not isinstance(message['content'], str):
            raise ValueError('text-only corpus expected')
        for call_index, call in enumerate(message.get('tool_calls') or []):
            function = call.get('function', call)
            arguments = function.get('arguments', {})
            details = dict(message_index=i, call_index=call_index,
                           argument_source_type=type(arguments).__name__)
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError as error:
                    raise SourceSchemaRejection('tool_arguments_invalid_json',
                        dict(details, argument_decoded_type='str')) from error
            if not isinstance(arguments, dict):
                raise SourceSchemaRejection('tool_arguments_not_object',
                    dict(details, argument_decoded_type=type(arguments).__name__))
            if 'arguments' in function:
                decoded_arguments.append((function, arguments))
            call_id = call['id']
            if call_id in calls:
                raise ValueError('duplicate tool call identity')
            calls.add(call_id)
        if role == 'tool' and message.get('tool_call_id') not in calls:
            raise SourceSchemaRejection('orphan_tool_response', dict(message_index=i))
    if not any(selected):
        raise ValueError('no selected assistant target')
    canonical = json.dumps({'messages': messages, 'tools': record.get('tools', [])},
                           ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    if decode_tool_arguments:
        for function, arguments in decoded_arguments:
            function['arguments'] = arguments
    first_user = next(m['content'] for m in messages if m['role'] == 'user')
    return dict(messages=messages, selected=list(selected), tools=copy.deepcopy(record.get('tools', [])),
                content_hash=hashlib.sha256(canonical.encode()).hexdigest(),
                prompt_group=hashlib.sha256(first_user.encode()).hexdigest(),
                source_id=record.get('uuid', record.get('metadata', {}).get('uuid')),
                metadata=copy.deepcopy(record.get('metadata', {})),
                parallel_tool_calls=record.get('parallel_tool_calls', False))


def instrument_template(template):
    """Capture official assistant rendering; annotate targets without changing bytes.

    This is deliberately specific to the pinned Super3.5 outer message branches.
    Headers remain context; content, function calls and official EOT remain targets.
    Every example must independently prove identical uninstrumented rendering/IDs.
    """
    start = '{%- if message.role == "assistant" %}'
    end = '{%- elif message.role == "user" or message.role == "system" %}'
    if template.count(start) != 1 or template.count(end) != 1:
        raise ValueError('unsupported official template structure')
    header = '<|im_start|>assistant\n'
    capture = start + '{%- set _nemochat_assistant -%}'
    release = '''{%- endset -%}
{%- if not _nemochat_assistant.startswith('<|im_start|>assistant\\n') -%}
{{- raise_exception('assistant header mismatch') -}}
{%- endif -%}
{{- _nemochat_assistant[:HEADER_LEN] -}}
{%- if message['_nemochat_train'] -%}
{% generation %}{{- _nemochat_assistant[HEADER_LEN:] -}}{% endgeneration %}
{%- else -%}
{{- _nemochat_assistant[HEADER_LEN:] -}}
{%- endif -%}
'''.replace('HEADER_LEN', str(len(header)))
    return template.replace(start, capture).replace(end, release + end)


def encode_record(record, dataset, tokenizer, sequence_length=4096):
    normalized = normalize_record(record, dataset, decode_tool_arguments=True)
    messages = normalized['messages']
    for message, selected in zip(messages, normalized['selected']):
        message['_nemochat_train'] = selected
    kwargs = dict(tools=normalized['tools'], enable_thinking=False, add_generation_prompt=False)
    official = tokenizer.apply_chat_template(messages, tokenize=False, **kwargs)
    template = instrument_template(tokenizer.chat_template)
    full = tokenizer(official, add_special_tokens=False)['input_ids']
    # Full prompt plus complete target must fit the actual context, not card target_tokens.
    if len(full) > sequence_length:
        return None
    annotated = tokenizer.apply_chat_template(messages, chat_template=template, tokenize=False, **kwargs)
    if annotated != official:
        raise ValueError('instrumentation changed official rendered bytes')
    encoded = tokenizer.apply_chat_template(messages, chat_template=template, tokenize=True,
        return_dict=True, return_assistant_tokens_mask=True, **kwargs)
    if list(encoded['input_ids']) != full:
        raise ValueError('instrumentation changed official token IDs')
    masks = list(encoded['assistant_masks'])
    if len(masks) != len(full) or not sum(masks):
        raise ValueError('missing assistant target spans')
    eos = tokenizer.eos_token_id
    if eos is None:
        raise ValueError('tokenizer EOS required for padding only')
    tokens = full + [eos] * (sequence_length + 1 - len(full))
    masks += [0] * (sequence_length + 1 - len(masks))
    return dict(normalized, tokens=tokens, loss_mask=masks[1:], nonpadding_tokens=len(full),
                supervised_tokens=sum(masks[1:]), official_render_sha256=hashlib.sha256(official.encode()).hexdigest())


def normalize_question(text):
    import re
    import unicodedata
    return re.sub(r'\s+', '', unicodedata.normalize('NFC', text)).casefold()


def prepare_subset(raw_directory, output, tokenizer, benchmark_questions, *, quota_per_file=16,
                   sequence_length=4096, heldout_modulus=100, quotas=None, reserved_prompt_groups=None,
                   excluded_datasets=None, reserved_prompt_group_datasets=None):
    """Write one-pass, full-example arrays and immutable source/offset provenance.

    Hash reservation is stable across expanded prefixes. Duplicate UUIDs, full
    content and identical initial user prompts never cross split boundaries.
    Missing requested SFT datasets block training rather than silently reweight.
    """
    from collections import Counter
    import re
    import numpy as np
    excluded_datasets = sorted(set(excluded_datasets or []))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    counts = Counter()
    per_source = {}
    seen_content, seen_ids = set(), set()
    split_groups = {'train': set(), 'heldout': set()}
    target_counts, nonpadding = Counter(), Counter()
    handles = {s: ((output/(s+'-tokens.bin')).open('xb'), (output/(s+'-masks.bin')).open('xb'),
                   (output/(s+'-dialogues.jsonl')).open('x')) for s in split_groups}
    receipts = []
    quarantine = (output/'source-schema-rejections.jsonl').open('x')
    rejection_counts = Counter()

    def reject(error, source, stats, record, line_no, start_offset):
        stats['schema_rejected'] += 1
        stats['schema_rejected:' + error.reason_code] += 1
        rejection_counts[error.reason_code] += 1
        quarantine.write(json.dumps(dict(source_dataset=source['dataset'],
            source_revision=source['revision'], source_key=source['key'], source_file=source['path'],
            source_line=line_no, source_byte_offset=start_offset,
            source_id=record.get('uuid', record.get('metadata', {}).get('uuid')),
            reason_code=error.reason_code, details=error.details), ensure_ascii=False) + '\n')
        quarantine.flush()

    try:
        for receipt_path in sorted(Path(raw_directory).glob('*.receipt.json')):
            source = json.loads(receipt_path.read_text())
            name = source['dataset'].removeprefix('nvidia/')
            if PINS.get(name) != source['revision']:
                raise ValueError('raw receipt revision not in explicit allowlist')
            raw_path = receipt_path.with_name(source['raw_file'])
            if raw_path.parent != receipt_path.parent or Path(source['raw_file']).name != source['raw_file']:
                raise ValueError('raw source path traversal')
            receipts.append(source)
            stats = Counter()
            per_source[source['key']] = stats
            if source['purpose'] == 'reserved_rl_seed':
                stats['reserved_rl_rows'] = source['rows']
                continue
            if source['dataset'] in excluded_datasets:
                stats.update(excluded=True, exclusion_reason='deferred_long_context_stage',
                             train=0, heldout=0, accepted=0, excluded_rows=source['rows'])
                continue
            quota = (quotas or {}).get(source['key'], quota_per_file)
            offset = 0
            with raw_path.open('rb') as raw:
                for line_no, line in enumerate(raw, 1):
                    start_offset = offset
                    offset += len(line)
                    stats['visited'] += 1
                    record = json.loads(line)
                    try:
                        normalized = normalize_record(record, source['dataset'])
                    except SourceSchemaRejection as error:
                        reject(error, source, stats, record, line_no, start_offset)
                        continue
                    identity = (source['dataset'], normalized['source_id']) if normalized['source_id'] else None
                    if normalized['content_hash'] in seen_content or (identity and identity in seen_ids):
                        stats['duplicate_dropped'] += 1
                        continue
                    contaminated = False
                    for message in normalized['messages']:
                        if message['role'] == 'user':
                            text = message['content']
                            candidates = (normalize_question(text), normalize_question(re.split('[?？]', text, 1)[0]+'?'))
                            if any(q in benchmark_questions for q in candidates):
                                contaminated = True
                    if contaminated:
                        stats['kmmlu_dropped'] += 1
                        continue
                    try:
                        encoded = encode_record(record, source['dataset'], tokenizer, sequence_length)
                    except SourceSchemaRejection as error:
                        reject(error, source, stats, record, line_no, start_offset)
                        continue
                    if encoded is None:
                        stats['oversized_complete_example_dropped'] += 1
                        continue
                    group = encoded['prompt_group']
                    # Group, not variant ID, owns deterministic heldout assignment.
                    split = 'heldout' if group in (reserved_prompt_groups or set()) or int(group[:16], 16) % heldout_modulus == 0 else 'train'
                    if group in split_groups['heldout' if split == 'train' else 'train']:
                        raise ValueError('split group leakage')
                    seen_content.add(encoded['content_hash'])
                    if identity:
                        seen_ids.add(identity)
                    split_groups[split].add(group)
                    tokens, masks, dialogues = handles[split]
                    np.asarray(encoded['tokens'], dtype=np.uint32).tofile(tokens)
                    np.asarray(encoded['loss_mask'], dtype=np.uint8).tofile(masks)
                    provenance = {k:v for k,v in encoded.items() if k not in ('tokens','loss_mask')}
                    provenance.update(source_dataset=source['dataset'], source_revision=source['revision'],
                        source_file=source['path'], source_line=line_no, source_byte_offset=start_offset,
                        source_key=source['key'], split=split)
                    dialogues.write(json.dumps(provenance, ensure_ascii=False)+'\n')
                    counts[split] += 1
                    stats[split] += 1
                    stats['accepted'] += 1
                    stats['assistant_tool_calls'] += sum(len(m.get('tool_calls') or []) for m in encoded['messages'])
                    stats['min_tokens'] = min(stats.get('min_tokens', encoded['nonpadding_tokens']), encoded['nonpadding_tokens'])
                    stats['max_tokens'] = max(stats['max_tokens'], encoded['nonpadding_tokens'])
                    stats['supervised_tokens'] += encoded['supervised_tokens']
                    target_counts[split] += encoded['supervised_tokens']
                    nonpadding[split] += encoded['nonpadding_tokens']
                    if stats['accepted'] >= quota:
                        break
            print(json.dumps({'source':source['key'], **stats}), flush=True)
    finally:
        quarantine.close()
        for group_handles in handles.values():
            for handle in group_handles:
                handle.close()
    for split in split_groups:
        for suffix, dtype, width in [('tokens',np.uint32,sequence_length+1),('masks',np.uint8,sequence_length)]:
            path = output/(split+'-'+suffix+'.bin')
            array = np.memmap(path, dtype=dtype, mode='r', shape=(counts[split],width)) if counts[split] else np.empty((0,width),dtype=dtype)
            np.save(output/(split+'-'+suffix+'.npy'), array, allow_pickle=False)
            del array
            path.unlink()
    expected = {'nvidia/'+n for n in PINS if 'Seed' not in n} - set(excluded_datasets)
    present = {s['dataset'] for s in receipts if per_source[s['key']]['accepted']}
    missing = sorted(expected - present)
    files = {}
    for path in output.iterdir():
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for chunk in iter(lambda: handle.read(4*1024**2), b''):
                digest.update(chunk)
        files[path.name] = dict(bytes=path.stat().st_size, sha256=digest.hexdigest())
    manifest = dict(schema='nemochat-nemotron-sft-v1', format='official-super35-selected-assistant-v1',
        status='blocked_missing_sft_sources' if missing else 'passed', missing_sft_sources=missing,
        sequence_length=sequence_length, train_dialogues=counts['train'], heldout_dialogues=counts['heldout'],
        train_loss_tokens=target_counts['train'], heldout_loss_tokens=target_counts['heldout'],
        nonpadding_tokens=dict(nonpadding), per_source={k:dict(v) for k,v in per_source.items()},
        sources=receipts, train_heldout_overlap=0, kmmlu_exact_question_matches=0,
        kmmlu_reserved_questions=len(benchmark_questions), files=files,
        source_schema_policy=dict(policy='quarantine whole conversation; no repair; only typed source contract errors; instrumentation/mask/token failures fatal',
            rejection_file='source-schema-rejections.jsonl', rejected_records=sum(rejection_counts.values()),
            reason_counts=dict(rejection_counts)),
        reasoning_policy='omit_reasoning_content_nonthinking_official_empty_think_markers_preserved',
        mask_policy='message-aligned selected assistants; full-render generation offsets; headers/tools/users masked',
        long_context_policy='complete prompt plus complete reference <= actual sequence length; no truncation',
        split_policy=f'sha256 first user prompt modulo {heldout_modulus}; UUID/full-content dedup; prefix sample only',
        training_repetition=False, quota_per_file=quota_per_file, requested_quotas=quotas,
        prior_reserved_prompt_groups=len(reserved_prompt_groups or set()),
        excluded_datasets=excluded_datasets,
        source_selection_policy=dict(excluded_datasets=excluded_datasets,
            exclusion_reason='deferred_long_context_stage', reserved_rl_seed='unlabeled_reserved'),
        active_training_source_keys=[s['key'] for s in receipts
            if s['purpose']=='sft' and s['dataset'] not in excluded_datasets],
        active_training_source_bytes=sum(s.get('retained_bytes',0) for s in receipts
            if s['purpose']=='sft' and s['dataset'] not in excluded_datasets),
        reservation_presence=dict(physically_present=len(set(reserved_prompt_groups or set()) & split_groups['heldout']),
            not_present_in_selected_corpus=len(set(reserved_prompt_groups or set()) - split_groups['heldout']),
            inactive_due_excluded_dataset=sum(bool(datasets) and set(datasets) <= set(excluded_datasets)
                for group,datasets in (reserved_prompt_group_datasets or {}).items()
                if group in set(reserved_prompt_groups or set()) - split_groups['heldout']),
            policy='entire prior blocklist enforced across active datasets; absent groups are not fabricated'),
        array_shapes={s+'_'+kind:[counts[s],sequence_length+(kind=='tokens')]
            for s in split_groups for kind in ('tokens','masks')},
        transfer_bytes=sum(s.get('transferred_bytes',0) for s in receipts),
        final_batch_policy='zero-loss padding only; no repeated real conversations')
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest


def audit_prepared(raw_directory, prepared, tokenizer):
    """Re-read original byte offsets and independently re-render every retained row."""
    import numpy as np
    prepared = Path(prepared)
    manifest = json.loads((prepared/'manifest.json').read_text())
    sources = {s['key']:s for s in manifest['sources']}
    verified = targets = tool_calls = selected = masked_context = 0
    for split in ('train','heldout'):
        tokens = np.load(prepared/(split+'-tokens.npy'),mmap_mode='r')
        masks = np.load(prepared/(split+'-masks.npy'),mmap_mode='r')
        count = 0
        with (prepared/(split+'-dialogues.jsonl')).open() as handle:
            for index,line in enumerate(handle):
                record = json.loads(line)
                source = sources[record['source_key']]
                with (Path(raw_directory)/source['raw_file']).open('rb') as raw:
                    raw.seek(record['source_byte_offset'])
                    original = json.loads(raw.readline())
                encoded = encode_record(original,source['dataset'],tokenizer,manifest['sequence_length'])
                if encoded is None or list(tokens[index]) != encoded['tokens']:
                    raise ValueError('prepared token IDs differ from original full official template')
                if list(masks[index]) != encoded['loss_mask']:
                    raise ValueError('prepared selected-assistant mask differs from original full render')
                verified += 1
                count += 1
                targets += sum(encoded['loss_mask'])
                selected += sum(encoded['selected'])
                masked_context += sum(m['role']=='assistant' and not flag for m,flag in zip(encoded['messages'],encoded['selected']))
                tool_calls += sum(len(m.get('tool_calls') or []) for m in encoded['messages'])
        if count != len(tokens) or count != manifest[split+'_dialogues']:
            raise ValueError('prepared conversation count mismatch')
    return dict(status='passed',conversations_verified=verified,loss_tokens_verified=targets,
                selected_assistant_messages=selected,context_only_assistant_messages=masked_context,
                structured_tool_calls_verified=tool_calls,source_offsets_reread=True,
                full_official_render_bytes_and_ids_equal=True,checkpoint_bytes_read=False)
