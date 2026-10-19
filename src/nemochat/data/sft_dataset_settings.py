"""CPU-only, fail-closed dataset settings; never imports or launches a trainer."""
import json
from collections import Counter
from pathlib import Path

from nemochat.data.nemotron_sft import PINS, pinned_sources

import yaml

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / 'sft/configs/nemotron_dataset_settings.yaml'


def load_settings(path=DEFAULT_CONFIG):
    """Read the single dataset-settings authority without side effects."""
    with Path(path).open() as handle:
        return yaml.safe_load(handle)


def repository_path(value, *, root=ROOT, fresh=False):
    """Reject traversal, symlink escapes and overwriting existing output paths."""
    path = Path(value)
    if '..' in path.parts:
        raise ValueError('path traversal is forbidden')
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root.resolve()) or resolved == root.resolve():
        raise ValueError('path must be inside repository root')
    if fresh and (resolved.exists() or (root / path).is_symlink()):
        raise ValueError('fresh output path required')
    return resolved


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _same(actual, expected, label):
    # JSON also distinguishes true from 1, unlike ordinary Python equality.
    _require(json.dumps(actual, sort_keys=True) == json.dumps(expected, sort_keys=True), label)


def validate_settings(settings, *, root=ROOT):
    """Validate pinned identities and compare against actual frozen local inputs.

    This checks CPU configuration evidence, not GPU launch readiness or model quality.
    No weight files are inspected and no tokenizer is instantiated here.
    """
    try:
        return _validate(settings, root)
    except FileNotFoundError as error:
        raise ValueError('local evidence unavailable; restore approved receipts before validation') from error
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError('malformed or incomplete dataset settings: ' + str(error)) from error


def _validate(s, root):
    _same(s['schema'], 'nemochat-nemotron-dataset-settings-v1', 'unsupported settings schema')
    sources = pinned_sources(repository_path(s['discovery'], root=root))
    rows = s['datasets']
    _require(len(rows) == len(PINS), 'exactly the five frozen datasets required')
    _same({row['id']: row['revision'] for row in rows},
          {'nvidia/' + name: rev for name, rev in PINS.items()}, 'dataset set/revision mismatch')
    _require(len({row['id'] for row in rows}) == len(rows), 'duplicate dataset')
    expected_stage = {}
    for row in rows:
        name = row['id']
        stage = ('reserved_rl_seed' if 'Seed' in name else
                 'deferred_long_context' if 'Long-Context' in name else 'current_sft')
        expected_stage[name] = stage
        _same(row['stage'], stage, 'dataset stage mismatch: ' + name)
        selected = [source for source in sources if source['dataset'] == name]
        _same(sorted(row['sources']), sorted(source['path'] for source in selected), 'source paths mismatch')
        _require(all(row['license'] == source['license'] for source in selected), 'pinned card license mismatch')
        _same(row['train'], stage != 'reserved_rl_seed', 'strict train boolean required')
        _same(row['heldout'], stage != 'reserved_rl_seed', 'strict heldout boolean required')
        _same(row['sequence_length'], s['stages'][stage]['sequence_length'], 'dataset sequence length mismatch')
        if 'Instruction-Following' in name:
            _same(row['train_turns_mask'], 'required_boolean_all_messages_aligned_not_assistant_subsequence', 'IFC mask contract')
            target = 'author_selected_assistants_only_false_assistants_remain_context'
        elif stage == 'reserved_rl_seed':
            target = 'none_no_gold_answers'
        elif stage == 'deferred_long_context':
            target = 'complete_reference_no_truncation'
        else:
            target = 'author_selected_assistants_else_all_assistants'
        _same(row['assistant_target_policy'], target, 'assistant target policy mismatch')

    current = load_settings(repository_path(s['current_runtime_config'], root=root))
    long = load_settings(repository_path(s['long_context_plan'], root=root))
    c, future = s['stages']['current_sft'], s['stages']['deferred_long_context']
    for key in ('sequence_length', 'context_parallel_size', 'expert_model_parallel_size',
                'global_batch_size', 'micro_batch_size', 'epochs', 'allow_repetition', 'seed', 'source_checkpoint'):
        _same(c[key], current[key], 'current runtime mismatch: ' + key)
    excluded = sorted(name for name, stage in expected_stage.items() if stage == 'deferred_long_context')
    _same(sorted(current['excluded_datasets']), excluded, 'current exclusion list mismatch')
    for key in ('sequence_length', 'precision', 'source_checkpoint', 'source_checkpoint_policy',
                'global_batch_size', 'auto_launch', 'native_launch_ready'):
        _same(future[key], long[key], 'long-context plan mismatch: ' + key)
    for key in ('micro_batch_size', 'physical_gpu_count', 'initial_context_parallel_size',
                'candidate_context_parallel_sizes', 'minimum_free_memory_fraction_after_training_step'):
        _same(future[key], long['parallelism_plan'][key], 'long-context memory plan mismatch: ' + key)
    _same(future['auto_launch'], False, 'long auto launch forbidden')
    _same(future['native_launch_ready'], False, '128k native validation still pending')
    _same(future['native_cp_validation'], 'pending_for_CP2_CP4_CP8', 'CP native validation not established')
    _same(long['changes_current_16k_sft'], False, 'long plan must not alter current stage')
    long_row = next(row for row in rows if row['stage'] == 'deferred_long_context')
    _same(long['dataset']['id'], long_row['id'], 'long plan dataset mismatch')
    _same(long['dataset']['revision'], long_row['revision'], 'long plan pin mismatch')
    _same(s['stages']['reserved_rl_seed'], dict(sequence_length=None, assistant_targets=False,
          train=False, heldout=False, policy='reserve_policies_tools_customer_scenarios_no_fabricated_answers'), 'RL seed must remain unlabeled')

    tokenizer = s['tokenizer']
    _same(tokenizer['manifest'], current['tokenizer_manifest'], 'current tokenizer manifest mismatch')
    _same(tokenizer['manifest'], long['tokenizer_manifest'], 'future tokenizer manifest mismatch')
    migration = json.loads(repository_path(tokenizer['manifest'], root=root).read_text())
    for key, source_key in [('repository', 'source_repository'), ('revision', 'new_revision'), ('vocab_size', 'vocab_size')]:
        _same(tokenizer[key], migration[source_key], 'tokenizer contract mismatch: ' + key)
    _same(migration['status'], 'passed', 'accepted migration required')
    _same(tokenizer['format'], current['format'], 'official format mismatch')
    for key in ('enable_thinking', 'remote_code', 'weight_changes'):
        _same(tokenizer[key], False, 'nonthinking local unchanged-weight contract required')

    prep = s['preparation']
    for key, expected in [('selection', 'full_eligible_no_fixed_mixture_weights_or_quotas'),
                          ('complete_prompt_plus_target', True), ('truncate', False),
                          ('oversized', 'exclude_and_count'),
                          ('tool_policy', 'preserve_tools_call_ids_response_ids_parallel_order_decode_JSON_object_arguments'),
                          ('reasoning_content', 'omit_nonthinking_preserve_official_empty_think_markers'),
                          ('mask_policy', 'official_full_render_selected_assistant_content_calls_EOT_not_headers_users_tools'),
                          ('inactive_excluded_source_groups', 'retain_blocklist_do_not_fabricate_rows'),
                          ('parity', 'official_render_bytes_and_full_token_ids_equal_then_source_offset_rerender_audit')]:
        _same(prep[key], expected, 'preparation contract mismatch: ' + key)
    _require(type(prep['heldout_modulus']) is int and prep['heldout_modulus'] >= 2, 'heldout modulus >=2 required')
    _same(prep['reserve_heldout'], current['prior_heldout_reservation'], 'reservation lineage mismatch')
    reservation = [json.loads(line) for line in repository_path(prep['reserve_heldout'], root=root).read_text().splitlines()]
    groups = {row['prompt_group'] for row in reservation}
    _same(len(groups), prep['reserved_prompt_groups'], 'entire prompt-group blocklist required')
    kmmlu = json.loads(repository_path(prep['kmmlu_exclusions'], root=root).read_text())
    _same(kmmlu['repository'], 'HAERAE-HUB/KMMLU', 'KMMLU reservation identity')
    _same(len(kmmlu['questions']), prep['kmmlu_reserved_questions'], 'complete KMMLU reservation required')
    audit = json.loads(repository_path(prep['schema_quarantine']['audit'], root=root).read_text())
    reasons = Counter()
    for source in audit['per_source']:
        reasons.update(source['rejection_counts'])
        if source['source_key'] == 'Nemotron-SFT-BBQ-v1--train':
            bbq = next(row for row in rows if row['id'].endswith('Nemotron-SFT-BBQ-v1'))
            _same(bbq['verified_raw_rows'], source['counts']['visited'], 'BBQ actual row count mismatch')
    _same(dict(reasons), prep['schema_quarantine']['reason_counts'], 'typed quarantine reasons mismatch')
    _same(audit['rejected_records'], sum(reasons.values()), 'quarantine count mismatch')
    _same(audit['status'], 'passed', 'schema audit must pass')
    _same(audit['fatal_unclassified_records'], 0, 'unclassified schema failures remain fatal')
    counts = dict(Counter(row['stage'] for row in rows))
    counts.update(datasets=len(rows), source_files=len(sources),
                  labeled_source_files=sum(source['purpose'] == 'sft' for source in sources))
    return dict(status='passed', counts=counts, runtime_crosschecks=True,
                reserved_prompt_groups=len(groups), kmmlu_reserved_questions=len(kmmlu['questions']),
                schema_rejected_records=sum(reasons.values()), native_128k_launch_ready=False,
                weight_files_read=False, gpu_used=False)


def compile_plan(settings, stage='current_sft', *, raw=None, output=None, root=ROOT):
    """Consume validated settings into a data-only command, never a trainer command.

    The legacy producer requires a positive quota. A strict upper bound on physical
    JSONL rows (largest pinned byte size + 1) disables early stopping without a
    manually invented mixture, generated quota file, or modifying live producer code.
    """
    import sys
    validation = validate_settings(settings, root=root)
    _require(stage in settings['stages'], 'unknown stage')
    _require((raw is None) == (output is None), '--raw and --output are required together')
    selected = [row for row in settings['datasets'] if row['stage'] == stage and row['train'] is True]
    excluded = sorted(row['id'] for row in settings['datasets']
                      if row['stage'] not in (stage, 'reserved_rl_seed'))
    plan = dict(schema='nemochat-nemotron-dataset-preparation-plan-v1', stage=stage,
                validation=validation, active_datasets=[row['id'] for row in selected],
                reserved_rl_datasets=[row['id'] for row in settings['datasets'] if row['stage'] == 'reserved_rl_seed'],
                excluded_datasets=excluded, settings=settings['stages'][stage],
                dataset_settings=selected, preparation_policy=settings['preparation'],
                tokenizer=settings['tokenizer'], prepare_argv=None, native_training_argv=None,
                execution_environment={'CUDA_VISIBLE_DEVICES': '', 'TOKENIZERS_PARALLELISM': 'false'},
                status='validated_data_plan_not_training_acceptance')
    if raw is not None:
        _require(stage != 'reserved_rl_seed', 'RL seed is reserved, not preparable SFT')
        raw_path = repository_path(raw, root=root)
        output_path = repository_path(output, root=root, fresh=True)
        _require(not output_path.is_relative_to(raw_path), 'output must not be inside raw inputs')
        sources = pinned_sources(repository_path(settings['discovery'], root=root))
        row_upper_bound = max(source['expected_bytes'] for source in sources) + 1
        prep = settings['preparation']
        argv = [sys.executable, str(root/'sft/data/nemotron_sft.py'), 'prepare',
                '--raw', str(raw_path), '--output', str(output_path),
                '--tokenizer-manifest', str(repository_path(settings['tokenizer']['manifest'], root=root)),
                '--kmmlu-exclusions', str(repository_path(prep['kmmlu_exclusions'], root=root)),
                '--reserve-heldout', str(repository_path(prep['reserve_heldout'], root=root)),
                '--heldout-modulus', str(prep['heldout_modulus']),
                '--sequence-length', str(settings['stages'][stage]['sequence_length']),
                '--quota-per-file', str(row_upper_bound)]
        for dataset in excluded:
            argv.extend(['--exclude-dataset', dataset])
        plan['prepare_argv'] = argv
        plan['quota_compatibility_policy'] = 'physical_byte_upper_bound_disables_legacy_row_quota_no_mixture_weights'
    return plan
