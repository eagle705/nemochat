"""Reuse an accepted reserved-ID alias migration; never change any weight.

This gate does not assert that reserved image/audio token semantics are equal.
It permits the previously audited 18..29 name transition on the exact ClimbMix
corpora, with the exact accepted new tokenizer backend and zero occurrences.
"""
import json
from pathlib import Path


def check(student, teacher, manifest_path, root):
    from nemochat.tokenization.posttraining import compare_tokenizers
    root = Path(root)
    manifest = json.loads(Path(manifest_path).read_text())
    if (manifest.get('status') != 'passed' or manifest.get('old_revision') != '2dc98e2afe4face0e4ce40972a915c45368bd34a'
            or manifest.get('new_revision') != 'fc067b34f6b514014d2f8bddba855d2e4bfdfba6'
            or manifest.get('allowed_changed_ids') != list(range(18, 30))
            or manifest.get('control_tokens_unchanged') is not True):
        raise ValueError('accepted tokenizer migration identity required')
    old = json.loads((Path(student) / 'tokenizer.json').read_text())
    new = json.loads((Path(teacher) / 'tokenizer.json').read_text())
    accepted = json.loads((root / manifest['tokenizer_path'] / 'tokenizer.json').read_text())
    if new != accepted:
        raise ValueError('teacher backend differs from exact accepted tokenizer migration')
    comparison = compare_tokenizers(old, new)
    required = {'climbmix_train': ('datasets/climbmix/nemochat_baby/native-02/train/corpus.bin', 10000269313),
                'climbmix_validation': ('datasets/climbmix/nemochat_baby/native-02/validation/corpus.bin', 2000000)}
    counts = manifest.get('datasetReservedCounts', {})
    for name, (path, tokens) in required.items():
        record = counts.get(name, {})
        if (record.get('status') != 'passed' or record.get('path') != path
                or record.get('tokens_scanned') != tokens or record.get('dtype') != 'int32'
                or set(record.get('stored_counts', {})) != {str(i) for i in range(18, 30)}
                or any(record['stored_counts'].values())):
            raise ValueError('complete zero-occurrence ClimbMix migration audit required')
    return dict(token_id_semantics_equal=False, ordinary_text_ids_equal=True,
                changed_reserved_ids=comparison['changed_ids'], migration_reused=True,
                migration_manifest=str(manifest_path), runtime_tokenizer=str(teacher),
                reserved_semantics='adopt_teacher_names_for_previously_reserved_ids',
                train_and_validation_reserved_ids_absent_by_prior_complete_audit=True,
                fresh_full_corpus_rescan_performed=False, weight_permutation_performed=False,
                checkpoint_integrity='not_checked_explicit_user_waiver')
