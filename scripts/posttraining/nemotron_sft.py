#!/usr/bin/env python3
"""CPU-only pinned Nemotron SFT discovery, bounded acquisition and preparation.

Run from repository root. Preparation uses the cached NeMo image's compatible
Transformers, not a host package upgrade. Never launches a GPU or labels RL seeds.
"""
import argparse
import json
from pathlib import Path
from nemochat.data.nemotron_sft import pinned_sources, acquire_source, prepare_subset

DISCOVERY = Path('results/pretraining/nemochat_baby/nemotron_sft/discovery')
MIGRATION = Path('results/pretraining/nemochat_baby/tokenizer_migration/manifest.json')
EXCLUSIONS = Path('results/pretraining/nemochat_baby/sft/data/raw/kmmlu-reserved-questions.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    discover = commands.add_parser('discover', help='read existing frozen metadata; never refresh pins')
    discover.add_argument('--discovery', type=Path, default=DISCOVERY)
    acquire = commands.add_parser('acquire', help='bounded prefixes only, never full multi-GB corpus')
    acquire.add_argument('--discovery', type=Path, default=DISCOVERY)
    acquire.add_argument('--output', type=Path, required=True)
    acquire.add_argument('--rows-per-file', type=int, default=100000)
    acquire.add_argument('--per-file-mib', type=int, default=256)
    acquire.add_argument('--aggregate-mib', type=int, default=8192)
    acquire.add_argument('--seed-rows', type=int, default=4)
    acquire.add_argument('--reuse-complete-prefixes', action='store_true')
    acquire.add_argument('--resume-incomplete-prefixes', action='store_true', help='exact Range or budgeted 200 replay; no duplicate append')
    prepare = commands.add_parser('prepare', help='official full-template tokens and selected-assistant masks')
    prepare.add_argument('--raw', type=Path, required=True)
    prepare.add_argument('--output', type=Path, required=True)
    prepare.add_argument('--tokenizer-manifest', type=Path, default=MIGRATION)
    prepare.add_argument('--kmmlu-exclusions', type=Path, default=EXCLUSIONS)
    prepare.add_argument('--quota-per-file', type=int, default=16)
    prepare.add_argument('--sequence-length', type=int, default=4096)
    prepare.add_argument('--exclude-dataset', action='append', default=[],
                         help='exclude entire pinned dataset from train and heldout; preserve registration')
    prepare.add_argument('--quota-json', type=Path, help='explicit source-key to eligible-row quota map')
    prepare.add_argument('--heldout-modulus', type=int, default=100)
    prepare.add_argument('--reserve-heldout', type=Path, help='prior prepared heldout-dialogues.jsonl; preserve ENTIRE reservation')
    args = parser.parse_args()
    if args.command == 'discover':
        print(json.dumps(pinned_sources(args.discovery), indent=2))
    elif args.command == 'acquire':
        if not 1 <= args.aggregate_mib <= 8192 or not 1 <= args.per_file_mib <= 1024:
            parser.error('aggregate <=8192 MiB and per-file <=1024 MiB required')
        failed_ledger = args.output/'failed-transfers.jsonl'
        spent = sum(json.loads(line)['transferred_bytes'] for line in failed_ledger.open()) if failed_ledger.exists() else 0
        receipts = []
        for source in pinned_sources(args.discovery):
            prior = args.output/(source['key']+'.receipt.json')
            if args.reuse_complete_prefixes and prior.exists():
                receipt = json.loads(prior.read_text())
                if any(receipt.get(k) != source[k] for k in source):
                    raise ValueError('existing prefix identity mismatch')
                if (args.output/receipt['raw_file']).stat().st_size != receipt['retained_bytes']:
                    raise ValueError('existing prefix size mismatch')
            else:
                cap = min(args.per_file_mib*1024**2, args.aggregate_mib*1024**2-spent)
                if cap <= 0:
                    raise ValueError('aggregate transfer budget exhausted')
                receipt = acquire_source(source, args.output, cap,
                    args.seed_rows if source['purpose'] == 'reserved_rl_seed' else args.rows_per_file,
                    resume=args.resume_incomplete_prefixes)
            spent += receipt['transferred_bytes']
            if spent > args.aggregate_mib*1024**2:
                raise ValueError('aggregate transfer budget exceeded')
            receipts.append(receipt)
            print(json.dumps({'key':source['key'],'rows':receipt['rows'],'transferred_bytes':spent}), flush=True)
        (args.output/'acquisition.json').write_text(json.dumps({'sources':receipts,'transferred_bytes':spent},indent=2))
    else:
        if args.quota_per_file < 1 or args.heldout_modulus < 2:
            parser.error('positive quota and heldout modulus >=2 required')
        from nemochat.tokenization.posttraining import load_posttraining_tokenizer
        tok = load_posttraining_tokenizer(args.tokenizer_manifest)
        exclusions = json.loads(args.kmmlu_exclusions.read_text())
        if exclusions.get('repository') != 'HAERAE-HUB/KMMLU' or not exclusions.get('questions'):
            raise ValueError('complete prior KMMLU dev/test reservation required')
        reserved = {json.loads(line)['prompt_group'] for line in args.reserve_heldout.open()} if args.reserve_heldout else set()
        reserved_sources = {}
        if args.reserve_heldout:
            for line in args.reserve_heldout.open():
                row = json.loads(line)
                reserved_sources.setdefault(row['prompt_group'], []).append(row['source_dataset'])
        report = prepare_subset(args.raw,args.output,tok,set(exclusions['questions']),
            quota_per_file=args.quota_per_file, heldout_modulus=args.heldout_modulus,
            quotas=json.loads(args.quota_json.read_text()) if args.quota_json else None,
            reserved_prompt_groups=reserved, sequence_length=args.sequence_length,
            excluded_datasets=args.exclude_dataset, reserved_prompt_group_datasets=reserved_sources)
        # Bind actual tokenizer and reserved-question lineage to prepared arrays.
        report['tokenizer_manifest'] = str(args.tokenizer_manifest)
        report['prior_heldout_reservation_file'] = str(args.reserve_heldout) if args.reserve_heldout else None
        report['tokenizer_path'] = json.loads(args.tokenizer_manifest.read_text())['tokenizer_path']
        report['tokenizer_revision'] = json.loads(args.tokenizer_manifest.read_text())['new_revision']
        report['decontamination'] = dict(repository=exclusions['repository'],revision=exclusions['revision'],
            reserved_question_file=str(args.kmmlu_exclusions),question_count=len(exclusions['questions']),
            scope='all dev/test normalized exact user question or question prefix; not semantic dedup')
        (args.output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({k:report[k] for k in ['status','missing_sft_sources','train_dialogues','heldout_dialogues','per_source']},indent=2))
        if report['status'] != 'passed':
            raise SystemExit('Requested SFT source missing eligible complete examples; inspect manifest before GPU launch')


if __name__ == '__main__':
    main()
