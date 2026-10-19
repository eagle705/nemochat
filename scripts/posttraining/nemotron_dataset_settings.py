#!/usr/bin/env python3
"""Validate pinned dataset settings and preview CPU-only preparation.

Default/--check/--preview are inert. --execute-prepare is explicit data preparation,
not a GPU/training launcher. Run execution only in the approved cached compatible
CPU image; this wrapper never installs packages or launches a container.
"""
import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess

from nemochat.data.nemotron_sft import pinned_sources
from nemochat.data.sft_dataset_settings import (
    ROOT, DEFAULT_CONFIG, compile_plan, load_settings, repository_path,
)


def execute_prepare(settings, plan):
    """Run only the trusted existing producer, with CUDA hidden and fresh output."""
    if plan['prepare_argv'] is None:
        raise ValueError('--execute-prepare requires an SFT stage and --raw/--output')
    migration = json.loads(repository_path(settings['tokenizer']['manifest']).read_text())
    expected = migration['runtime_verification']['transformers_version']
    if version('transformers') != expected:
        raise ValueError(f'CPU preparation runtime requires approved cached Transformers {expected} '
                         f'TokenizersBackend; found {version("transformers")}. '
                         'Do not upgrade the host; run explicitly in the approved CPU image.')
    argv = plan['prepare_argv']
    raw = repository_path(argv[argv.index('--raw')+1])
    repository_path(argv[argv.index('--output')+1], fresh=True)
    expected_sources = pinned_sources(repository_path(settings['discovery']))
    receipts = {p.stem.removesuffix('.receipt'): json.loads(p.read_text()) for p in raw.glob('*.receipt.json')}
    if set(receipts) != {source['key'] for source in expected_sources}:
        raise ValueError('raw receipt set must contain exactly the eight pinned sources')
    for source in expected_sources:
        receipt = receipts[source['key']]
        if any(receipt.get(key) != value for key, value in source.items()):
            raise ValueError('raw receipt pinned identity mismatch: ' + source['key'])
        path = repository_path(raw/receipt['raw_file'])
        if path.parent != raw or not path.is_file():
            raise ValueError('raw receipt must name an existing file in the raw directory')
        if receipt['purpose'] == 'sft' and (receipt['retained_bytes'] != source['expected_bytes']
                                           or path.stat().st_size != source['expected_bytes']):
            raise ValueError('full eligible selection requires complete pinned raw source receipts')
    environment = os.environ.copy()
    environment.update(plan['execution_environment'])
    # No shell, arbitrary executable, remote URL, container creation or GPU activation.
    subprocess.run(argv, cwd=ROOT, env=environment, check=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check', action='store_true', help='validate registry and actual frozen local inputs (default)')
    mode.add_argument('--preview', action='store_true', help='preview stage plan; no preparation or training')
    mode.add_argument('--execute-prepare', action='store_true', help='explicit CPU data preparation in approved compatible runtime only')
    parser.add_argument('--stage', choices=['current_sft', 'deferred_long_context', 'reserved_rl_seed'], default='current_sft')
    parser.add_argument('--raw', type=Path, help='complete pinned raw directory; required with --output to emit command')
    parser.add_argument('--output', type=Path, help='fresh prepared data directory inside repository; never overwritten')
    parser.add_argument('--write-plan', type=Path, help='write a small JSON plan to a fresh repository path; never starts preparation')
    parser.add_argument('--json', action='store_true', help='print complete validated plan as JSON')
    args = parser.parse_args(argv)
    try:
        settings = load_settings(repository_path(args.config))
        plan = compile_plan(settings, args.stage, raw=args.raw, output=args.output)
        if args.execute_prepare:
            execute_prepare(settings, plan)
        text = json.dumps(plan, indent=2) + '\n'
        if args.write_plan:
            path = repository_path(args.write_plan, fresh=True)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('x') as handle:
                handle.write(text)
        if args.json:
            print(text, end='')
        else:
            counts = plan['validation']['counts']
            print(f"passed: {counts['datasets']} pinned datasets; {counts['current_sft']} current SFT, "
                  f"{counts['deferred_long_context']} deferred, {counts['reserved_rl_seed']} RL reserved; "
                  f"stage={args.stage}; native 128k launch ready=false")
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
