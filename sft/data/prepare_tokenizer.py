#!/usr/bin/env python3
"""Apply the approved name-only Super3.5 tokenizer migration, without GPUs/weights.

Run download, audit, verify-runtime (pinned CPU image), then publish, in order.
Only tokenizer assets are hashed; no checkpoint file is opened by this tool.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
import numpy as np
import yaml
from nemochat.tokenization.posttraining import (
    ALLOWED_CHANGED_IDS, compare_tokenizers, open_indexed_tokens,
    scan_reserved_ids, render_chat,
)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def asset_receipt(directory, names):
    return {name: {'bytes': (directory / name).stat().st_size,
                   'sha256': hashlib.sha256((directory / name).read_bytes()).hexdigest()}
            for name in names}


def read_chat_template(directory):
    path = directory / 'chat_template.jinja'
    if path.exists():
        return path.read_text()
    return json.loads((directory / 'tokenizer_config.json').read_text()).get('chat_template')


def verify_old_official(cache, official):
    for name in ['tokenizer.json', 'tokenizer_config.json']:
        if (cache / name).read_bytes() != (official / name).read_bytes():
            raise ValueError(f'cached tokenizer differs from pinned official old tokenizer: {name}')
    return {'cached_backend_identical': True, 'cached_config_identical': True,
            'cached_chat_template_present': (cache / 'chat_template.jinja').exists(),
            'official_assets': asset_receipt(official, [p.name for p in official.iterdir() if p.is_file()])}


def download(config):
    revision = config['new_revision']
    if not re.fullmatch('[0-9a-f]{40}', revision):
        raise ValueError('immutable full revision required')
    allowed = {'tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja', 'README.md'}
    if set(config['assets']) != allowed:
        raise ValueError('tokenizer-only asset allowlist violated')
    directory = Path(config['tokenizer_path'])
    directory.mkdir(parents=True, exist_ok=True)
    for name in config['assets']:
        url = f"https://huggingface.co/{config['source_repository']}/resolve/{revision}/{name}"
        data = urllib.request.urlopen(url, timeout=120).read()
        target = directory / name
        if target.exists() and target.read_bytes() != data:
            raise ValueError(f'will not overwrite different asset: {target}')
        if not target.exists():
            with target.open('xb') as handle:
                handle.write(data)
    old = Path(config['old_tokenizer_path'])
    old_official = Path(config['output_directory']) / 'old-official'
    old_official.mkdir(exist_ok=True, parents=True)
    for name in ['tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja']:
        url = f"https://huggingface.co/{config['old_source_repository']}/resolve/{config['old_revision']}/{name}"
        data = urllib.request.urlopen(url, timeout=120).read()
        target = old_official / name
        if target.exists() and target.read_bytes() != data:
            raise ValueError(f'will not overwrite different old official asset: {target}')
        if not target.exists():
            with target.open('xb') as handle:
                handle.write(data)
    provenance = verify_old_official(old, old_official)
    comparison = compare_tokenizers(json.loads((old / 'tokenizer.json').read_text()),
                                    json.loads((directory / 'tokenizer.json').read_text()))
    comparison['new_assets'] = asset_receipt(directory, config['assets'])
    comparison['old_assets'] = asset_receipt(old, ['tokenizer.json', 'tokenizer_config.json'])
    comparison['chat_template_equal'] = read_chat_template(old_official) == read_chat_template(directory)
    comparison['old_official_provenance'] = provenance
    comparison['old_source_repository'] = config['old_source_repository']
    comparison['old_revision'] = config['old_revision']
    comparison['tokenizer_config_equal'] = (old / 'tokenizer_config.json').read_bytes() == (directory / 'tokenizer_config.json').read_bytes()
    comparison['source_repository'] = config['source_repository']
    comparison['new_revision'] = revision
    write_json(Path(config['output_directory']) / 'comparison.json', comparison)
    return comparison


def audit(config):
    results = {}
    output = Path(config['output_directory']) / 'dataset-audit.json'
    for name, spec in config['corpora'].items():
        if 'prefix' in spec:
            tokens, metadata = open_indexed_tokens(spec['prefix'])
            if tokens.size != spec['tokens']:
                raise ValueError(f'{name} corpus token count mismatch')
            path = spec['prefix'] + '.bin'
        else:
            tokens = np.load(spec['npy'], mmap_mode='r', allow_pickle=False)
            if list(tokens.shape) != spec['shape']:
                raise ValueError(f'{name} packed shape mismatch')
            metadata, path = {}, spec['npy']
        result = scan_reserved_ids(tokens)
        result.update(metadata)
        result['path'] = path
        results[name] = result
        write_json(output, results)
        print(name, result['tokens_scanned'], result['status'], flush=True)
    if any(r['status'] != 'passed' for r in results.values()):
        raise ValueError('reserved token occurrence blocks migration; see dataset-audit.json')
    return results


def verify_runtime(config):
    """Exercise real official AutoTokenizer and Jinja in the pinned CPU image."""
    import transformers
    from transformers import AutoTokenizer
    old = AutoTokenizer.from_pretrained(config['old_tokenizer_path'], local_files_only=True, trust_remote_code=False)
    new = AutoTokenizer.from_pretrained(config['tokenizer_path'], local_files_only=True, trust_remote_code=False)
    probes = ['Hello, world!', '안녕하세요. 한국어 토큰 검증입니다.',
              '¿Cómo estás? ¡Programación en español!',
              'def f(x):\n    return x ** 2\n', '\t  spaces\n\n trailing  ',
              '<|im_start|>assistant\n<think></think><tool_call>']
    parity = []
    for text in probes:
        a = old.encode(text, add_special_tokens=False)
        b = new.encode(text, add_special_tokens=False)
        if a != b:
            raise ValueError(f'ordinary/control text parity failed: {text!r}')
        parity.append({'text': text, 'ids': b, 'equal': True})
    backend = json.loads((Path(config['tokenizer_path']) / 'tokenizer.json').read_text())
    inverse = {i: text for text, i in backend['model']['vocab'].items()}
    for i in ALLOWED_CHANGED_IDS:
        if new.encode(inverse[i], add_special_tokens=False) != [i]:
            raise ValueError(f'special token mapping failed: {i}')
    messages = [{'role': 'system', 'content': 'Answer directly.'},
                {'role': 'user', 'content': '안녕, answer in Spanish.'},
                {'role': 'assistant', 'content': 'Hola.'}]
    rendered = render_chat(new, messages)
    generation = render_chat(new, messages[:2], add_generation_prompt=True)
    if '<|im_start|>assistant\n<think></think>' not in generation:
        raise ValueError('official no-thinking generation prompt not observed')
    if len(new) != 131072 or new.eos_token_id != 11:
        raise ValueError('vocab/EOS changed')
    tool_messages = [{'role': 'user', 'content': 'Check weather.'},
        {'role': 'assistant', 'content': '', 'tool_calls': [
            {'id': 'call_42', 'type': 'function', 'function': {'name': 'weather', 'arguments': {'city': 'Seoul'}}}]},
        {'role': 'tool', 'content': 'Sunny', 'tool_call_id': 'call_42'},
        {'role': 'assistant', 'content': 'Sunny.'}]
    tools = [{'type': 'function', 'function': {'name': 'weather', 'description': 'Weather',
               'parameters': {'type': 'object', 'properties': {'city': {'type': 'string'}}}}}]
    tool_render = render_chat(new, tool_messages, tools=tools)
    if '<tool_call>' not in tool_render or 'Sunny' not in tool_render:
        raise ValueError('tool structure not rendered')
    result = {'status': 'passed', 'transformers_version': transformers.__version__,
        'tokenizer_class': type(new).__name__, 'vocab_size': len(new), 'eos_id': new.eos_token_id,
        'ordinary_text_parity': parity, 'no_thinking_render': rendered,
        'no_thinking_generation_prompt': generation, 'tool_render': tool_render,
        'tool_call_id_rendered': 'call_42' in tool_render,
        'template_policy': 'official file unchanged; enable_thinking=False',
        'gpu_used': False, 'remote_code': False}
    write_json(Path(config['output_directory']) / 'runtime-verification.json', result)
    return result


def publish(config):
    output = Path(config['output_directory'])
    comparison = json.loads((output / 'comparison.json').read_text())
    audits = json.loads((output / 'dataset-audit.json').read_text())
    runtime = json.loads((output / 'runtime-verification.json').read_text())
    # Recheck actual tokenizer backend and asset bytes before publishing.
    actual = compare_tokenizers(json.loads((Path(config['old_tokenizer_path']) / 'tokenizer.json').read_text()),
                                json.loads((Path(config['tokenizer_path']) / 'tokenizer.json').read_text()))
    if actual['changed_ids'] != comparison['changed_ids'] or asset_receipt(Path(config['tokenizer_path']), config['assets']) != comparison['new_assets']:
        raise ValueError('tokenizer assets changed after verification')
    if set(audits) != set(config['corpora']) or any(r['status'] != 'passed' or sum(r['stored_counts'].values()) for r in audits.values()):
        raise ValueError('reserved-ID corpus gate failed')
    if runtime['status'] != 'passed':
        raise ValueError('runtime gate failed')
    manifest = {'schema': config['schema'], 'status': 'passed',
        'tokenizer_path': config['tokenizer_path'], 'source_repository': config['source_repository'],
        'old_revision': config['old_revision'], 'new_revision': config['new_revision'],
        'vocab_size': 131072, 'allowed_changed_ids': list(ALLOWED_CHANGED_IDS),
        'control_tokens_unchanged': True, 'datasetReservedCounts': audits,
        'backend_comparison': comparison, 'runtime_verification': runtime,
        'chat_format': config['chat_format'], 'enable_thinking': False,
        'weight_policy': 'unchanged: no resize, reorder, reinit, copies or checkpoint reads',
        'base_tokenizer_policy': 'original cached tokenizer and checkpoint untouched',
        'assistant_mask_policy': 'dataset adapter owns full-template spans and selected-turn masks'}
    write_json(output / 'manifest.json', manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['download', 'audit', 'verify-runtime', 'publish'])
    parser.add_argument('--config', default='sft/configs/tokenizer_super35.yaml')
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    result = {'download': download, 'audit': audit, 'verify-runtime': verify_runtime, 'publish': publish}[args.stage](config)
    print(json.dumps({'stage': args.stage, 'status': result.get('status', 'passed')}))


if __name__ == '__main__':
    main()
