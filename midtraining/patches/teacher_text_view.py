#!/usr/bin/env python3
"""Extract the frozen language backbone without executing Hub Python code.

Safetensors payload bytes are copied unchanged. Vision/projector and unused MTP
heads are excluded explicitly; original downloaded shards remain untouched.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import struct


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def convert(source, output):
    source, output = Path(source), Path(output)
    config = json.loads((source / 'config.json').read_text())
    if config.get('model_type') != 'nemotron_h_omni' or config.get('llm_config', {}).get('model_type') != 'nemotron_h':
        raise ValueError('explicit Nemotron Omni language-backbone source required')
    weight_map = json.loads((source / 'model.safetensors.index.json').read_text())['weight_map']
    selected = {key: value for key, value in weight_map.items()
                if key.startswith('language_model.backbone.') or key == 'language_model.lm_head.weight'}
    if not selected or 'language_model.lm_head.weight' not in selected:
        raise ValueError('language backbone and LM head required')
    output.mkdir(parents=True, exist_ok=True)
    identity = {'source': str(source.resolve()), 'operation': 'language_model-prefix-strip-backbone-and-lm-head-only',
                'mtp': 'excluded_not_needed_for_next_token_teacher'}
    identity_path = output / 'source-identity.json'
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError('existing text view belongs to another source')
    write_json(identity_path, identity)
    result_map = {}; total_bytes = 0; completed = []
    for shard in sorted(set(selected.values())):
        if Path(shard).name != shard:
            raise ValueError('nested/traversing shard path rejected')
        keys = {key for key, value in selected.items() if value == shard}
        destination = output / shard
        with (source / shard).open('rb') as reader:
            size = struct.unpack('<Q', reader.read(8))[0]
            if size > 64 * 1024 * 1024:
                raise ValueError('unreasonable safetensors header')
            header = json.loads(reader.read(size)); payload_start = 8 + size
            tensors = sorted(keys, key=lambda key: header[key]['data_offsets'][0])
            next_offset = 0; new_header = {'__metadata__': {'format': 'pt'}}
            for key in tensors:
                entry = dict(header[key]); start, end = entry['data_offsets']
                if not 0 <= start <= end <= (source / shard).stat().st_size - payload_start:
                    raise ValueError('tensor offset outside source shard')
                entry['data_offsets'] = [next_offset, next_offset + end - start]
                next_offset += end - start
                new_header[key.removeprefix('language_model.')] = entry
            encoded = json.dumps(new_header, separators=(',', ':')).encode()
            encoded += b' ' * ((-len(encoded)) % 8)
            expected_size = 8 + len(encoded) + next_offset
            # Restartable conversion: only a matching header and full-length output is reusable.
            reusable = False
            if destination.exists() and destination.stat().st_size == expected_size:
                with destination.open('rb') as old:
                    n = struct.unpack('<Q', old.read(8))[0]
                    reusable = json.loads(old.read(n)) == new_header
            if not reusable:
                temporary = destination.with_suffix(destination.suffix + '.partial')
                with temporary.open('wb') as writer:
                    writer.write(struct.pack('<Q', len(encoded))); writer.write(encoded)
                    for key in tensors:
                        start, end = header[key]['data_offsets']; reader.seek(payload_start + start)
                        remaining = end - start
                        while remaining:
                            data = reader.read(min(16 * 1024 * 1024, remaining))
                            if not data:
                                raise ValueError('truncated source tensor')
                            writer.write(data); remaining -= len(data)
                    writer.flush(); os.fsync(writer.fileno())
                temporary.replace(destination)
        total_bytes += next_offset; completed.append(shard)
        result_map.update({key.removeprefix('language_model.'): shard for key in tensors})
        write_json(output / 'conversion-progress.json', {'status': 'converting', 'completed_shards': completed,
                   'expected_shards': len(set(selected.values())), 'tensor_bytes_written': total_bytes})
    text_config = dict(config['llm_config'])
    text_config.pop('auto_map', None)
    text_config.update(architectures=['NemotronHForCausalLM'], num_nextn_predict_layers=0)
    write_json(output / 'config.json', text_config)
    write_json(output / 'model.safetensors.index.json', {'metadata': {'total_size': total_bytes}, 'weight_map': result_map})
    for name in ['tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'chat_template.jinja', 'LICENSE', 'LICENSE.txt']:
        if (source / name).is_file():
            shutil.copyfile(source / name, output / name)
    receipt = dict(identity, status='text_backbone_materialized_not_gpu_validated', language_tensors=len(result_map),
                   excluded_tensors=len(weight_map) - len(result_map), shards=len(completed), tensor_bytes=total_bytes,
                   trust_remote_code=False)
    write_json(output / 'conversion-completed.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(convert(args.source, args.output), indent=2))


if __name__ == '__main__':
    main()
