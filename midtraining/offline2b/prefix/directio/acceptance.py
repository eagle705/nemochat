"""Immutable expectation preparation from authenticated historical cache audit."""
import hashlib
import json
from pathlib import Path
from .binding import source_contract
from .transport import expectation

ROOT = Path('/raid/yoon/nemochat')
OUT = ROOT / 'results/midtraining/offline1920-parent-01/directio-acceptance-01'
SCR = Path('/home/yoon/.hermes/cache/scratch/offline1920-prefix03-student-stt7nda8')
SRC = SCR / 'staged/src/megatron/bridge/training/cached_kd/utils_logits.py'
IMAGE = 'nvcr.io/nvidia/nemo@sha256:fdd6e9c7929b76c8624ddfea939dda345beaf67695d9c3994994c18f9e79b9bf'
CACHE = ROOT / 'results/midtraining/offline2b-production1920-01/cache'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_manifest():
    audit = ROOT / 'results/midtraining/offline1920-parent-01/cache-codec-audit-01/shards.jsonl'
    proof = ROOT / 'results/midtraining/offline1920-parent-01/readpath-investigation-01/comparison.json'
    previous = [json.loads(x) for x in audit.read_text().splitlines()]
    compared = json.loads(proof.read_text())
    correct70 = next(x['direct'] for x in compared if x['path'].endswith('__70.tar'))
    shards = []
    for row in previous:
        sha = correct70['sha256'] if (row['rank'], row['iteration']) == (7, 70) else row['sha256']
        if not row['ok'] and (row['rank'], row['iteration']) != (7, 70):
            raise RuntimeError('unauthenticated bad historical audit entry')
        e = expectation(CACHE / row['name'], sha)
        assert e.stat[2] == row['bytes']
        payload_sha = correct70['members'][1]['sha256'] if (row['rank'], row['iteration']) == (7, 70) else row['payload_sha256']
        shards.append({'path': e.path, 'sha256': e.sha256, 'stat': e.stat,
                       'rank': row['rank'], 'iteration': row['iteration'], 'payload_sha256': payload_sha})
    shards.sort(key=lambda x: (x['iteration'], x['rank']))
    pairs = [(x['rank'], x['iteration']) for x in shards]
    expected = {(r, i) for r in range(8) for i in range(1143)}
    if len(pairs) != 9144 or len(set(pairs)) != 9144 or set(pairs) != expected:
        raise RuntimeError('full audit rank/iteration coverage mismatch')
    if {str(p) for p in CACHE.glob('*.tar')} != {x['path'] for x in shards}:
        raise RuntimeError('closed-tar inventory differs from audit')
    budget_path = ROOT / 'results/midtraining/offline2b-prefixtrain-01/continuation-03/run-config.json'
    budget = json.loads(budget_path.read_text())
    metadata = compared[0]['direct']['members'][0]['metadata']
    return {'version': 1, 'image': IMAGE, 'native_source': str(SRC), 'sdk_root': str(SCR/'sdk'),
            'native_contract': source_contract(SRC), 'metadata': metadata, 'metadata_hash': metadata['hash'],
            'budget': budget, 'checkpoint_weights_read': False, 'checkpoint_integrity': 'not_checked_explicit_user_waiver',
            'override': {'rank': 7, 'iteration': 70, 'sha256': correct70['sha256'],
                         'provenance': 'authenticated regenerated tar + historical direct-byte comparison; original path retained'},
            'provenance': {str(p): digest(p) for p in (audit, proof, budget_path)},
            'shards': shards}


def freeze():
    path = OUT / 'expectations.json'
    payload = (json.dumps(build_manifest(), sort_keys=True, separators=(',', ':'))+'\n').encode()
    with path.open('xb') as f: f.write(payload)
    path.chmod(0o444)
    receipt = {'path': str(path), 'sha256': hashlib.sha256(payload).hexdigest(), 'bytes': len(payload)}
    with (OUT/'expectations-receipt.json').open('x') as f: json.dump(receipt, f, indent=2)
    return receipt


if __name__ == '__main__':
    print(json.dumps(freeze(), indent=2))
