"""Opt-in integration API for a NEW scratch generation; never launches anything."""
import hashlib
import json
from pathlib import Path
from .binding import install
from .transport import DirectTransport, Expectation


def load_transport(manifest_path, expected_sha256):
    data = Path(manifest_path).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise RuntimeError('expectation manifest SHA mismatch')
    manifest = json.loads(data)
    pairs = [(x['rank'], x['iteration']) for x in manifest['shards']]
    if len(pairs) != 9144 or len(set(pairs)) != 9144 or set(pairs) != {(r,i) for r in range(8) for i in range(1143)}:
        raise RuntimeError('incomplete/duplicate approved prefix')
    if any(Path(x['path']).name != f"cp0_dp{x['rank']}__{x['iteration']}.tar" for x in manifest['shards']):
        raise RuntimeError('rank/iteration filename mapping mismatch')
    budget = manifest['budget']
    if (budget['train_samples'], budget['train_iters'], budget['warmup_iters'], budget['decay_iters']) != (468736,3662,37,366):
        raise RuntimeError('full sampler / WSD budget changed')
    transport = DirectTransport([Expectation(x['path'], x['sha256'], tuple(x['stat'])) for x in manifest['shards']])
    return manifest, transport


def activate(native_module, manifest_path, expected_sha256, *, opt_in=False):
    if opt_in is not True:
        raise RuntimeError('explicit directio opt-in required')
    manifest, transport = load_transport(manifest_path, expected_sha256)
    install(native_module, manifest['native_contract'], transport)
    return {'transport':'O_DIRECT', 'manifest_sha256':expected_sha256,
            'shards':len(transport.entries), 'native_codec_unchanged':True}
