"""Regenerate the isolated patch against exact installed Bridge source evidence."""
import difflib
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent

FACTORY = '''


def make_cached_kd_forward_step(config, runtime_factory, *, resume_binding=None):
    """Opt-in native Core cached KD; existing default/ModelOpt paths are unchanged.

    Use the returned callable as pretrain(forward_step_func=...) in a with block.
    Native cache generation instead uses its explicit cache_forward method and
    never executes an optimizer/backward step. See cached_kd.integration.
    """
    from megatron.bridge.training.cached_kd.integration import CachedKDForwardStep

    return CachedKDForwardStep(
        config, runtime_factory, resume_binding=resume_binding,
        common_forward=_forward_step_common,
    )
'''


def main():
    original = (BASE/'evidence/bridge-installed/training__gpt_step.py').read_text()
    target = 'src/megatron/bridge/training/gpt_step.py'
    pieces = [''.join(difflib.unified_diff(original.splitlines(True),
              (original+FACTORY).splitlines(True), fromfile='a/'+target,tofile='b/'+target))]
    for source in sorted((BASE/'bridge_cached_kd').glob('*.py')):
        dest = 'src/megatron/bridge/training/cached_kd/'+source.name
        pieces.append(''.join(difflib.unified_diff([],source.read_text().splitlines(True),
                      fromfile='/dev/null',tofile='b/'+dest)))
    (BASE/'megatron-bridge-0.6.1-cached-kd.patch').write_text(''.join(pieces))
    provenance = dict(bridge_version='0.6.1',
        bridge_base_sha=(BASE/'evidence/bridge-installed/.git__HEAD').read_text().strip(),
        bridge_gpt_step_sha256=hashlib.sha256(original.encode()).hexdigest(),
        image='nvcr.io/nvidia/nemo@sha256:fdd6e9c7929b76c8624ddfea939dda345beaf67695d9c3994994c18f9e79b9bf',
        core_sources=json.loads((BASE/'evidence/installed-core-provenance.json').read_text())['files'],
        note='Installed Bridge base, NOT upstream main; no installed SDK edits. Core modules ported from installed snapshots.')
    (BASE/'evidence/integration-base.json').write_text(json.dumps(provenance,indent=2)+'\n')


if __name__=='__main__':
    main()
