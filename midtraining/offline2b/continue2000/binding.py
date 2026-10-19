"""Explicit ancestry, never recompute the checkpoint's old identity."""
import json

def adapter_class(base,binding,broker,ancestry_sha,extended_sha):
    if binding['dynamic_manifest_sha256']!=ancestry_sha:raise ValueError('checkpoint ancestry mismatch')
    class ContinuedForward(base):
        def validate_resume(self,runtime):
            # Exactly two dynamic extension fields are separated from native
            # immutable sampler/objective fields. All native fields compare.
            original={k:v for k,v in binding.items() if k not in ('dynamic_state','dynamic_manifest_sha256')}
            expected=base._binding(self,runtime,runtime.iteration())
            if json.loads(json.dumps(expected))!=original:
                raise ValueError('native full checkpoint binding mismatch; no refresh allowed')
        def _binding(self,runtime,iteration):
            b=base._binding(self,runtime,iteration)
            b.update(dynamic_state=broker.snapshot(),dynamic_manifest_sha256=extended_sha,
                ancestry_dynamic_manifest_sha256=ancestry_sha)
            return b
    return ContinuedForward
