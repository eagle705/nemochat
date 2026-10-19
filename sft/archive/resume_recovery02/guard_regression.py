"""Pinned-image negative controls; no source files or SDK files modified."""
import importlib
from unittest.mock import patch

def prove(parent, control):
    entry=importlib.import_module('megatron.bridge.training.pretrain')
    train=importlib.import_module('megatron.bridge.training.train')
    wrapper=entry.pretrain
    guard=parent.guard_native_sources
    results={}
    def denied(name,call):
        try:call()
        except (PermissionError,RuntimeError):results[name]='DENIED'
        else:raise AssertionError(name+' unexpectedly allowed')
        control.require(entry.pretrain is wrapper,'wrapper not restored after source guard')
    with patch.object(entry,'pretrain',lambda:None):
        try:guard()
        except PermissionError:results['unknown_local_callable']='DENIED'
        else:raise AssertionError('unknown wrapper accepted')
    code=wrapper.__code__
    wrapper.__code__=code.replace(co_name='changed_local_code')
    try:denied('changed_local_code',guard)
    finally:wrapper.__code__=code
    sha=control.sha
    with patch.object(control,'sha',lambda p:'0'*64 if str(p)==str(control.HERE/'native.py') else sha(p)):
        denied('changed_local_source_digest',guard)
    with patch.object(train,'train_step',lambda:None):
        denied('unknown_sdk_callable',guard)
    import hashlib
    real=hashlib.sha256
    class Wrong:
        def hexdigest(self):return '0'*64
    with patch.object(hashlib,'sha256',lambda data=b'':Wrong() if b'def pretrain(' in data else real(data)):
        denied('changed_pinned_sdk_digest',guard)
    receipt=guard()
    control.require(receipt['unchanged_sdk_guard_invoked'] is True,'real SDK guard absent')
    control.write(control.OUT/'config-sdk03/source-guard-regression.json',dict(status='PINNED_ACTUAL_SOURCE_GUARD_REGRESSIONS_PASS',negative_controls=results,positive=receipt,no_gpu=True,no_weights=True))
