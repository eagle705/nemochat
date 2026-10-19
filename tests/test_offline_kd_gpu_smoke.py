"""Bounded operational-smoke controls; not GPU evidence by themselves."""
import importlib.util
from pathlib import Path
import numpy as np

PATH = Path(__file__).resolve().parents[1]/'midtraining/patches/offline_kd_pr/gpu_smoke/control.py'

def load():
    assert PATH.is_file(), 'GPU smoke control implementation is missing'
    spec = importlib.util.spec_from_file_location('smoke_control', PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def test_real_windows_are_shifted_disjoint_and_explicitly_repeated():
    m = load()
    values = np.arange(2200, dtype=np.int32) + 100
    train, held = m.select_windows(values, values+10000, source='real-source')
    assert train.shape == (2, 513)
    assert held.shape == (1, 513)
    assert train[1,0] == values[512]
    assert not set(train.ravel()).intersection(held.ravel())
    assert m.exposure_order() == list(range(8))*4
    assert m.contract()['optimizer_calls'] == 32
    assert m.contract()['lr_warmup_iters'] == 0

def test_supervisor_uses_registered_runc_with_gpu_device_request():
    path=PATH.with_name('supervisor.py')
    spec=importlib.util.spec_from_file_location('smoke_supervisor',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert getattr(module,'RUNTIME',None)=='runc', 'Docker has no registered nvidia runtime; use NVIDIA DeviceRequests on runc'

def test_native_student_stage_exists_and_is_bridge_not_private_loop():
    import ast
    tree=ast.parse(PATH.with_name('native.py').read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='student')
    calls=[n for n in ast.walk(node) if isinstance(n,ast.Call)]
    assert any(isinstance(n.func,ast.Name) and n.func.id=='pretrain' for n in calls), 'student must call native Bridge pretrain'
    assert not any(isinstance(n.func,ast.Attribute) and n.func.attr in ('backward','step') for n in calls)

def test_container_pythonpath_can_import_project_tokenizer_gate(monkeypatch):
    import pytest
    spec=importlib.util.spec_from_file_location('smoke_supervisor',PATH.with_name('supervisor.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    captured={}
    def create(method,path,data=None):
        captured.update(data)
        raise RuntimeError('payload captured before any Engine side effect')
    monkeypatch.setattr(module.engine,'request',create)
    with pytest.raises(RuntimeError,match='payload captured'):
        module.run('teacher',Path('/owned/scratch'))
    env=next(v.split('=',1)[1] for v in captured['Env'] if v.startswith('PYTHONPATH='))
    assert '/repo/src' in env.split(':'), 'native_kd tokenizer alias gate imports nemochat from repo src'
    assert 'PYTORCH_ALLOC_CONF=expandable_segments:True' in captured['Env'], '241GB teacher needs non-fragmenting native allocator; no model substitution'

def test_producer_binding_rejects_changed_windows(tmp_path):
    import hashlib
    import pytest
    m=load()
    path=tmp_path/'windows.npz';path.write_bytes(b'actual-smoke-window-artifact')
    data={'windows_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    teacher={'contract':m.contract(),'source_data':data,'iterations':32}
    assert hasattr(m,'verify_producer_binding'), 'reader must authenticate its actual window artifact to the teacher manifest'
    m.verify_producer_binding(path,data,teacher)
    path.write_bytes(b'changed')
    with pytest.raises(ValueError,match='window'):
        m.verify_producer_binding(path,data,teacher)

def test_runtime_blend_has_native_prefixes_weights_and_json_stability():
    import ast,json
    tree=ast.parse(PATH.with_name('native.py').read_text())
    runtime=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='runtime')
    call=next(n for n in ast.walk(runtime) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='CoreRuntime')
    blend=ast.literal_eval(next(k.value for k in call.keywords if k.arg=='blend'))
    assert len(blend)==2,'native Core blend must contain prefixes and weights'
    prefixes,weights=blend
    assert len(prefixes)==len(weights)==1
    assert json.loads(json.dumps(blend))==blend,'checkpoint-sidecar JSON must preserve exact binding'

def test_deadline_cleanup_rejects_foreign_container():
    import pytest
    path=PATH.with_name('deadline.py')
    assert path.exists(),'durable deadline must stop only exact task-owned Engine IDs'
    spec=importlib.util.spec_from_file_location('smoke_deadline',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    actual={'Id':'owned-id','Name':'/nemochat-offline-kd-smoke-01-teacher-1',
            'Config':{'Labels':{'nemochat.offline-kd-smoke.owner':'offline-kd-smoke-01'}},
            'HostConfig':{'DeviceRequests':[{'DeviceIDs':['0']}]}}
    module.verify_owned('owned-id',actual)
    with pytest.raises(ValueError):module.verify_owned('foreign-id',actual)
    actual['Config']['Labels']['nemochat.offline-kd-smoke.owner']='another-job'
    with pytest.raises(ValueError):module.verify_owned('owned-id',actual)

def test_native_adam_kernel_batching_preserves_arguments_and_order():
    path=PATH.with_name('optimizer_compat.py')
    assert path.exists(),'EP1 native TE Adam exceeds its simultaneous tensor-handle capacity'
    spec=importlib.util.spec_from_file_location('smoke_opt',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    calls=[]
    def kernel(chunk,noop,lists,*args,**kwargs):
        calls.append((chunk,noop,lists,args,kwargs))
    wrapped=module.bounded_kernel(kernel,limit=3)
    data=[list(range(8)),list(range(100,108))]
    wrapped(65536,'flag',data,.001,.9,step=7)
    assert [len(c[2][0]) for c in calls]==[3,3,2]
    assert [v for c in calls for v in c[2][0]]==data[0]
    assert all(c[0]==65536 and c[1]=='flag' and c[3]==(.001,.9) and c[4]=={'step':7} for c in calls)

def test_reuse_requires_completed_teacher_and_clean_engine_exit():
    import pytest
    m=load()
    assert hasattr(m,'accept_teacher_reuse'),'retry must preserve accepted teacher cache, not regenerate or substitute it'
    receipt={'iterations':32,'contract':m.contract()}
    m.accept_teacher_reuse(receipt,{'state':{'ExitCode':0,'Running':False}})
    with pytest.raises(ValueError):m.accept_teacher_reuse(receipt,{'state':{'ExitCode':1,'Running':False}})

def test_teacher_native_writer_zero_output_is_not_labeled_lm_loss():
    import ast
    tree=ast.parse(PATH.with_name('native.py').read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='teacher')
    assert not any(isinstance(n,ast.keyword) and n.arg=='teacher_lm_loss' for n in ast.walk(node)), 'Core saver disables teacher CE: label its zero output as a placeholder'
