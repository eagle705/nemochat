"""Frozen task-local control; corpus is real native indexed producer output."""
import numpy as np

KEYS = ('code_concepts','algorithmic','formal_logic','climbmix_replay')
REVISION = '4ba52e8ffa79e9693936c1a819ac89f551ceeef6'
STUDENT = 'checkpoints/exports/nemochat_baby_base_hf_bf16_02'
TEACHER = f'checkpoints/teachers/nemotron35-super-base/{REVISION}-text'

def contract():
    return dict(optimizer_calls=32, sequence_length=512, micro_batch_size=1,
                global_batch_size=1, lr=1e-5, lr_warmup_iters=0, clip_grad=1.,
                train_windows=8, heldout_windows=4, repeats=4, seed=43,
                alpha=.5, top_k=32, checkpoint_integrity='not_checked_explicit_user_waiver',
                student=STUDENT,teacher=TEACHER,teacher_revision=REVISION,
                objective='megatron-lm-ghost-tail-fixed-alpha',
                parallel=dict(tensor_model_parallel_size=1,pipeline_model_parallel_size=1,
                              context_parallel_size=1,expert_model_parallel_size=1,
                              expert_tensor_parallel_size=1,sequence_parallel=False))

def exposure_order():
    return list(range(8))*4

def select_windows(train, validation, *, source):
    if len(train)<1025 or len(validation)<513:
        raise ValueError('insufficient real source window capacity: '+source)
    a = np.stack([np.asarray(train[i*512:i*512+513], dtype=np.int32) for i in range(2)])
    b = np.stack([np.asarray(validation[:513], dtype=np.int32)])
    for v in (a,b):
        if v.min()<0 or v.max()>=131072 or np.any((v>=18)&(v<=29)):
            raise ValueError('invalid or ambiguous reserved IDs in new windows: '+source)
    return a,b

def verify_producer_binding(path,data,teacher):
    import hashlib
    if hashlib.sha256(path.read_bytes()).hexdigest()!=data['windows_sha256']:
        raise ValueError('actual window artifact differs from immutable producer manifest')
    if teacher.get('contract')!=contract() or teacher.get('source_data')!=data or teacher.get('iterations')!=32:
        raise ValueError('teacher/cache/window producer binding changed')

def accept_teacher_reuse(receipt,engine):
    if receipt.get('iterations')!=32 or receipt.get('contract')!=contract() or engine['state']['ExitCode']!=0 or engine['state']['Running']:
        raise ValueError('cannot reuse incomplete/failed teacher producer')
