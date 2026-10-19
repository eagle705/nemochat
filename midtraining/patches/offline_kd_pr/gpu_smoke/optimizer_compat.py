"""Task-local TE Adam kernel launch batching, not an optimizer replacement.

Pinned TE has a fixed 20MiB NVTETensor handle pool (20321 handles). EP1
student parameter groups can exceed this when Adam wraps four tensor lists
at once. Keep Bridge/MCore/TE Adam, clipping, state, and scheduler unchanged;
only divide independent tensor lists into bounded native CUDA launches.
"""

def bounded_kernel(native,limit=1024):
    def call(chunk_size,noop,tensor_lists,*args,**kwargs):
        lengths={len(x) for x in tensor_lists}
        if len(lengths)!=1:raise ValueError('native Adam tensor lists must align')
        size=len(tensor_lists[0])
        for start in range(0,size,limit):
            native(chunk_size,noop,[x[start:start+limit] for x in tensor_lists],*args,**kwargs)
    return call

def install_and_verify():
    import torch
    import transformer_engine.pytorch.optimizers.fused_adam as module
    from transformer_engine.pytorch.optimizers import FusedAdam
    native=module.tex.multi_tensor_adam
    # Real CUDA regression, test fixture only: compare unchanged native Adam
    # against the exact launch-splitting path, including optimizer moments.
    initial=[torch.arange(1,6,device='cuda',dtype=torch.float32)+i for i in range(11)]
    a=[torch.nn.Parameter(x.clone()) for x in initial]
    b=[torch.nn.Parameter(x.clone()) for x in initial]
    baseline=FusedAdam(a,lr=1e-5,weight_decay=.01)
    split=FusedAdam(b,lr=1e-5,weight_decay=.01)
    split.multi_tensor_adam=bounded_kernel(native,limit=3)
    for step in range(3):
        for p,q in zip(a,b):p.grad=torch.ones_like(p)*(step+1);q.grad=p.grad.clone()
        baseline.step();split.step()
    max_diff=max(float((p-q).abs().max()) for p,q in zip(a,b))
    assert max_diff==0.,max_diff
    for p,q in zip(a,b):
        for key in ('exp_avg','exp_avg_sq'):
            assert torch.equal(baseline.state[p][key],split.state[q][key])
    module.tex.multi_tensor_adam=bounded_kernel(native)
    # The failing tensor-count class without large allocations: native TE
    # optimizer now successfully handles 6000 scalar parameters on real CUDA.
    many=[torch.nn.Parameter(torch.ones(1,device='cuda')) for _ in range(6000)]
    opt=FusedAdam(many,lr=1e-5)
    for p in many:p.grad=torch.ones_like(p)
    opt.step()
    assert all(float(p)<1 for p in many)
    del initial,a,b,baseline,split,many,opt
    torch.cuda.empty_cache()
    return dict(installed=True,native_optimizer='transformer_engine.FusedAdam',
                native_kernel='multi_tensor_adam',max_parameters_per_launch=1024,
                real_cuda_small_state_parity=True,max_parameter_abs_difference=max_diff,
                real_cuda_6000_parameter_update=True,student_training_optimizer_calls=0,
                installed_sdk_files_modified=False)
