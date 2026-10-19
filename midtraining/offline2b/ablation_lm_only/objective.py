"""Coefficient-matched native LM objective, with unweighted LM reporting."""
LM_COEFFICIENT = 0.5

def coefficient_matched_loss(native_loss, output, coefficient=LM_COEFFICIENT):
    if coefficient != 0.5:
        raise ValueError('B must retain the C LM coefficient 0.5')
    lm, count, metrics = native_loss(output)
    total = coefficient * lm
    # Independent report storage: native microbatch normalization may be in-place.
    report = {k: v.detach().clone() for k,v in metrics.items()}
    report['total loss'] = report['lm loss'].clone()
    report['total loss'][0] *= coefficient
    return total, count, report

def forward_step(common, state, iterator, model, return_schedule_plan=False):
    from megatron.bridge.training.gpt_step import _create_loss_function
    output, mask = common(state, iterator, model, return_schedule_plan)
    native = _create_loss_function(mask,
        check_for_nan_in_loss=state.cfg.rerun_state_machine.check_for_nan_in_loss,
        check_for_spiky_loss=state.cfg.rerun_state_machine.check_for_spiky_loss)
    # Native heldout LM remains unweighted, just as in C.
    if not model.training:
        return output, native
    return output, lambda tensor: coefficient_matched_loss(native, tensor)
