"""Ordinary B=1.0LM; practical-recipe comparison against C=0.5LM+0.5KD."""
LM_COEFFICIENT=1.0

def ordinary_lm_loss(native_loss, output):
    lm,count,metrics=native_loss(output)
    report={k:v.detach().clone() for k,v in metrics.items()}
    report['total loss']=report['lm loss'].clone()
    return LM_COEFFICIENT*lm,count,report

def forward_step(common,state,iterator,model,return_schedule_plan=False):
    from megatron.bridge.training.gpt_step import _create_loss_function
    output,mask=common(state,iterator,model,return_schedule_plan)
    native=_create_loss_function(mask,
        check_for_nan_in_loss=state.cfg.rerun_state_machine.check_for_nan_in_loss,
        check_for_spiky_loss=state.cfg.rerun_state_machine.check_for_spiky_loss)
    if not model.training:return output,native
    return output,lambda tensor:ordinary_lm_loss(native,tensor)
