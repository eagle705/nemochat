import unittest
from types import SimpleNamespace as S
import contract as c

class Resume(unittest.TestCase):
 def test_real_resume_fixture_and_wrong_state_fatal(self):
  scheduler=S(num_steps=115840)
  optimizer=S(param_groups=[{'lr':8.829268292682928e-6}])
  state=S(step=905,consumed_train_samples=115840)
  self.assertEqual(c.validate_restored(state,scheduler,optimizer,{'loaded':True,'step':905,'optimizer':True,'rng':True,'scheduler':True}),905)
  for field,value in [('step',0),('consumed_train_samples',0)]:
   bad=S(**vars(state));setattr(bad,field,value)
   with self.assertRaises(RuntimeError):c.validate_restored(bad,scheduler,optimizer,{'loaded':True,'step':905,'optimizer':True,'rng':True,'scheduler':True})
  with self.assertRaises(RuntimeError):c.validate_restored(state,scheduler,optimizer,{})

 def test_budget_is_cumulative_and_deadline(self):
  b=c.budget(c.TRAIN_STOP-3900)
  self.assertGreater(b['execution_update_cap'],905)
  self.assertLessEqual((b['execution_update_cap']-905)*b['conservative_iteration_seconds']+300,3900)
  with self.assertRaises(RuntimeError):c.budget(c.TRAIN_STOP-300)
  self.assertTrue(c.stop_due(c.TRAIN_STOP-10,18))
  self.assertFalse(c.stop_due(c.TRAIN_STOP-30,18))

 def test_capped_clean_exit_needs_save_not_heldout(self):
  self.assertEqual(c.termination(0,True,10,[915],915),'native_execution_cap_saved')
  self.assertEqual(c.termination(0,True,10,[915],1000),'native_deadline_saved')
  self.assertEqual(c.termination(0,True,10,[],915),'unexpected_exit')
  self.assertEqual(c.termination(1,True,10,[915],915),'unexpected_exit')

 def test_accept_native_inventory_without_common_pt_and_no_heldout(self):
  import tempfile,json
  from pathlib import Path
  from unittest.mock import patch
  import controller
  with tempfile.TemporaryDirectory() as temp:
   out=Path(temp);d=out/'main';d.mkdir()
   ck=d/'checkpoint/iter_0000906';ck.mkdir(parents=True)
   for name in ('train_state.pt','run_config.yaml','.metadata','metadata.json',*[f'__{r}_0.distcp' for r in range(8)]):(ck/name).write_bytes(b'CPU_FIXTURE_NOT_CHECKPOINT')
   for rank in range(8):
    (d/f'rank-{rank}-updates.jsonl').write_text(json.dumps(dict(update=906,segment_update=1,native_optimizer=True,finite_positive_gradient=True,loss=2.,grad_norm=1.,optimizer_step_param_group_lrs=[1e-5]))+'\n')
    proof=dict(loaded=True,optimizer=True,scheduler=True,rng=True)
    controller.write(d/f'rank-{rank}-load.json',dict(source=c.CHECKPOINT,stage_step=905,optimizer_scheduler_rng_cursor_reset=False,actual_native_restore_provenance=proof))
    controller.write(d/f'rank-{rank}-save.json',dict(save_steps=[906],save_optim=True,save_rng=True,native_synchronous_save_completed=True))
    controller.write(d/f'rank-{rank}-entrypoint.json',dict(native_exit_code=0,updates=1,termination_reason='native_execution_cap_saved'))
   with patch.object(controller,'OUT',out):
    result=controller.accept(dict(Running=False,ExitCode=0,OOMKilled=False,Error=''),dict(execution_update_cap=906))
    self.assertEqual(result['final_step'],906)

 def test_config_requires_strict_native_fullstate(self):
  import native
  cfg=S(train=S(train_iters=34135),checkpoint=S(),scheduler=S())
  def finalize():
   self.assertFalse(cfg.checkpoint.load_optim and cfg.checkpoint.load_main_params_from_ckpt)
  cfg.checkpoint.finalize=finalize
  native.runtime_config(cfg,{'updates':34135,'execution_update_cap':1400})
  self.assertFalse(cfg.checkpoint.load_main_params_from_ckpt)
  self.assertEqual(cfg.checkpoint.dist_ckpt_strictness,'raise_all')
  self.assertEqual(cfg.checkpoint.load,c.CHECKPOINT)
  self.assertTrue(cfg.checkpoint.load_optim and cfg.checkpoint.load_rng and cfg._checkpoint_load_required)
  self.assertFalse(cfg.checkpoint.finetune or cfg.scheduler.override_opt_param_scheduler)
  self.assertTrue(cfg.scheduler.use_checkpoint_opt_param_scheduler)

 def test_deadline_saves_committed_cursor_before_forward(self):
  import resume_sft
  events=[];state=S(train_state=S(step=907),floating_point_operations_so_far=1.)
  state.train_state.floating_point_operations_so_far=1.
  ctx=S(state=state,model=['CPU_FIXTURE'],optimizer='optimizer',scheduler='scheduler')
  with self.assertRaises(SystemExit) as exit:
   resume_sft.deadline_exit(ctx,[], 'native_manager',lambda *args:events.append(('save',args[0].train_state.step,args[5])),lambda context:events.append(('ack',context.state.train_state.step)))
  self.assertEqual(exit.exception.code,0)
  self.assertEqual(events,[('save',907,'native_manager'),('ack',907)])
  events=[]
  with self.assertRaises(SystemExit):resume_sft.deadline_exit(ctx,[907],'native_manager',lambda *a:events.append('BAD_DUPLICATE_SAVE'),lambda *a:events.append('BAD_ACK'))
  self.assertEqual(events,[])

if __name__=='__main__':unittest.main()
