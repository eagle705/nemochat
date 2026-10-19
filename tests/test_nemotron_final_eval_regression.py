"""Regression for actual GB128 calibration double-final-evaluation failure.

The pinned SDK always performs post-loop validation on normal budget completion.
These are CPU scheduling controls, not native GPU acceptance evidence.
"""
import runpy

NATIVE = runpy.run_path('sft/training/native_nemotron_sft.py')


def test_normal_budget_completion_does_not_consume_an_extra_heldout_window():
    due = NATIVE.get('in_loop_final_evaluation_due')
    assert callable(due), 'Missing SDK-aware in-loop final-evaluation scheduling'
    plan = {'updates': 20, 'max_hours': .25}
    forced = [i for i in range(1, 21) if due(i, 460., plan)]
    # SDK reserves one final window for 20 updates and eval_interval=100.
    assert forced == []
    assert len(forced) + 1 == 1


def test_early_duration_exit_still_forces_final_validation_before_native_exit():
    due = NATIVE.get('in_loop_final_evaluation_due')
    assert callable(due), 'Missing SDK-aware in-loop final-evaluation scheduling'
    plan = {'updates': 1000, 'max_hours': 1 / 60}
    assert due(1, 206., plan)
    assert not due(1, 59., plan)
    # Normal budget exit relies on SDK final validation even if deadline passed.
    assert not due(1000, 206., plan)
