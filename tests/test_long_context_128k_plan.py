"""Deferred128K stage-plan controls; not nativeCP/runtime acceptance."""
from pathlib import Path
import yaml


def test_deferred_long_stage_is_128k_and_cannot_start_current_training():
    path = Path('sft/configs/nemochat_baby_long_context_128k.plan.yaml')
    assert path.is_file(), 'Missing separately scoped128K long-context stage plan'
    c = yaml.safe_load(path.read_text())
    assert c['schema'] == 'nemochat-deferred-long-context-plan-v1'
    assert c['sequence_length'] == 131072
    assert c['dataset']['id'] == 'nvidia/Nemotron-SFT-Long-Context-Transduction-v1'
    assert c['auto_launch'] is False
    assert c['changes_current_16k_sft'] is False
    assert c['source_checkpoint'] is None
    assert c['native_launch_ready'] is False


def test_long_stage_uses_conservative_cp_and_requires_real_memory_proof():
    c = yaml.safe_load(Path('sft/configs/nemochat_baby_long_context_128k.plan.yaml').read_text())
    p = c['parallelism_plan']
    assert p['initial_context_parallel_size'] == 8
    assert p['candidate_context_parallel_sizes'] == [8, 4, 2]
    assert p['physical_gpu_count'] == 8 and p['micro_batch_size'] == 1
    assert p['minimum_free_memory_fraction_after_training_step'] >= .15
    assert p['native_training_probe_required'] is True
    assert p['unverified_cp_must_not_launch_production'] is True
    assert c['data_policy']['truncate_reference_answers'] is False
    assert c['data_policy']['measure_complete_prompt_plus_reference'] is True
