"""Reference explanations and actionable hypotheses, not attribution proof."""
from dashboard.capture_analysis import semantic, _candidates


def test_allgather_explanation_describes_rank_order_not_a_model_layer():
    category, meaning, evidence = semantic('ncclDevKernel_AllGather_RING_LL(args)')
    assert category == 'NCCL'
    assert 'AllGather' in meaning and 'rank' in meaning and '순서' in meaning
    assert '모든' in meaning
    assert evidence == 'symbol_reference'


def test_sendrecv_is_point_to_point_not_asserted_collective_role():
    _, meaning, evidence = semantic('ncclDevKernel_SendRecv(args)')
    assert 'SendRecv' in meaning and '송수신' in meaning
    assert '집합 통신 계열' not in meaning
    assert evidence == 'symbol_reference'


def test_optimizer_tensor_norm_explanation_does_not_claim_gradient_callsite():
    _, meaning, evidence = semantic('multi_tensor_l2norm::L2NormFunctor<float>')
    assert 'L2' in meaning
    assert '미확인' in meaning
    assert evidence == 'symbol_reference'


def test_candidate_actions_depend_on_observed_operator_family():
    def phase(label, name, category):
        return {'id': label, 'host_ms': 10, 'same_process_contained_api_count': 2,
                'kernels': [{'name': name, 'category': category, 'count': 2,
                             'total_ms': 5, 'percent': 50}]}
    cards = _candidates([{'rank': 0, 'update': 3, 'phases': [
        phase('forward', 'ncclDevKernel_AllGather_RING_LL', 'NCCL'),
        phase('backward+sync', 'ncclDevKernel_SendRecv', 'NCCL'),
        phase('optimizer', 'bfloat16_copy_kernel_cuda', 'fill_copy_cast')]}])
    actions = {c['phase']: c['action'] for c in cards}
    assert len(set(actions.values())) == 3
    assert 'rank' in actions['forward']
    assert 'bucket' in actions['forward']
    assert '송수신' in actions['backward+sync']
    assert '복사' in actions['optimizer']
    assert all('A/B' in action for action in actions.values())


def test_config_rejects_unsafe_inspector_and_source_paths_before_collection(tmp_path):
    import pytest
    from dashboard.server import validate_config
    for field in ['nsight_analysis', 'nsight_native_source', 'nsight_source_provenance']:
        with pytest.raises(ValueError):
            validate_config(tmp_path, {'profiling': {field: '../outside.json'}})
        validate_config(tmp_path, {'profiling': {field: 'results/local.json'}})
