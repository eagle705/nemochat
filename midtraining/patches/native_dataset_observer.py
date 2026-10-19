"""Process-local native GPT registry observer for the pinned Bridge 0.6.1 image.

No installed file edit or replacement dataset builder. The caller first checks
its pinned runtime; the parent's source allowlist separately freezes this file.
Retire when native callbacks expose built dataset indices/startup-eval budgeting.
"""


def install(c, publish, reload_only=False, *, no_repeat_gate,
            evaluation_budget_with_startup, keys, weights):
    import numpy as np
    from megatron.bridge.training.config import GPTDatasetConfig
    from megatron.bridge.data import utils
    original = utils._REGISTRY[GPTDatasetConfig]

    def audited(samples, config):
        requested_samples = list(samples)
        if not reload_only and 'validation' in c:
            samples = evaluation_budget_with_startup(samples, c['validation']['eval_iters'],
                                                     c['validation']['eval_global_batch_size'])
        train, valid, test = original(samples, config)
        if not reload_only:
            if train is None:
                raise ValueError('native train dataset missing')
            guard = no_repeat_gate(train)
            if not hasattr(train, 'dataset_index') or len(getattr(train, 'datasets', [])) != 4:
                raise ValueError('actual four-source native blend index required')
            counts = np.bincount(train.dataset_index[:samples[0]].astype('int64'), minlength=4).tolist()
            if len(counts) != 4 or any(x <= 0 for x in counts) or sum(counts) != samples[0]:
                raise ValueError('native source sample coverage failed')
            first_counts = np.bincount(train.dataset_index[:128].astype('int64'), minlength=4).tolist()
            if len(first_counts) != 4 or any(x <= 0 for x in first_counts):
                raise ValueError('actual first global batch must cover all four sources')
            publish('dataset-built', dict(**guard, source_samples=dict(zip(keys, counts)),
                first_global_batch_source_samples=dict(zip(keys, first_counts)),
                source_sample_scope='native planned horizon; first batch separately audited',
                native_blend=True, fixed_sequence_exposure_weights=list(weights),
                validation_native_budget_samples=requested_samples[1],
                validation_actual_reserved_samples=samples[1],
                validation_repetition='explicit native fixed heldout pool across evaluation passes'))
        return train, valid, test

    utils._REGISTRY[GPTDatasetConfig] = audited

    def restore():
        utils._REGISTRY[GPTDatasetConfig] = original
    return restore
