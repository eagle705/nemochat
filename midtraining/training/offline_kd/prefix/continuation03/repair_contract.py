"""Sparse teacher repair: original native global cursor, no schedule change."""

def attach_native_writer(saver, model, unwrap_model):
    # Match CachedKDForwardStep._setup: Float16Module is not the native head.
    saver.attach_hooks(unwrap_model(model))

def repair_plan(iterations):
    if not iterations or any(type(i) is not int or not 0 <= i < 1143 for i in iterations):
        raise ValueError('nonempty completed-prefix integer iterations required')
    return [dict(iteration=i, consumed_samples=i*128, microbatches=16)
            for i in sorted(set(iterations))]


def build_overlay(original, repairs, overlay, iterations, stop=1143, repaired_names=None):
    selected = {r['iteration'] for r in repair_plan(iterations)}
    rows = []
    for rank in range(8):
        for iteration in range(stop):
            name = f'cp0_dp{rank}__{iteration}.tar'
            regenerated = iteration in selected and (repaired_names is None or name in repaired_names)
            path = repairs/f'iteration-{iteration}'/name if regenerated else original/name
            if not path.is_file():
                raise FileNotFoundError(path)
            rows.append(dict(name=name, source=str(path.resolve()), regenerated=regenerated))
    overlay.mkdir(exist_ok=False)
    for row in rows:
        (overlay/row['name']).symlink_to(row['source'])
    return rows
