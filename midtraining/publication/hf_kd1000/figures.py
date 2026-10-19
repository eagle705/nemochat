"""Render supplemental Hub figures from the immutable accepted KD trajectory."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil


def render(root, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=False)
    run = root / 'results/midtraining/offline2b-prefixtrain-1000-01'
    ranks = [[json.loads(x) for x in (run / f'student-rank{i}-updates.jsonl').read_text().splitlines()] for i in range(8)]
    for rows in ranks:
        assert len(rows) == 1000 and [x['update'] for x in rows] == list(range(1, 1001))
        assert all(not x['skipped'] and np.isfinite(x['grad_norm']) and x['grad_norm'] > 0 for x in rows)
        assert [x['physical_iteration'] for x in rows] == [x['physical_iteration'] for x in ranks[0]]
        assert [x['lr_used'] for x in rows] == [x['lr_used'] for x in ranks[0]]
    summary = json.loads((run / 'charts/final-20261018-214512/summary.json').read_text())
    source = run / 'charts/final-20261018-214512/lr-loss-progress.png'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == summary['figure_sha256']
    shutil.copyfile(source, out / 'lr_loss.png')
    steps = np.arange(1, 1001)
    norm = np.array([x['grad_norm'] for x in ranks[0]])
    seconds = np.array([max(rows[i]['seconds'] for rows in ranks) for i in range(1000)])
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True)
    axes[0].plot(steps, norm, color='#7651a8', linewidth=1, label='Recorded pre-clip Grad Norm')
    axes[0].axhline(1.0, linestyle='--', color='#777777', label='Native clip threshold = 1.0')
    axes[0].set_ylabel('Grad Norm'); axes[0].legend()
    axes[1].plot(steps, seconds, color='#378986', linewidth=.7, alpha=.8)
    axes[1].set_ylabel('Callback window seconds (MAX over 8 ranks)')
    axes[1].set_xlabel('Accepted native optimizer call (call 1 uses LR = 0)')
    axes[1].text(.01, .98, 'Not end-to-end step time: excludes save / eval / post-callback preflight.', transform=axes[1].transAxes, va='top', fontsize=9)
    for ax in axes: ax.grid(alpha=.18)
    fig.suptitle('KD mid-training: observed gradient and callback timing diagnostics')
    fig.savefig(out / 'gradient_timing.png', dpi=150); plt.close(fig)
    pattern = r'validation loss at iteration\s+(\d+)( on validation set)?\s*\|\s*lm loss value:\s*([0-9.eE+-]+)'
    validation = [dict(step=int(step), phase='end_of_training' if end else 'interval', lm_loss=float(loss)) for step, end, loss in re.findall(pattern, (run / 'student-engine.log').read_text(errors='replace'))]
    assert len(validation) == 5
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    for phase, label, marker, color in [('interval', 'Periodic native validation', 'o', '#2468a2'), ('end_of_training', 'Separate end-of-training validation', '*', '#e18b24')]:
        rows = [x for x in validation if x['phase'] == phase]
        ax.scatter([x['step'] for x in rows], [x['lm_loss'] for x in rows], label=label, marker=marker, color=color, s=95)
        for row in rows: ax.annotate(f"{row['lm_loss']:.4f}", (row['step'], row['lm_loss']), xytext=(-8, 10), textcoords='offset points', ha='right')
    ax.set(xlabel='Native optimizer call', ylabel='Native logged Validation LM Loss', xlim=(190, 1100), ylim=(.82, 1.54))
    ax.set_title('Validation snapshots: changing iterator, NOT fixed matched-evaluation results')
    ax.grid(alpha=.18); ax.legend(loc='lower left')
    fig.savefig(out / 'validation_snapshots.png', dpi=150); plt.close(fig)
    names = ['lr_loss.png', 'gradient_timing.png', 'validation_snapshots.png']
    receipt = dict(calls=1000, files={n:dict(bytes=(out/n).stat().st_size, sha256=hashlib.sha256((out/n).read_bytes()).hexdigest()) for n in names}, validation_observations=validation, source_run='offline2b-prefixtrain-1000-01', model_payload_bytes_hashed=False)
    (out / 'figure_manifest.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/raid/yoon/nemochat'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(render(args.root, args.out), indent=2))
