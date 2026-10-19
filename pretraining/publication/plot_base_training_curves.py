"""Render observed BASE pretraining curves; no weights or credentials read."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'results/pretraining/nemochat_baby/training/runs/baby-main-01'
OUT = ROOT / 'results/pretraining/nemochat_baby/hf_card_figures'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    updates = [json.loads(line) for line in (RUN / 'rank-0-updates.jsonl').read_text().splitlines()]
    heldout = [json.loads(line) for line in (RUN / 'rank-0-heldout.jsonl').read_text().splitlines()]
    assert len(updates) == 19074 and updates[-1]['update'] == 19074
    steps = np.array([x['update'] for x in updates])
    tokens = np.array([x['processed_positions'] for x in updates]) / 1e9
    lr = np.array([x['lr'] for x in updates])
    loss = np.array([x['loss'] for x in updates])
    assert np.isfinite(lr).all() and np.isfinite(loss).all()
    boundary = (19074 - 1908) * 524288 / 1e9
    plt.rcParams.update({'font.size': 11, 'figure.dpi': 160})
    for kind in ('learning_rate', 'loss'):
        fig, ax = plt.subplots(figsize=(10, 4.5))
        if kind == 'learning_rate':
            ax.plot(tokens, lr, color='#2563eb', linewidth=1.7, label='Observed Learning Rate')
            ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
            ax.set_ylabel('Learning Rate')
            ax.set_title('NemoChat BASE — observed WSD Learning Rate')
        else:
            ax.plot(tokens, loss, color='#94a3b8', alpha=0.35, linewidth=0.45, label='Training Loss (each update)')
            smooth = np.convolve(loss, np.ones(100)/100, mode='valid')
            ax.plot(tokens[99:], smooth, color='#2563eb', linewidth=1.6, label='Training Loss (trailing 100-update mean)')
            ax.plot([x['update']*524288/1e9 for x in heldout], [x['heldout_ce'] for x in heldout], color='#ea580c', linewidth=1.5, label='Validation CE (fixed heldout window)')
            ax.set_ylabel('Loss / Cross-Entropy')
            ax.set_title('NemoChat BASE — observed pretraining Loss')
        ax.axvspan(boundary, tokens[-1], color='#16a34a', alpha=0.08, label='LR decay phase')
        ax.set_xlabel('Processed next-token positions (billions)')
        ax.set_xlim(0, tokens[-1])
        ax.grid(alpha=0.2)
        ax.legend(fontsize=9)
        fig.tight_layout()
        path = OUT / f'{kind}.png'
        fig.savefig(path)
        plt.close(fig)
        print(path)
    receipt = {'source_run': str(RUN.relative_to(ROOT)), 'updates': len(updates), 'final_update': int(steps[-1]), 'final_processed_positions': updates[-1]['processed_positions'], 'heldout_evaluations': len(heldout), 'loss_smoothing': 'trailing 100-update mean; raw updates also shown', 'validation_scope': 'fixed reused ClimbMix heldout window, not standardized benchmark', 'figures': ['learning_rate.png', 'loss.png']}
    (OUT / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')


if __name__ == '__main__':
    main()
