"""CPU plotting of actual prediction records, with explicit excluded cells."""
import io
from pathlib import Path
from .io import FreshOutputs
from .niah import score_predictions


def plot_predictions(rows, output_prefix, *, outputs=None):
    if outputs is None:
        paths = [str(output_prefix) + suffix for suffix in (".png", ".svg", ".metadata.json")]
        with FreshOutputs(paths) as reserved:
            return _plot_predictions(rows, output_prefix, reserved)
    return _plot_predictions(rows, output_prefix, outputs)


def _plot_predictions(rows, output_prefix, outputs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import Normalize

    cells = score_predictions(rows)
    lengths = sorted({cell["context_length"] for cell in cells})
    depths = sorted({cell["depth"] for cell in cells})
    matrix = np.full((len(depths), len(lengths)), np.nan)
    by_key = {(cell["context_length"], cell["depth"]): cell for cell in cells}
    for (length, depth), cell in by_key.items():
        if cell["accuracy"] is not None:
            matrix[depths.index(depth), lengths.index(length)] = cell["accuracy"]
    fig, ax = plt.subplots(figsize=(max(6, 2 * len(lengths)), max(4, 1.2 * len(depths))))
    cmap = plt.get_cmap("viridis").with_extremes(bad="#dddddd")
    image = ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap, norm=Normalize(0, 1), aspect="auto")
    fig.colorbar(image, ax=ax, label="Exact-match accuracy (valid responses only)")
    colors = {"unsupported": "#f4ba72", "timeout": "#8fcce8", "fixture": "#d4b7ef",
              "tokenization_mismatch": "#ee9d9d", "tokenization_unverified": "#ee9d9d", "error": "#ee9d9d"}
    for y, depth in enumerate(depths):
        for x, length in enumerate(lengths):
            cell = by_key.get((length, depth))
            if cell is None:
                text = "not sampled"
            elif cell["count"]:
                text = (f"{cell['accuracy']:.2f}\nn={cell['count']}/{cell['total']}\n"
                        f"95% [{cell['wilson_low']:.2f}, {cell['wilson_high']:.2f}]")
                if cell["excluded"]:
                    text += "\n" + ", ".join(f"{s}:{n}" for s, n in cell["statuses"].items() if s != "ok")
            else:
                statuses = cell["statuses"]
                status = next(iter(statuses)) if len(statuses) == 1 else "mixed exclusions"
                ax.add_patch(plt.Rectangle((x - .5, y - .5), 1, 1,
                                          color=colors.get(status, "#bbbbbb")))
                text = "\n".join(f"{s}: {n}" for s, n in statuses.items()) + "\nn=0"
            ax.text(x, y, text, ha="center", va="center", fontsize=8,
                    color="white" if cell and cell["accuracy"] is not None and cell["accuracy"] < .5 else "black")
    kind = rows[0]["data_kind"]
    title = "SYNTHETIC FIXTURE — NOT MODEL PERFORMANCE" if kind == "fixture" else "Measured NIAH completions"
    if kind == "measured":
        title += f"\n{rows[0].get('model', 'unknown model')}"
    ax.set_title(title)
    ax.set_xticks(range(len(lengths)), [str(n) for n in lengths])
    ax.set_yticks(range(len(depths)), [f"{d:.0%}" for d in depths])
    ax.set_xlabel("Full context budget (prompt + generation reservation)")
    ax.set_ylabel("Needle depth within haystack tokens")
    fig.tight_layout()
    prefix = Path(output_prefix)
    paths = []
    try:
        for extension in ("png", "svg"):
            buffer = io.BytesIO()
            fig.savefig(buffer, format=extension, dpi=150)
            path = str(prefix) + "." + extension
            outputs.write(path, buffer.getvalue())
            paths.append(path)
    finally:
        plt.close(fig)
    metadata = {"data_kind": kind, "measured": kind == "measured", "synthetic_fixture": kind == "fixture",
                "model": rows[0].get("model"), "endpoint": rows[0].get("endpoint"),
                "scale": [0, 1], "scoring": "exact match; whitespace normalization only",
                "confidence": "95% Wilson interval; valid responses only",
                "warning": "Excluded/unsupported/timeout results are not zeros. Fixtures are not performance data.",
                "cells": cells}
    outputs.write_json(str(prefix) + ".metadata.json", metadata)
    return paths
