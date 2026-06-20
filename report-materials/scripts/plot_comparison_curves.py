"""Plot the I.3 controlled-comparison curves from cached CSVs.

Reads the ``compare_<label>_history.csv`` files written by
``export_comparison_runs.py`` and emits three figures to ``report_assets/``:

* ``comparison_mean_reward`` -- held-out eval reward for K=2 vs K=8. Only the two
  default-reward runs are overlaid: reward is comparable only within a fixed
  reward function, so the new-reward run is compared via KL and accuracy.
* ``comparison_kl`` -- reference-KL for all three runs (KL is reward-independent).
* ``diagnostic_response_length`` -- held-out eval completion length for all three
  runs, the response-length GRPO failure-mode diagnostic.

The reward and length figures use the held-out eval series (``eval_*`` columns,
logged every 64 steps); these are plotted as plain markerless lines. KL has no
meaningful held-out counterpart here, so it stays on the dense per-step train
series, drawn as a faint raw trace plus a bold 64-step centred moving average.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Figures live in data-and-figs/; their cached CSVs live in data-and-figs/data/.
FIG_DIR = Path(__file__).resolve().parent.parent / "data-and-figs"
DATA_DIR = FIG_DIR / "data"
DEFAULT_MA = 64

# label -> (csv stem, legend label, colour). Colours are stable across figures.
RUNS = {
    "k2": ("compare_k2_history", "K=2 (default reward)", "tab:red"),
    "k8": ("compare_k8_history", "K=8 (default reward)", "tab:blue"),
    "k8_newreward": ("compare_k8_newreward_history", "K=8 (new reward)", "tab:green"),
}

FIGURES = {
    "comparison_mean_reward": {
        "column": "eval_reward_mean",
        "runs": ["k2", "k8"],
        "ylabel": "Held-out eval reward",
        "title": "Held-out eval reward (default-reward runs)",
        "smooth": False,
    },
    "comparison_kl": {
        "column": "kl",
        "runs": ["k2", "k8", "k8_newreward"],
        "ylabel": r"$\mathrm{KL}(\pi_\theta \,\|\, \pi_{\mathrm{ref}})$",
        "title": "Reference-KL divergence",
        "smooth": True,
    },
    "diagnostic_response_length": {
        "column": "eval_mean_length",
        "runs": ["k2", "k8", "k8_newreward"],
        "ylabel": "Mean eval completion length (tokens)",
        "title": "Response length over training (held-out eval)",
        "smooth": False,
    },
}


def centred_moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Centred moving average with shrinking windows at the edges."""
    v = np.asarray(values, dtype=float)
    kernel = np.ones(window)
    sums = np.convolve(v, kernel, mode="same")
    counts = np.convolve(np.ones_like(v), kernel, mode="same")
    return sums / counts


def load_series(path: Path, column: str) -> tuple[np.ndarray, np.ndarray]:
    """Read (step, value) for one column, dropping rows where it is blank."""
    steps, values = [], []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            cell = row.get(column, "")
            if cell == "":
                continue
            steps.append(int(row["step"]))
            values.append(float(cell))
    return np.asarray(steps), np.asarray(values)


def make_figure(name: str, cfg: dict, data_dir: Path, fig_dir: Path, ma_window: int) -> None:
    plt.figure(figsize=(7, 4))
    for label in cfg["runs"]:
        stem, legend, colour = RUNS[label]
        steps, values = load_series(data_dir / f"{stem}.csv", cfg["column"])
        if len(steps) == 0:
            print(f"  warning: no {cfg['column']} data for {label}; skipping")
            continue
        if cfg.get("smooth"):
            plt.plot(steps, values, linewidth=0.7, alpha=0.2, color=colour)
            plt.plot(steps, centred_moving_average(values, ma_window), linewidth=1.8,
                     color=colour, label=legend)
        else:
            plt.plot(steps, values, linewidth=1.6, color=colour, label=legend)
    plt.xlabel("GRPO step")
    plt.ylabel(cfg["ylabel"])
    plt.title(cfg["title"])
    plt.legend(loc="best", fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out = fig_dir / name
    plt.savefig(out.with_suffix(".png"), dpi=200)
    plt.savefig(out.with_suffix(".pdf"))
    plt.close()
    print(f"wrote {out}.pdf / .png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="dir of cached CSVs")
    parser.add_argument("--fig-dir", type=Path, default=FIG_DIR, help="dir to write figures")
    parser.add_argument("--ma-window", type=int, default=DEFAULT_MA)
    parser.add_argument("--figures", nargs="*", choices=sorted(FIGURES), default=sorted(FIGURES))
    args = parser.parse_args()

    for name in args.figures:
        make_figure(name, FIGURES[name], args.data_dir, args.fig_dir, args.ma_window)


if __name__ == "__main__":
    main()
