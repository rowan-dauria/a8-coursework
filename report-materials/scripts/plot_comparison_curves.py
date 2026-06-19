"""Plot the I.3 controlled-comparison training curves from cached CSVs.

Reads the ``compare_<label>_history.csv`` files written by
``export_comparison_runs.py`` and emits three figures to ``report_assets/``:

* ``comparison_mean_reward`` -- mean reward for K=2 vs K=8. Only the two
  default-reward runs are overlaid: mean reward is comparable only within a
  fixed reward function, so the new-reward run is compared via KL and accuracy.
* ``comparison_kl`` -- reference-KL for all three runs (KL is reward-independent).
* ``diagnostic_response_length`` -- mean completion length for all three runs,
  the response-length GRPO failure-mode diagnostic.

Each curve is drawn as a faint raw per-step trace plus a bold 64-step centred
moving average, mirroring ``plot_baseline_curves.py`` (the data is noisy).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DEFAULT_DIR = Path(__file__).resolve().parent.parent / "report_assets"
DEFAULT_MA = 64

# label -> (csv stem, legend label, colour). Colours are stable across figures.
RUNS = {
    "k2": ("compare_k2_history", "K=2 (default reward)", "tab:red"),
    "k8": ("compare_k8_history", "K=8 (default reward)", "tab:blue"),
    "k8_newreward": ("compare_k8_newreward_history", "K=8 (new reward)", "tab:green"),
}

FIGURES = {
    "comparison_mean_reward": {
        "column": "reward_mean",
        "runs": ["k2", "k8"],
        "ylabel": "Mean reward",
        "title": "Mean reward (default-reward runs)",
    },
    "comparison_kl": {
        "column": "kl",
        "runs": ["k2", "k8", "k8_newreward"],
        "ylabel": r"$\mathrm{KL}(\pi_\theta \,\|\, \pi_{\mathrm{ref}})$",
        "title": "Reference-KL divergence",
    },
    "diagnostic_response_length": {
        "column": "mean_length",
        "runs": ["k2", "k8", "k8_newreward"],
        "ylabel": "Mean completion length (tokens)",
        "title": "Response length over training",
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


def make_figure(name: str, cfg: dict, data_dir: Path, ma_window: int) -> None:
    plt.figure(figsize=(7, 4))
    for label in cfg["runs"]:
        stem, legend, colour = RUNS[label]
        steps, values = load_series(data_dir / f"{stem}.csv", cfg["column"])
        if len(steps) == 0:
            print(f"  warning: no {cfg['column']} data for {label}; skipping")
            continue
        plt.plot(steps, values, linewidth=0.7, alpha=0.2, color=colour)
        plt.plot(steps, centred_moving_average(values, ma_window), linewidth=1.8,
                 color=colour, label=legend)
    plt.xlabel("GRPO step")
    plt.ylabel(cfg["ylabel"])
    plt.title(cfg["title"])
    plt.legend(loc="best", fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out = data_dir / name
    plt.savefig(out.with_suffix(".png"), dpi=200)
    plt.savefig(out.with_suffix(".pdf"))
    plt.close()
    print(f"wrote {out}.pdf / .png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--ma-window", type=int, default=DEFAULT_MA)
    parser.add_argument("--figures", nargs="*", choices=sorted(FIGURES), default=sorted(FIGURES))
    args = parser.parse_args()

    for name in args.figures:
        make_figure(name, FIGURES[name], args.data_dir, args.ma_window)


if __name__ == "__main__":
    main()
