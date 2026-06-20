"""Plot the I.3 controlled-comparison curves from cached CSVs.

Reads the ``compare_<label>_history.csv`` files written by
``export_comparison_runs.py`` and emits two figures to ``data-and-figs/``:

* ``comparison_reward_kl`` -- a side-by-side pair: (left) held-out eval reward
  :math:`\\bar{r}` for the two default-reward runs (K=2 vs K=8) and (right)
  reference-KL for all three runs. Reward is comparable only within a fixed
  reward function, so the new-reward run is judged on KL and accuracy; KL is
  reward-independent so all three overlay.
* ``diagnostic_response_length`` -- a standalone full-width figure of held-out
  eval completion length for all three runs, the response-length GRPO
  failure-mode diagnostic (see the I.3 write-up for why this, and not an
  advantage-variance curve, is the diagnostic that is reconstructible from the
  logged scalars).

The reward and length series use the held-out eval columns (``eval_*``, logged
every 64 steps) and are drawn as plain markerless lines. KL has no meaningful
held-out counterpart here, so it stays on the dense per-step train series, drawn
as a bold 64-step centred moving average on a symlog axis (the raw per-step
trace is omitted, and symlog rather than log handles the zero/negative KL during
the K=2 collapse).
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

# Per-panel specs, reused by both the combined figure and the standalone diagnostic.
PANELS = {
    "reward": {
        "column": "eval_reward_mean",
        "runs": ["k2", "k8"],
        "ylabel": r"$\bar{r}$ (held-out eval)",
        "smooth": False,
    },
    "kl": {
        "column": "kl",
        "runs": ["k2", "k8", "k8_newreward"],
        "ylabel": r"$\mathrm{KL}(\pi_\theta \,\|\, \pi_{\mathrm{ref}})$",
        "smooth": True,
    },
    "diagnostic_response_length": {
        "column": "eval_mean_length",
        "runs": ["k2", "k8", "k8_newreward"],
        "ylabel": "Mean eval completion length (tokens)",
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


def draw_panel(ax: plt.Axes, cfg: dict, data_dir: Path, ma_window: int) -> None:
    """Overlay every run in ``cfg`` onto a single axes."""
    for label in cfg["runs"]:
        stem, legend, colour = RUNS[label]
        steps, values = load_series(data_dir / f"{stem}.csv", cfg["column"])
        if len(steps) == 0:
            print(f"  warning: no {cfg['column']} data for {label}; skipping")
            continue
        if cfg.get("smooth"):
            # Dense per-step series: plot only the 64-step moving average. The raw
            # trace is omitted to declutter the (symlog) KL panel.
            ax.plot(steps, centred_moving_average(values, ma_window), linewidth=1.8,
                    color=colour, label=legend)
        else:
            ax.plot(steps, values, linewidth=1.6, color=colour, label=legend)
    ax.set_xlabel("GRPO step")
    ax.set_ylabel(cfg["ylabel"])
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)


def make_combined_reward_kl(data_dir: Path, fig_dir: Path, ma_window: int) -> None:
    """Side-by-side reward (left) and reference-KL (right) overlays."""
    fig, (ax_r, ax_k) = plt.subplots(1, 2, figsize=(12, 4))
    draw_panel(ax_r, PANELS["reward"], data_dir, ma_window)
    draw_panel(ax_k, PANELS["kl"], data_dir, ma_window)
    # KL spans ~0.2 (bounded K=8) to ~10 (K=2 collapse spike); symlog spreads the
    # bounded band while still showing the spike. The linear region below linthresh
    # absorbs the K=2 collapse zeros and the small negative KL estimates.
    ax_k.set_yscale("symlog", linthresh=0.05)
    fig.tight_layout()
    out = fig_dir / "comparison_reward_kl"
    fig.savefig(out.with_suffix(".png"), dpi=200)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"wrote {out}.pdf / .png")


def make_standalone(name: str, data_dir: Path, fig_dir: Path, ma_window: int) -> None:
    """Single full-width figure for one panel spec (the response-length diagnostic)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    draw_panel(ax, PANELS[name], data_dir, ma_window)
    fig.tight_layout()
    out = fig_dir / name
    fig.savefig(out.with_suffix(".png"), dpi=200)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"wrote {out}.pdf / .png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="dir of cached CSVs")
    parser.add_argument("--fig-dir", type=Path, default=FIG_DIR, help="dir to write figures")
    parser.add_argument("--ma-window", type=int, default=DEFAULT_MA)
    args = parser.parse_args()

    make_combined_reward_kl(args.data_dir, args.fig_dir, args.ma_window)
    make_standalone("diagnostic_response_length", args.data_dir, args.fig_dir, args.ma_window)


if __name__ == "__main__":
    main()
