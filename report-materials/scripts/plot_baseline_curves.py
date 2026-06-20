"""Plot the baseline (I.1) training curves: held-out eval reward and reference-KL.

Both series come from W&B run ``jgs4c6kl`` (K=2, default config, 3364 steps):

* ``rewards/eval/score/mean`` -- held-out eval reward (logged every 64 steps),
  cached to ``baseline_mean_reward_curve.csv``.
* ``actor/train/kl`` -- the dense per-step reference-KL
  :math:`\\mathrm{KL}(\\pi_\\theta \\,\\|\\, \\pi_{\\mathrm{ref}})` over the 3364
  training steps, cached to ``baseline_kl_curve.csv``. W&B keeps logging a
  zero-filled tail past step 3364 after training stops, so we trim to
  ``step <= MAX_STEP``.

The two series are drawn as a single side-by-side figure ``baseline_curves``
(left: r-bar held-out eval reward, plain line; right: reference-KL, faint
per-step trace plus a bold 64-step centred moving average, mirroring the I.3
comparison style). Cached CSVs are used when present unless ``--refresh`` forces
a W&B re-pull, so the figure regenerates offline.

    python report-materials/scripts/plot_baseline_curves.py            # cache-first
    python report-materials/scripts/plot_baseline_curves.py --refresh  # force W&B re-pull
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
FIG_DIR = SCRIPT_DIR.parent / "data-and-figs"
DATA_DIR = FIG_DIR / "data"

DEFAULT_ENTITY = "felsomoye-university-of-cambridge"
DEFAULT_PROJECT = "tunix"
BASELINE_RUN = "jgs4c6kl"
EVAL_TAG = "rewards/eval/score/mean"
KL_TAG = "actor/train/kl"
MAX_STEP = 3364  # baseline trained 3364 steps; W&B logs a zero tail past this.
MA_WINDOW = 64

REWARD_CSV = "baseline_mean_reward_curve.csv"
KL_CSV = "baseline_kl_curve.csv"


def fetch_wandb(entity: str, project: str, run_id: str, tag: str,
                max_step: int | None = None) -> list[tuple[int, float]]:
    import wandb

    run = wandb.Api().run(f"{entity}/{project}/runs/{run_id}")
    pts: list[tuple[int, float]] = []
    for row in run.scan_history(keys=["_step", tag], page_size=2000):
        step, value = row.get("_step"), row.get(tag)
        if step is None or value is None:
            continue
        if max_step is not None and step > max_step:
            continue
        pts.append((int(step), float(value)))
    pts.sort()
    return pts


def write_csv(path: Path, pts: list[tuple[int, float]], value_col: str) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", value_col])
        writer.writerows(pts)


def read_csv(path: Path) -> list[tuple[int, float]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # header
        return [(int(s), float(v)) for s, v in reader]


def load_series(cache: Path, tag: str, args: argparse.Namespace,
                max_step: int | None = None) -> tuple[list[tuple[int, float]], str]:
    """Cache-first load of one tag: read the CSV unless ``--refresh`` is given."""
    if cache.exists() and not args.refresh:
        return read_csv(cache), str(cache)
    pts = fetch_wandb(args.entity, args.project, BASELINE_RUN, tag, max_step)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(cache, pts, tag)
    return pts, f"W&B {BASELINE_RUN} ({tag})"


def centred_moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Centred moving average with shrinking windows at the edges."""
    v = np.asarray(values, dtype=float)
    kernel = np.ones(window)
    sums = np.convolve(v, kernel, mode="same")
    counts = np.convolve(np.ones_like(v), kernel, mode="same")
    return sums / counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", default=DEFAULT_ENTITY)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--refresh", action="store_true", help="force re-pull from W&B")
    args = parser.parse_args()

    reward, rsrc = load_series(DATA_DIR / REWARD_CSV, EVAL_TAG, args)
    kl, ksrc = load_series(DATA_DIR / KL_CSV, KL_TAG, args, max_step=MAX_STEP)

    fig, (ax_r, ax_k) = plt.subplots(1, 2, figsize=(12, 4))

    # Left: held-out eval reward (plain markerless line).
    ax_r.plot([s for s, _ in reward], [v for _, v in reward], linewidth=1.6, color="tab:blue")
    ax_r.set_xlabel("GRPO step")
    ax_r.set_ylabel(r"$\bar{r}$ (held-out eval)")
    ax_r.grid(alpha=0.3)

    # Right: per-step reference-KL (faint raw + bold 64-step centred moving average).
    k_steps = np.array([s for s, _ in kl])
    k_vals = np.array([v for _, v in kl])
    ax_k.plot(k_steps, k_vals, linewidth=0.6, alpha=0.12, color="tab:blue")
    ax_k.plot(k_steps, centred_moving_average(k_vals, MA_WINDOW), linewidth=1.8, color="tab:blue")
    # KL sits near ~0.5 but spikes to ~40 during the late collapse; symlog (linear
    # below 1, log above) keeps the baseline level and the spikes both legible.
    ax_k.set_yscale("symlog", linthresh=1.0)
    ax_k.set_xlabel("GRPO step")
    ax_k.set_ylabel(r"$\mathrm{KL}(\pi_\theta \,\|\, \pi_{\mathrm{ref}})$")
    ax_k.grid(alpha=0.3)

    fig.tight_layout()
    out = FIG_DIR / "baseline_curves"
    fig.savefig(out.with_suffix(".png"), dpi=200)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"wrote {out}.pdf / .png (reward: {len(reward)} pts [{rsrc}]; kl: {len(kl)} pts [{ksrc}])")


if __name__ == "__main__":
    main()
