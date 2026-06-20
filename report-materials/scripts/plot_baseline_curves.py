"""Plot the baseline (I.1) held-out eval-reward curve.

The baseline is W&B run ``jgs4c6kl`` (K=2, default config, 3364 steps). The
held-out eval reward ``rewards/eval/score/mean`` (logged every 64 steps) is
pulled from W&B and cached to ``baseline_mean_reward_curve.csv`` so the figure
can be regenerated offline; the cached CSV is used when present unless
``--refresh`` is given. The curve is drawn as a plain markerless line.

The baseline reference-KL figure (``baseline_kl_curve``) is the per-step train
KL and is generated separately on the TPU (see
``tpu-2026-fork/report_assets/``); it is not produced here.

    python report-materials/scripts/plot_baseline_curves.py            # cache-first
    python report-materials/scripts/plot_baseline_curves.py --refresh  # force W&B re-pull
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
FIG_DIR = SCRIPT_DIR.parent / "data-and-figs"
DATA_DIR = FIG_DIR / "data"

DEFAULT_ENTITY = "felsomoye-university-of-cambridge"
DEFAULT_PROJECT = "tunix"
BASELINE_RUN = "jgs4c6kl"
EVAL_TAG = "rewards/eval/score/mean"
CSV_NAME = "baseline_mean_reward_curve.csv"


def fetch_wandb(entity: str, project: str, run_id: str) -> list[tuple[int, float]]:
    import wandb

    run = wandb.Api().run(f"{entity}/{project}/runs/{run_id}")
    pts: list[tuple[int, float]] = []
    for row in run.scan_history(keys=["_step", EVAL_TAG], page_size=2000):
        step, value = row.get("_step"), row.get(EVAL_TAG)
        if step is None or value is None:
            continue
        pts.append((int(step), float(value)))
    pts.sort()
    return pts


def write_csv(path: Path, pts: list[tuple[int, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", EVAL_TAG])
        writer.writerows(pts)


def read_csv(path: Path) -> list[tuple[int, float]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # header
        return [(int(s), float(v)) for s, v in reader]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", default=DEFAULT_ENTITY)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--refresh", action="store_true", help="force re-pull from W&B")
    args = parser.parse_args()

    cache = DATA_DIR / CSV_NAME
    if cache.exists() and not args.refresh:
        pts = read_csv(cache)
        source = str(cache)
    else:
        pts = fetch_wandb(args.entity, args.project, BASELINE_RUN)
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        write_csv(cache, pts)
        source = f"W&B {BASELINE_RUN}"

    steps = [s for s, _ in pts]
    values = [v for _, v in pts]

    plt.figure(figsize=(7, 4))
    plt.plot(steps, values, linewidth=1.6, color="tab:blue")
    plt.xlabel("GRPO step")
    plt.ylabel("Held-out eval reward")
    plt.title("Baseline GRPO held-out eval reward")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    out = FIG_DIR / "baseline_mean_reward_curve"
    plt.savefig(out.with_suffix(".png"), dpi=200)
    plt.savefig(out.with_suffix(".pdf"))
    plt.close()
    print(f"wrote {out}.pdf / .png ({len(pts)} points, source: {source})")


if __name__ == "__main__":
    main()
