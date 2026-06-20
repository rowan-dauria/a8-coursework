"""Plot the appendix group-size (K) sweep held-out eval-reward curves.

For each run in the K = 2, 4, 8, 16 sweep, pull the held-out eval reward series
``rewards/eval/score/mean`` from W&B (logged every 64 steps) and overlay the
raw values for all four K as lines on a single set of axes. No smoothing is
applied: the series is already sparse, so each logged value is a genuine
periodic eval.

Sources mirror ``export_comparison_runs.py`` (W&B pull) and the CSV-fallback
pattern in ``plot_comparison_curves.py``: each run's series is cached as a tidy
``ksweep_<label>_eval_reward.csv`` so the figure can be regenerated offline.

    python report-materials/scripts/plot_ksweep_eval_reward.py            # cache-first
    python report-materials/scripts/plot_ksweep_eval_reward.py --refresh  # force W&B re-pull
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

EVAL_TAG = "rewards/eval/score/mean"

# label -> (W&B run id, legend label, colour). Colours are stable for the figure.
RUNS = {
    "k2": ("jgs4c6kl", "K=2", "tab:red"),
    "k4": ("x4j7yhdp", "K=4", "tab:orange"),
    "k8": ("m3xp6k97", "K=8", "tab:blue"),
    "k16": ("xqbl406c", "K=16", "tab:green"),
}


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
        writer.writerow(["step", "eval_score_mean"])
        writer.writerows(pts)


def read_csv(path: Path) -> list[tuple[int, float]]:
    pts: list[tuple[int, float]] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pts.append((int(row["step"]), float(row["eval_score_mean"])))
    return pts


def load_series(label: str, run_id: str, entity: str, project: str, refresh: bool) -> list[tuple[int, float]]:
    cache = DATA_DIR / f"ksweep_{label}_eval_reward.csv"
    if cache.exists() and not refresh:
        return read_csv(cache)
    pts = fetch_wandb(entity, project, run_id)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(cache, pts)
    return pts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", default=DEFAULT_ENTITY)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--refresh", action="store_true", help="force re-pull from W&B")
    args = parser.parse_args()

    plt.figure(figsize=(7, 4))
    for label, (run_id, legend, colour) in RUNS.items():
        pts = load_series(label, run_id, args.entity, args.project, args.refresh)
        if not pts:
            print(f"  warning: no {EVAL_TAG} data for {label} ({run_id}); skipping")
            continue
        steps = [s for s, _ in pts]
        values = [v for _, v in pts]
        plt.plot(steps, values, linewidth=1.4, color=colour, label=legend)
        print(f"  {label} ({run_id}): {len(pts)} eval points, steps {steps[0]}..{steps[-1]}")

    plt.xlabel("GRPO step")
    plt.ylabel(r"$\bar{r}$ (held-out eval)")
    plt.legend(loc="best", fontsize=9, title="group size")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    out = FIG_DIR / "ksweep_eval_reward"
    plt.savefig(out.with_suffix(".png"), dpi=200)
    plt.savefig(out.with_suffix(".pdf"))
    plt.close()
    print(f"wrote {out}.pdf / .png")


if __name__ == "__main__":
    main()
