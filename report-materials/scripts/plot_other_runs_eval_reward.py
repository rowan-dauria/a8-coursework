"""Plot held-out eval-reward curves for the other-parameter-change runs (appendix).

These are the I.3 exploratory runs that vary a single training knob other than the
group size or the reward reweighting (which already appear in the report): learning
rate, LoRA rank/alpha, the reference-KL coefficient beta, and the length penalty.
For each run the held-out eval reward ``rewards/eval/score/mean`` (logged every 64
steps on the 64-example eval) is pulled from W&B and overlaid as a raw markerless
line, mirroring ``plot_ksweep_eval_reward.py``.

Each run's series is cached as ``other_<label>_eval_reward.csv`` so the figure can
be regenerated offline; the cache is used when present unless ``--refresh``.

    python report-materials/scripts/plot_other_runs_eval_reward.py            # cache-first
    python report-materials/scripts/plot_other_runs_eval_reward.py --refresh  # force re-pull
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
    "lr1e5": ("hozux9t6", "LR 1e-5 (K=8)", "tab:red"),
    "lora_r128_g2": ("v5cvlwkm", "rank/$\\alpha$ 128 (K=2)", "tab:orange"),
    "lora_r128_g8": ("aoz8dtkp", "rank/$\\alpha$ 128 (K=8)", "tab:green"),
    "kl_beta1e-6": ("8rmv0hgg", r"$\beta$=1e-6 (K=2)", "tab:purple"),
    "kl_beta032": ("oet2tfjd", r"$\beta$=0.32 (K=2)", "tab:brown"),
    "reward_length_g2": ("jcp0b5cy", "length penalty (K=2)", "tab:pink"),
    "reward_length_g8bs1": ("cyay16mj", "length penalty (K=8)", "tab:blue"),
    "hard_medium_data": ("6yiowy1y", "hard/medium data (K=8)", "tab:gray"),
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
    with path.open(newline="", encoding="utf-8") as f:
        return [(int(r["step"]), float(r["eval_score_mean"])) for r in csv.DictReader(f)]


def load_series(label: str, run_id: str, entity: str, project: str, refresh: bool) -> list[tuple[int, float]]:
    cache = DATA_DIR / f"other_{label}_eval_reward.csv"
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

    plt.figure(figsize=(8, 4))
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
    plt.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    out = FIG_DIR / "other_runs_eval_reward"
    plt.savefig(out.with_suffix(".png"), dpi=200)
    plt.savefig(out.with_suffix(".pdf"))
    plt.close()
    print(f"wrote {out}.pdf / .png")


if __name__ == "__main__":
    main()
