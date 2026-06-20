#!/usr/bin/env python3
"""Export I.3 comparison training curves onto a common global-step axis.

Caches one tidy CSV per run in ``report_assets/`` so the comparison figures can
be regenerated offline by ``plot_comparison_curves.py`` (mirroring the
CSV-fallback pattern in ``plot_baseline_curves.py``). For each run we keep three
series: mean reward, reference-KL, and mean completion length (the
response-length diagnostic).

Data sources differ by run because the long K=8 runs were not streamed to W&B
in full:

* ``k2`` (keh7f5es) streamed every metric to W&B, so it is pulled from W&B.
* The two 5864-step K=8 runs were trained in two phases and only logged locally
  to TensorBoard. Each is reconstructed by stitching the shared 3364-step phase-1
  base (run ``m3xp6k97``) with the run's own phase-2 event file. On resume the
  reward/length step counter reset to 0 while KL kept the global step, so phases
  are stitched per metric: a metric whose first step in a later phase is not past
  the running maximum is shifted to continue after it (the others are already on
  the global axis). This recovers a continuous step 0..5864 trajectory.

The new-reward run's phase-1 was trained with the *default* reward, so its reward
column changes scale at the resume boundary; mean reward is only comparable
within a fixed reward function (see ``plot_comparison_curves.py``).

A JSON sidecar records each run's source, final step, and final metric values.

    python analysis/export_comparison_runs.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_OUT_DIR = SCRIPT_DIR.parent / "report_assets"
CHECKPOINTS = REPO_ROOT / "model-checkpoints"

DEFAULT_ENTITY = "felsomoye-university-of-cambridge"
DEFAULT_PROJECT = "tunix"

# TensorBoard scalar tags, mapped to the tidy CSV column names used downstream.
# We use rewards/train/score/mean (the mean per-rollout *total* reward summed over
# the four reward functions, i.e. the GRPO mean reward r-bar) rather than
# rewards/train/mean, which is the same series divided by the number of reward
# functions (4) and has no GRPO meaning.
METRICS = {
    "rewards/train/score/mean": "reward_mean",
    "actor/train/kl": "kl",
    "completions/train/mean_length": "mean_length",
    # Held-out eval counterparts (logged every EVAL_EVERY_N_STEPS=64 steps). The
    # reward/length plots use these eval series; KL stays on the train series
    # (there is no meaningful held-out reference-KL plot here).
    "rewards/eval/score/mean": "eval_reward_mean",
    "completions/eval/mean_length": "eval_mean_length",
}

# Shared 3364-step K=8 phase-1 base both long runs resumed from.
PHASE1 = CHECKPOINTS / "m3xp6k97/tensorboard/grpo/events.out.tfevents.1781270315.t1v-n-f769470f-w-0"

# label -> source spec. "wandb" pulls from W&B; "tb" stitches ordered event files.
RUNS: dict[str, dict[str, Any]] = {
    "k2": {"wandb": "keh7f5es"},
    "k8": {
        "tb": [
            PHASE1,
            CHECKPOINTS / "8k-baseline-6516-steps-rd/tensorboard/grpo/events.out.tfevents.1781397623.t1v-n-f769470f-w-0",
        ]
    },
    "k8_newreward": {
        "tb": [
            CHECKPOINTS / "k-8-new-reward/tensorboard/grpo/events.out.tfevents.1781270315.t1v-n-f769470f-w-0",
            CHECKPOINTS / "k-8-new-reward/tensorboard/grpo/events.out.tfevents.1781297867.t1v-n-f769470f-w-0",
        ]
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", default=DEFAULT_ENTITY)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


# --------------------------------------------------------------------------
# W&B source (k2)
# --------------------------------------------------------------------------

def collect_wandb(entity: str, project: str, run_id: str) -> tuple[dict[str, list[tuple[int, float]]], dict[str, Any]]:
    import wandb

    run = wandb.Api().run(f"{entity}/{project}/runs/{run_id}")
    series: dict[str, list[tuple[int, float]]] = {col: [] for col in METRICS.values()}
    for row in run.scan_history(keys=["_step", *METRICS], page_size=1000):
        step = row.get("_step")
        if step is None:
            continue
        for tag, col in METRICS.items():
            value = row.get(tag)
            if value is not None:
                series[col].append((int(step), float(value)))
    info = {"source": "wandb", "run_id": run.id, "name": run.name, "url": run.url}
    return series, info


# --------------------------------------------------------------------------
# TensorBoard source (k8, k8_newreward) with per-metric phase stitching
# --------------------------------------------------------------------------

def read_tb(path: Path, tag: str) -> list[tuple[int, float]]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    acc = EventAccumulator(str(path), size_guidance={"scalars": 0})
    acc.Reload()
    if tag not in acc.Tags().get("scalars", []):
        return []
    return [(int(p.step), float(p.value)) for p in acc.Scalars(tag)]


def stitch_phases(files: list[Path], tag: str) -> list[tuple[int, float]]:
    """Concatenate a metric across ordered phases onto a global step axis.

    A phase whose first step is not strictly past the running max is treated as
    a reset counter and shifted to continue after it; phases already on the
    global axis are kept as-is.
    """
    out: list[tuple[int, float]] = []
    running_max = -1
    for path in files:
        pts = read_tb(path, tag)
        if not pts:
            continue
        first = pts[0][0]
        offset = (running_max + 1 - first) if first <= running_max else 0
        out.extend((step + offset, value) for step, value in pts)
        running_max = out[-1][0]
    return out


def collect_tb(files: list[Path]) -> tuple[dict[str, list[tuple[int, float]]], dict[str, Any]]:
    for path in files:
        if not path.exists():
            raise FileNotFoundError(f"TensorBoard event file not found: {path}")
    series = {col: stitch_phases(files, tag) for tag, col in METRICS.items()}
    info = {
        "source": "tensorboard",
        "phase_files": [str(p.relative_to(REPO_ROOT)) for p in files],
    }
    return series, info


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------

def write_csv(path: Path, series: dict[str, list[tuple[int, float]]]) -> int:
    """Merge per-metric series into step-keyed rows (NaN-safe) and write CSV."""
    by_step: dict[int, dict[str, float]] = {}
    for col, pts in series.items():
        for step, value in pts:
            by_step.setdefault(step, {})[col] = value
    columns = ["step", *METRICS.values()]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for step in sorted(by_step):
            row = by_step[step]
            writer.writerow([step, *(row.get(col, "") for col in METRICS.values())])
    return max(by_step) if by_step else 0


def final_values(series: dict[str, list[tuple[int, float]]]) -> dict[str, float | None]:
    return {col: (pts[-1][1] if pts else None) for col, pts in series.items()}


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    meta: dict[str, Any] = {}
    for label, spec in RUNS.items():
        print(f"Building {label} ...")
        if "wandb" in spec:
            series, info = collect_wandb(args.entity, args.project, spec["wandb"])
        else:
            series, info = collect_tb(spec["tb"])

        csv_path = args.output_dir / f"compare_{label}_history.csv"
        max_step = write_csv(csv_path, series)
        finals = final_values(series)
        meta[label] = {
            **info,
            "max_step": max_step,
            "n_points": {col: len(pts) for col, pts in series.items()},
            "final_values": finals,
        }
        finals_str = ", ".join(
            f"{c}={v:.3f}" for c, v in finals.items() if v is not None and not math.isnan(v)
        )
        print(f"  wrote {csv_path} (max step {max_step}; final {finals_str})")

    meta_path = args.output_dir / "compare_runs_meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"wrote {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
