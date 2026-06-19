"""Export baseline I.1 reward and KL curves from TensorBoard events.

The per-step series are read from the run's TensorBoard event file when it is
available; otherwise the previously exported ``<name>.csv`` in ``--out-dir`` is
used, so the figures can be regenerated off the TPU.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DEFAULT_EVENT_FILE = Path(
    "/tmp/content/tmp/tensorboard/grpo/events.out.tfevents.1780942641.t1v-n-f769470f-w-0"
)
DEFAULT_OUT_DIR = Path("report_assets")

SERIES = {
    "baseline_mean_reward_curve": {
        # score/mean is the mean per-rollout total reward (sum of the four reward
        # functions), i.e. the GRPO mean reward r-bar. rewards/train/mean is the
        # same series divided by the number of reward functions (4).
        "tag": "rewards/train/score/mean",
        "ylabel": "Mean reward",
        "title": "Baseline GRPO training reward",
        "ma_window": 64,
    },
    "baseline_kl_curve": {
        "tag": "actor/train/kl",
        "ylabel": "KL(pi_theta || pi_ref)",
        "title": "Baseline KL divergence",
        "ma_window": None,
    },
}


def load_series(event_file: Path, tag: str) -> list[tuple[int, float]]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    accumulator = EventAccumulator(str(event_file))
    accumulator.Reload()
    tags = accumulator.Tags().get("scalars", [])
    if tag not in tags:
        raise ValueError(f"Scalar tag {tag!r} not found. Available tags: {sorted(tags)}")
    return [(point.step, point.value) for point in accumulator.Scalars(tag)]


def load_series_from_csv(path: Path) -> list[tuple[int, float]]:
    with path.open(newline="") as f:
        reader = csv.reader(f)
        next(reader)  # header: step, <tag>
        return [(int(step), float(value)) for step, value in reader]


def centred_moving_average(values: list[float], window: int) -> np.ndarray:
    """Centred moving average with shrinking windows at the edges."""
    v = np.asarray(values, dtype=float)
    kernel = np.ones(window)
    sums = np.convolve(v, kernel, mode="same")
    counts = np.convolve(np.ones_like(v), kernel, mode="same")
    return sums / counts


def write_csv(path: Path, rows: list[tuple[int, float]], value_name: str) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", value_name])
        writer.writerows(rows)


def write_plot(
    path_base: Path,
    rows: list[tuple[int, float]],
    ylabel: str,
    title: str,
    ma_window: int | None = None,
) -> None:
    steps = [step for step, _ in rows]
    values = [value for _, value in rows]

    plt.figure(figsize=(7, 4))
    if ma_window:
        plt.plot(steps, values, linewidth=0.8, alpha=0.25, color="tab:blue",
                 label="raw (per step)")
        plt.plot(steps, centred_moving_average(values, ma_window), linewidth=1.8,
                 color="tab:blue", label=f"{ma_window}-step moving average")
        plt.legend(loc="upper right", fontsize=8)
    else:
        plt.plot(steps, values, linewidth=1.6)
    plt.xlabel("GRPO step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path_base.with_suffix(".png"), dpi=200)
    plt.savefig(path_base.with_suffix(".pdf"))
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event-file", type=Path, default=DEFAULT_EVENT_FILE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--series",
        nargs="*",
        choices=sorted(SERIES),
        default=sorted(SERIES),
        help="Which series to (re)generate. Defaults to all.",
    )
    args = parser.parse_args()

    use_tb = args.event_file.exists()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for name in args.series:
        cfg = SERIES[name]
        if use_tb:
            rows = load_series(args.event_file, cfg["tag"])
            write_csv(args.out_dir / f"{name}.csv", rows, cfg["tag"])
            source = "tensorboard"
        else:
            csv_path = args.out_dir / f"{name}.csv"
            if not csv_path.exists():
                raise FileNotFoundError(
                    f"{args.event_file} not found and no fallback CSV at {csv_path}"
                )
            rows = load_series_from_csv(csv_path)
            source = str(csv_path)
        write_plot(args.out_dir / name, rows, cfg["ylabel"], cfg["title"], cfg["ma_window"])
        print(f"wrote {name}: {len(rows)} points (source: {source})")


if __name__ == "__main__":
    main()
