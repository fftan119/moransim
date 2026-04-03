from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def plot_estimation_grid(
    scored_csv: str | Path,
    output_dir: str | Path,
) -> Path:
    scored_csv = Path(scored_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Group estimated_r values by (true_r, true_i0)
    data: dict[tuple, list[float]] = defaultdict(list)
    with scored_csv.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                true_r = float(row["true_r"])
                true_i0 = int(row["true_i0"])
                est_r = float(row["estimated_r"])
            except (KeyError, ValueError, TypeError):
                continue
            data[(true_r, true_i0)].append(est_r)

    if not data:
        raise ValueError(f"No valid rows found in {scored_csv}")

    points = sorted(data.keys())
    n = len(points)
    fig, axes = plt.subplots(1, n, figsize=(max(6, 2.5 * n), 5), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, (true_r, true_i0) in zip(axes, points):
        estimates = data[(true_r, true_i0)]
        ax.violinplot(estimates, positions=[0], showmedians=True)
        ax.axhline(true_r, color="red", linestyle="--", linewidth=1.5, label=f"True r={true_r}")
        ax.set_title(f"r={true_r}\ni0={true_i0}", fontsize=9)
        ax.set_xticks([])
        ax.set_ylabel("Estimated r" if ax == axes[0] else "")
        ax.legend(fontsize=7)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.25))

    fig.suptitle("GPT r estimation vs true r across grid points", fontsize=12)
    fig.tight_layout()
    out_path = output_dir / "estimation_grid.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {out_path}  ({n} grid points)")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize r estimation results across grid.")
    parser.add_argument("--group", default=None, help="Group name")
    parser.add_argument("--scored-csv", default=None, help="Direct path to estimation_scored.csv")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    if args.scored_csv:
        scored_csv = Path(args.scored_csv)
    elif args.group:
        scored_csv = Path("data/groups") / args.group / "results" / "estimation_scored.csv"
    else:
        scored_csv = Path("data/results/estimation_scored.csv")

    output_dir = Path(args.output_dir) if args.output_dir else scored_csv.parent.parent / "plots"
    plot_estimation_grid(scored_csv=scored_csv, output_dir=output_dir)


if __name__ == "__main__":
    main()
