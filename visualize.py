from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def _highlight_hist_bin(ax, data, true_value, bins, xlabel, title):
    counts, edges, patches = ax.hist(data, bins=bins)

    # Color the bar whose interval contains the true value
    highlighted = False
    for left, right, patch in zip(edges[:-1], edges[1:], patches):
        if left <= true_value < right or (true_value == edges[-1] and right == edges[-1]):
            patch.set_hatch("//")
            patch.set_linewidth(2.0)
            highlighted = True
            break

    # Also draw a vertical line so the exact true value is obvious
    ax.axvline(true_value, linestyle="--", linewidth=2, label=f"Actual value = {true_value}")

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    ax.legend()


def plot_distributions(
    scored_csv: str | Path,
    output_dir: str | Path = "data/results/plots",
    r_bins: int = 12,
    i0_bins: int | None = None,
) -> None:
    scored_csv = Path(scored_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(scored_csv)

    required = ["estimated_r", "true_r", "estimated_i0", "true_i0"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["estimated_r"] = pd.to_numeric(df["estimated_r"], errors="coerce")
    df["true_r"] = pd.to_numeric(df["true_r"], errors="coerce")
    df["estimated_i0"] = pd.to_numeric(df["estimated_i0"], errors="coerce")
    df["true_i0"] = pd.to_numeric(df["true_i0"], errors="coerce")
    df = df.dropna(subset=required)

    true_r_values = sorted(df["true_r"].unique())
    true_i0_values = sorted(df["true_i0"].unique())

    if len(true_r_values) != 1 or len(true_i0_values) != 1:
        raise ValueError(
            "This plot expects one fixed true_r and one fixed true_i0 across the file. "
            f"Found true_r values={true_r_values} and true_i0 values={true_i0_values}. "
            "Filter to one experiment/setting first."
        )

    true_r = float(true_r_values[0])
    true_i0 = int(true_i0_values[0])

    # ---- r distribution ----
    fig, ax = plt.subplots(figsize=(8, 5))
    _highlight_hist_bin(
        ax=ax,
        data=df["estimated_r"],
        true_value=true_r,
        bins=r_bins,
        xlabel="Estimated r",
        title="Distribution of Estimated r",
    )
    fig.tight_layout()
    r_path = output_dir / "r_distribution.png"
    fig.savefig(r_path, dpi=200)
    plt.close(fig)

    # ---- i0 distribution ----
    i0_data = df["estimated_i0"].astype(int)

    if i0_bins is None:
        min_i0 = int(i0_data.min())
        max_i0 = int(i0_data.max())
        # integer-centered bins
        bins = np.arange(min_i0 - 0.5, max_i0 + 1.5, 1)
    else:
        bins = i0_bins

    fig, ax = plt.subplots(figsize=(8, 5))
    _highlight_hist_bin(
        ax=ax,
        data=i0_data,
        true_value=true_i0,
        bins=bins,
        xlabel="Estimated i0",
        title="Distribution of Estimated i0",
    )
    fig.tight_layout()
    i0_path = output_dir / "i0_distribution.png"
    fig.savefig(i0_path, dpi=200)
    plt.close(fig)

    print(f"Saved: {r_path}")
    print(f"Saved: {i0_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot distributions of GPT estimates.")
    parser.add_argument(
        "--scored-csv",
        default="data/results/scored_predictions.csv",
        help="Path to scored_predictions.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="data/results/plots",
        help="Directory for saved plots",
    )
    parser.add_argument(
        "--r-bins",
        type=int,
        default=12,
        help="Number of bins for r histogram",
    )
    args = parser.parse_args()

    plot_distributions(
        scored_csv=args.scored_csv,
        output_dir=args.output_dir,
        r_bins=args.r_bins,
    )


if __name__ == "__main__":
    main()