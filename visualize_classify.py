from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def plot_classification_grid(
    voted_csv: str | Path,
    output_dir: str | Path = "data/results/plots",
    i0_max: int = 20,
    r_max: float = 2.0,
    use_true_label: bool = False,
) -> Path:
    voted_csv = Path(voted_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Deduplicate by (true_r, true_i0) — last write wins
    label_col = "true_label" if use_true_label else "majority_vote"
    seen: dict[tuple, dict] = {}
    with voted_csv.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                r_val = float(row["true_r"])
                i0_val = int(row["true_i0"])
                label = row[label_col].strip().upper()
            except (KeyError, ValueError):
                continue
            if label in ("X", "O"):
                seen[(r_val, i0_val)] = {"r": r_val, "i0": i0_val, "label": label}

    rows = list(seen.values())
    if not rows:
        raise ValueError(f"No valid rows found in {voted_csv}")

    fig, ax = plt.subplots(figsize=(10, 7))

    for entry in rows:
        i0 = entry["i0"]
        r = entry["r"]
        label = entry["label"]
        color = "red" if label == "X" else "black"
        ax.text(
            i0, r, label,
            ha="center", va="center",
            fontsize=14, fontweight="bold",
            color=color,
        )

    ax.set_xlim(-0.5, i0_max + 0.5)
    ax.set_ylim(0, r_max)
    ax.invert_yaxis()  # r=0 at top, r=r_max at bottom

    ax.set_xlabel("Initial mutants  $i_0$", fontsize=13)
    ax.set_ylabel("Relative fitness  $r$", fontsize=13)
    label_source = "true label" if use_true_label else "GPT majority vote"
    ax.set_title(
        f"Fixation probability classification ({label_source})\n"
        r"$\mathbf{X}$ = $\rho > 0.5$  (red),  $\mathbf{O}$ = $\rho < 0.5$  (black)",
        fontsize=12,
    )

    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.25))
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)

    fig.tight_layout()
    out_path = output_dir / "classification_grid.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    print(f"Saved: {out_path}  ({len(rows)} points plotted)")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot X/O classification grid over (i0, r) space.")
    parser.add_argument(
        "--voted-csv",
        default="data/results/classify_voted.csv",
        help="Path to classify_voted.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="data/results/plots",
        help="Directory for saved plot",
    )
    parser.add_argument(
        "--i0-max",
        type=int,
        default=20,
        help="Maximum i0 value on x-axis",
    )
    parser.add_argument(
        "--r-max",
        type=float,
        default=2.0,
        help="Maximum r value on y-axis",
    )
    parser.add_argument(
        "--true-label",
        action="store_true",
        help="Plot the analytically computed true label instead of GPT majority vote",
    )
    args = parser.parse_args()

    plot_classification_grid(
        voted_csv=args.voted_csv,
        output_dir=args.output_dir,
        i0_max=args.i0_max,
        r_max=args.r_max,
        use_true_label=args.true_label,
    )


if __name__ == "__main__":
    main()