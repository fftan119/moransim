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
) -> Path:
    voted_csv = Path(voted_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    with voted_csv.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                r_val = float(row["true_r"])
                i0_val = int(row["true_i0"])
                label = row["majority_vote"].strip().upper()
            except (KeyError, ValueError):
                continue
            if label in ("X", "O"):
                rows.append({"r": r_val, "i0": i0_val, "label": label})

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

    # Axes orientation: x = i0 (0 left → i0_max right), y = r (0 bottom → r_max top)
    # but we want r descending top-to-bottom so invert y
    ax.set_xlim(-0.5, i0_max + 0.5)
    ax.set_ylim(0, r_max)
    ax.invert_yaxis()  # r=0 at top, r=r_max at bottom

    ax.set_xlabel("Initial mutants  $i_0$", fontsize=13)
    ax.set_ylabel("Relative fitness  $r$", fontsize=13)
    ax.set_title(
        "GPT classification: fixation probability $\\rho$ vs 0.5\n"
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

    print(f"Saved: {out_path}")
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
    args = parser.parse_args()

    plot_classification_grid(
        voted_csv=args.voted_csv,
        output_dir=args.output_dir,
        i0_max=args.i0_max,
        r_max=args.r_max,
    )


if __name__ == "__main__":
    main()
