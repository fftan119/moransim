from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def _rho_matrix(N: int, i_vals: np.ndarray, r_vals: np.ndarray) -> np.ndarray:
    rho = np.zeros((len(r_vals), len(i_vals)))
    for a, r in enumerate(r_vals):
        for b, i in enumerate(i_vals):
            if np.isclose(r, 1.0):
                rho[a, b] = i / N
            else:
                rho[a, b] = (1 - r ** (-i)) / (1 - r ** (-N))
    return rho


def plot_classification_grid(
    voted_csv: str | Path,
    output_dir: str | Path,
    use_true_label: bool = False,
    N: int = 20,
    i0_max: int | None = None,
    r_max: float | None = None,
) -> Path:
    voted_csv = Path(voted_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    all_i0 = [e["i0"] for e in rows]
    all_r  = [e["r"]  for e in rows]
    _i0_max = i0_max if i0_max is not None else max(N - 1, max(all_i0))
    _r_max  = r_max  if r_max  is not None else round(max(all_r) + 0.25, 2)

    fig, ax = plt.subplots(figsize=(10, 7))

    # --- rho = 0.5 contour — computed well beyond axis bounds so line reaches edges ---
    i_vals = np.linspace(-1, _i0_max + 2, 600)
    r_vals = np.linspace(0.001, _r_max + 1.0, 600)
    I, R = np.meshgrid(i_vals, r_vals)
    rho = _rho_matrix(N, i_vals, r_vals)

    cs = ax.contour(I, R, rho, levels=[0.5], colors="blue", linewidths=1.5)

    # Place label in top-right corner of the plot, out of the way
    ax.text(
        _i0_max - 0.3, 0.1,
        r"$\rho = 0.5$",
        color="blue", fontsize=10,
        va="top", ha="right",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="blue", alpha=0.7),
    )

    # --- X / O symbols ---
    for entry in rows:
        color = "red" if entry["label"] == "X" else "black"
        ax.text(entry["i0"], entry["r"], entry["label"],
                ha="center", va="center", fontsize=8, fontweight="bold", color=color)

    ax.set_xlim(-0.5, _i0_max + 0.5)
    ax.set_ylim(0, _r_max)
    ax.invert_yaxis()
    ax.set_xlabel("Initial mutants  $i_0$", fontsize=13)
    ax.set_ylabel("Relative fitness  $r$", fontsize=13)
    label_source = "true label" if use_true_label else "GPT majority vote"
    ax.set_title(
        f"Fixation probability classification ({label_source})  [N={N}]\n"
        r"$\mathbf{X}$ = $\rho > 0.5$  (red),  $\mathbf{O}$ = $\rho < 0.5$  (black),  "
        "blue line = $\\rho = 0.5$",
        fontsize=12,
    )
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.25))
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)

    fig.tight_layout()
    out_path = output_dir / "classification_grid.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {out_path}  ({len(rows)} points plotted, N={N})")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot X/O classification grid.")
    parser.add_argument("--group", default=None)
    parser.add_argument("--voted-csv", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--N", type=int, default=20)
    parser.add_argument("--i0-max", type=int, default=None)
    parser.add_argument("--r-max", type=float, default=None)
    parser.add_argument("--true-label", action="store_true")
    args = parser.parse_args()

    if args.voted_csv:
        voted_csv = Path(args.voted_csv)
    elif args.group:
        voted_csv = Path("data/groups") / args.group / "results" / "classify_voted.csv"
    else:
        voted_csv = Path("data/results/classify_voted.csv")

    output_dir = Path(args.output_dir) if args.output_dir else voted_csv.parent.parent / "plots"

    plot_classification_grid(
        voted_csv=voted_csv,
        output_dir=output_dir,
        use_true_label=args.true_label,
        N=args.N,
        i0_max=args.i0_max,
        r_max=args.r_max,
    )


if __name__ == "__main__":
    main()