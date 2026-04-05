from __future__ import annotations

"""
Generates a CSV table from estimation_scored.csv where:
  - rows = true_r values (descending, r=0 at top)
  - columns = true_i0 values (ascending, i0=0 at left)
  - cells = mean estimated_r across all replicates for that (r, i0) point

Usage:
    python tabulate_estimation.py --group r_estimation_N20
    python tabulate_estimation.py --scored-csv path/to/estimation_scored.csv
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


def build_table(
    scored_csv: str | Path,
    output_csv: str | Path,
) -> Path:
    scored_csv = Path(scored_csv)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # Collect estimated_r values per (true_r, true_i0)
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

    # Get sorted unique r and i0 values
    r_values = sorted({r for r, i in data}, reverse=True)   # descending (r=0 at top)
    i0_values = sorted({i for r, i in data})                # ascending

    # Write table
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        # Header row: first col is r\i0, then i0 values
        writer.writerow(["r \\ i0"] + i0_values)
        for r in r_values:
            row = [r]
            for i0 in i0_values:
                estimates = data.get((r, i0), [])
                if estimates:
                    row.append(round(float(np.mean(estimates)), 4))
                else:
                    row.append("")   # empty if point not yet run
            writer.writerow(row)

    n_points = len(data)
    n_replicates = sum(len(v) for v in data.values())
    print(f"Table written to {output_csv}")
    print(f"  {n_points} grid points, {n_replicates} total replicates")
    print(f"  {len(r_values)} r values x {len(i0_values)} i0 values")
    return output_csv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a CSV table of mean estimated r per (r, i0) grid point."
    )
    parser.add_argument("--group", default=None, help="Group name (e.g. r_estimation_N20)")
    parser.add_argument("--scored-csv", default=None, help="Direct path to estimation_scored.csv")
    parser.add_argument("--output-csv", default=None, help="Output table CSV path")
    args = parser.parse_args()

    if args.scored_csv:
        scored_csv = Path(args.scored_csv)
    elif args.group:
        scored_csv = Path("data/groups") / args.group / "results" / "estimation_scored.csv"
    else:
        scored_csv = Path("data/results/estimation_scored.csv")

    if args.output_csv:
        output_csv = Path(args.output_csv)
    elif args.group:
        output_csv = Path("data/groups") / args.group / "results" / "estimation_table.csv"
    else:
        output_csv = Path("data/results/estimation_table.csv")

    build_table(scored_csv=scored_csv, output_csv=output_csv)


if __name__ == "__main__":
    main()