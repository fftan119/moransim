from __future__ import annotations

"""
Computes accuracy, precision, and recall from classify_voted.csv files
for one or more groups, grouped by population size N.

Usage:
    python compute_classification_stats.py --groups N15rho N20rho
    python compute_classification_stats.py --groups N15rho N20rho --latex
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def compute_stats(voted_csv: Path) -> dict:
    """
    Compute accuracy, precision, recall from a classify_voted.csv.
    X is treated as the positive class (rho > 0.5).
    Ties (T) are excluded from all metrics.
    """
    rows = []
    with voted_csv.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                true_label   = row["true_label"].strip().upper()
                majority     = row["majority_vote"].strip().upper()
                true_N       = int(row["true_N"])
            except (KeyError, ValueError):
                continue
            rows.append({
                "true_label": true_label,
                "majority":   majority,
                "true_N":     true_N,
            })

    if not rows:
        return {}

    # Group by N
    by_N: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_N[row["true_N"]].append(row)

    results = {}
    for N, group in sorted(by_N.items()):
        # Exclude ties
        valid = [r for r in group if r["majority"] in ("X", "O")]
        n_configs = len(group)
        n_valid   = len(valid)

        if n_valid == 0:
            results[N] = {"N": N, "configs": n_configs, "accuracy": None,
                          "precision": None, "recall": None}
            continue

        # TP: predicted X, true X
        # FP: predicted X, true O
        # FN: predicted O, true X
        # TN: predicted O, true O
        TP = sum(1 for r in valid if r["majority"] == "X" and r["true_label"] == "X")
        FP = sum(1 for r in valid if r["majority"] == "X" and r["true_label"] == "O")
        FN = sum(1 for r in valid if r["majority"] == "O" and r["true_label"] == "X")
        TN = sum(1 for r in valid if r["majority"] == "O" and r["true_label"] == "O")

        accuracy  = (TP + TN) / n_valid if n_valid > 0 else None
        precision = TP / (TP + FP) if (TP + FP) > 0 else None
        recall    = TP / (TP + FN) if (TP + FN) > 0 else None

        results[N] = {
            "N":         N,
            "configs":   n_configs,
            "accuracy":  accuracy,
            "precision": precision,
            "recall":    recall,
            "TP": TP, "FP": FP, "FN": FN, "TN": TN,
            "ties": n_configs - n_valid,
        }

    return results


def fmt(val, decimals=3) -> str:
    if val is None:
        return "---"
    return f"{val:.{decimals}f}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute classification stats from classify_voted.csv files."
    )
    parser.add_argument(
        "--groups", nargs="+", required=True,
        help="Group names to include (e.g. N15rho N20rho)"
    )
    parser.add_argument(
        "--latex", action="store_true",
        help="Print LaTeX table row format"
    )
    args = parser.parse_args()

    all_results: dict[int, dict] = {}

    for group in args.groups:
        voted_csv = Path("data/groups") / group / "results" / "classify_voted.csv"
        if not voted_csv.exists():
            print(f"WARNING: {voted_csv} not found, skipping.")
            continue
        stats = compute_stats(voted_csv)
        for N, row in stats.items():
            # Merge — if same N appears in multiple groups, combine
            if N not in all_results:
                all_results[N] = row
            else:
                # Just warn for now
                print(f"WARNING: N={N} appears in multiple groups — using first occurrence.")

    if not all_results:
        print("No results found.")
        return

    print("\n── Classification Statistics ──────────────────────────────")
    print(f"{'N':>4}  {'Configs':>8}  {'Accuracy':>10}  {'Precision':>10}  {'Recall':>8}  {'Ties':>6}")
    print("-" * 60)
    for N, r in sorted(all_results.items()):
        print(
            f"{N:>4}  {r['configs']:>8}  "
            f"{fmt(r['accuracy']):>10}  "
            f"{fmt(r['precision']):>10}  "
            f"{fmt(r['recall']):>8}  "
            f"{r.get('ties', 0):>6}"
        )
        if "TP" in r:
            print(f"       TP={r['TP']} FP={r['FP']} FN={r['FN']} TN={r['TN']}")

    if args.latex:
        print("\n── LaTeX table rows ───────────────────────────────────────")
        for N, r in sorted(all_results.items()):
            print(
                f"{N} & {r['configs']} & 20 & "
                f"{fmt(r['accuracy'])} & "
                f"{fmt(r['precision'])} & "
                f"{fmt(r['recall'])} \\\\"
            )


if __name__ == "__main__":
    main()