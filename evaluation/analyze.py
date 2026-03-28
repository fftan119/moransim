from __future__ import annotations

import csv
from pathlib import Path


def summarize_scores(scored_csv: str | Path) -> str:
    scored_csv = Path(scored_csv)

    count = 0
    exact_both = 0.0
    sum_abs_error_r = 0.0
    sum_abs_error_i0 = 0.0

    with scored_csv.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            count += 1
            exact_both += float(row.get("exact_both") or 0)
            try:
                sum_abs_error_r += float(row.get("abs_error_r") or 0)
                sum_abs_error_i0 += float(row.get("abs_error_i0") or 0)
            except ValueError:
                pass

    if count == 0:
        return "No scored predictions."

    return (
        "Overall summary\n"
        f"- n={count}, "
        f"exact_both={exact_both/count:.3f}, "
        f"mean_abs_error_r={sum_abs_error_r/count:.3f}, "
        f"mean_abs_error_i0={sum_abs_error_i0/count:.3f}"
    )