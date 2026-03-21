from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def score_predictions(summary_csv: str | Path, parsed_csv: str | Path, scored_csv: str | Path) -> Path:
    summary_csv = Path(summary_csv)
    parsed_csv = Path(parsed_csv)
    scored_csv = Path(scored_csv)
    scored_csv.parent.mkdir(parents=True, exist_ok=True)

    truth: dict[str, dict[str, str]] = {}
    with summary_csv.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            truth[f"{row['run_id']}::{row['crop_name']}"] = row

    scored_rows: list[dict[str, Any]] = []
    with parsed_csv.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = row["custom_id"]
            gold = truth.get(key, {})
            est_i0 = _to_float(row.get("estimated_i0"))
            est_r = _to_float(row.get("estimated_r"))
            true_i0 = _to_float(gold.get("true_i0"))
            true_r = _to_float(gold.get("true_r"))
            scored_rows.append(
                {
                    **row,
                    "true_i0": true_i0,
                    "true_r": true_r,
                    "abs_error_i0": None if est_i0 is None or true_i0 is None else abs(est_i0 - true_i0),
                    "abs_error_r": None if est_r is None or true_r is None else abs(est_r - true_r),
                    "exact_i0": int(est_i0 == true_i0) if est_i0 is not None and true_i0 is not None else 0,
                    "exact_r": int(est_r == true_r) if est_r is not None and true_r is not None else 0,
                    "exact_both": int(est_i0 == true_i0 and est_r == true_r)
                    if est_i0 is not None and est_r is not None and true_i0 is not None and true_r is not None
                    else 0,
                }
            )

    fieldnames = [
        "custom_id", "estimated_i0", "estimated_r", "confidence", "reasoning_summary", "raw_content",
        "true_i0", "true_r", "abs_error_i0", "abs_error_r", "exact_i0", "exact_r", "exact_both",
    ]
    with scored_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(scored_rows)
    return scored_csv
