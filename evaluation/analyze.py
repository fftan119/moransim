from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


def summarize_scores(scored_csv: str | Path) -> str:
    scored_csv = Path(scored_csv)
    by_crop: dict[str, dict[str, float]] = defaultdict(lambda: {"count": 0, "exact_both": 0, "sum_abs_error_r": 0.0, "sum_abs_error_i0": 0.0})

    with scored_csv.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            custom_id = row["custom_id"]
            crop_name = custom_id.split("::", 1)[1] if "::" in custom_id else "unknown"
            stats = by_crop[crop_name]
            stats["count"] += 1
            stats["exact_both"] += float(row.get("exact_both") or 0)
            try:
                stats["sum_abs_error_r"] += float(row.get("abs_error_r") or 0)
                stats["sum_abs_error_i0"] += float(row.get("abs_error_i0") or 0)
            except ValueError:
                pass

    lines = ["Crop summary"]
    for crop_name, stats in sorted(by_crop.items()):
        count = max(stats["count"], 1)
        lines.append(
            f"- {crop_name}: n={int(stats['count'])}, exact_both={stats['exact_both']/count:.3f}, "
            f"mean_abs_error_r={stats['sum_abs_error_r']/count:.3f}, "
            f"mean_abs_error_i0={stats['sum_abs_error_i0']/count:.3f}"
        )
    return "\n".join(lines)
