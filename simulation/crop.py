from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable


def _read_rows(csv_path: str | Path) -> list[dict[str, str]]:
    with Path(csv_path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_rows(rows: Iterable[dict[str, str]], out_path: str | Path) -> Path:
    out_path = Path(out_path)
    rows = list(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else [
        "step",
        "event",
        "birth_index",
        "birth_type",
        "death_index",
        "death_type",
        "mutants_before",
        "mutants_after",
        "N",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def make_crop_variants(raw_trace_csv: str | Path, output_dir: str | Path, *, prefix_k: int = 10, suffix_k: int = 10, stride: int = 3) -> dict[str, Path]:
    rows = _read_rows(raw_trace_csv)
    output_dir = Path(output_dir)
    stem = Path(raw_trace_csv).stem

    variants = {
        "full": rows,
        "prefix10": rows[:prefix_k],
        "suffix10": rows[-suffix_k:] if rows else [],
        "stride3": rows[::stride] if stride > 0 else rows,
    }

    out: dict[str, Path] = {}
    for crop_name, crop_rows in variants.items():
        out[crop_name] = _write_rows(crop_rows, output_dir / f"{stem}__{crop_name}.csv")
    return out
