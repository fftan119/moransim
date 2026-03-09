from __future__ import annotations

import csv
from pathlib import Path
from typing import Literal

CropMode = Literal["full", "prefix", "suffix", "window", "stride"]


def crop_trace(
    input_csv: str | Path,
    output_csv: str | Path,
    mode: CropMode = "full",
    k: int | None = None,
    start: int | None = None,
    stop: int | None = None,
    stride: int | None = None,
) -> Path:
    input_csv = Path(input_csv)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with input_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    if mode == "full":
        cropped = rows
    elif mode == "prefix":
        if k is None:
            raise ValueError("k is required for prefix crop")
        cropped = rows[:k]
    elif mode == "suffix":
        if k is None:
            raise ValueError("k is required for suffix crop")
        cropped = rows[-k:]
    elif mode == "window":
        if start is None or stop is None:
            raise ValueError("start and stop are required for window crop")
        cropped = rows[start:stop]
    elif mode == "stride":
        if stride is None or stride <= 0:
            raise ValueError("positive stride is required for stride crop")
        cropped = rows[::stride]
    else:
        raise ValueError(f"Unsupported crop mode: {mode}")

    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cropped)

    return output_csv
