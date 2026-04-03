from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def parse_estimation_outputs(output_jsonl: str | Path, parsed_csv: str | Path) -> Path:
    output_jsonl = Path(output_jsonl)
    parsed_csv = Path(parsed_csv)
    parsed_csv.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    with output_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            try:
                content = record["response"]["body"]["choices"][0]["message"]["content"]
            except Exception:
                content = None

            parsed: dict[str, Any] = {}
            if content:
                try:
                    parsed = json.loads(content)
                except json.JSONDecodeError:
                    parsed = {}

            custom_id = record.get("custom_id", "")
            run_id = custom_id.removeprefix("estimate__")
            exp_id = "_".join(run_id.split("_")[:-1]) if "_run" in run_id else run_id

            r_raw = parsed.get("estimated_r")
            try:
                estimated_r = float(r_raw) if r_raw is not None else None
            except (TypeError, ValueError):
                estimated_r = None

            rows.append({
                "custom_id": custom_id,
                "run_id": run_id,
                "exp_id": exp_id,
                "estimated_r": estimated_r,
                "raw_content": content,
            })

    fieldnames = ["custom_id", "run_id", "exp_id", "estimated_r", "raw_content"]
    with parsed_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return parsed_csv
