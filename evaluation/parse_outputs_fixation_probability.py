from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def _extract_message_content(response_line: dict[str, Any]) -> str | None:
    try:
        return response_line["response"]["body"]["choices"][0]["message"]["content"]
    except Exception:
        return None


def parse_classify_outputs(output_jsonl: str | Path, parsed_csv: str | Path) -> Path:
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
            content = _extract_message_content(record)
            parsed: dict[str, Any] = {}
            if content:
                try:
                    parsed = json.loads(content)
                except json.JSONDecodeError:
                    parsed = {"raw_content": content}

            custom_id = record.get("custom_id", "")
            # custom_id format: classify__exp001_run01
            run_id = custom_id.removeprefix("classify__")
            # exp_id is everything before the last _runXX
            exp_id = "_".join(run_id.split("_")[:-1]) if "_run" in run_id else run_id

            rows.append({
                "custom_id": custom_id,
                "run_id": run_id,
                "exp_id": exp_id,
                "label": parsed.get("label"),
                "estimated_r": parsed.get("estimated_r"),
                "estimated_i0": parsed.get("estimated_i0"),
                "rho_estimated": parsed.get("rho_estimated"),
                "reasoning_summary": parsed.get("reasoning_summary"),
                "raw_content": content,
            })

    fieldnames = [
        "custom_id", "run_id", "exp_id", "label",
        "estimated_r", "estimated_i0", "rho_estimated",
        "reasoning_summary", "raw_content",
    ]
    with parsed_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return parsed_csv
