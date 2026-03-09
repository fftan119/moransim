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


def parse_batch_outputs(output_jsonl: str | Path, parsed_csv: str | Path) -> Path:
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
            rows.append(
                {
                    "custom_id": record.get("custom_id"),
                    "estimated_i0": parsed.get("estimated_i0"),
                    "estimated_r": parsed.get("estimated_r"),
                    "confidence": parsed.get("confidence"),
                    "reasoning_summary": parsed.get("reasoning_summary"),
                    "raw_content": content,
                }
            )

    with parsed_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [
            "custom_id", "estimated_i0", "estimated_r", "confidence", "reasoning_summary", "raw_content"
        ])
        writer.writeheader()
        writer.writerows(rows)
    return parsed_csv
