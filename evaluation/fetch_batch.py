from __future__ import annotations

import json
import os
from pathlib import Path

import dotenv
from openai import OpenAI


def read_batch_ids(file_path: str | Path) -> list[dict]:
    file_path = Path(file_path)
    if not file_path.exists():
        return []
    rows: list[dict] = []
    with file_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def fetch_completed_batches(batch_ids_jsonl: str | Path, output_dir: str | Path) -> list[Path]:
    dotenv.load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    client = OpenAI(api_key=api_key)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    for row in read_batch_ids(batch_ids_jsonl):
        batch_id = row["batch_job_id"]
        batch = client.batches.retrieve(batch_id)
        if batch.status == "completed" and batch.output_file_id:
            content = client.files.content(batch.output_file_id)
            out_path = output_dir / f"{batch_id}_output.jsonl"
            out_path.write_bytes(content.content)
            outputs.append(out_path)
        elif batch.status == "failed" and batch.error_file_id:
            err = client.files.content(batch.error_file_id)
            (output_dir / f"{batch_id}_errors.jsonl").write_bytes(err.content)
    return outputs
