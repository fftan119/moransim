from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import dotenv
from openai import OpenAI

from .prompts import build_system_prompt, build_user_prompt_from_csv


def make_task(row: dict[str, str], model_name: str) -> dict:
    return {
        "custom_id": f"{row['run_id']}::{row['crop_name']}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model_name,
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": build_system_prompt()},
                {"role": "user", "content": build_user_prompt_from_csv(row["trace_csv"])},
            ],
        },
    }


def send_batch(summary_csv: str | Path, batch_jsonl: str | Path, model_name: str | None = None) -> str:
    dotenv.load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    client = OpenAI(api_key=api_key)
    model_name = model_name or os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")

    summary_csv = Path(summary_csv)
    batch_jsonl = Path(batch_jsonl)
    batch_jsonl.parent.mkdir(parents=True, exist_ok=True)

    tasks: list[dict] = []
    with summary_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            tasks.append(make_task(row, model_name=model_name))

    with batch_jsonl.open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps(task) + "\n")

    uploaded = client.files.create(file=batch_jsonl.open("rb"), purpose="batch")
    batch_job = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )

    job_record = batch_jsonl.with_name(f"batch_job_ids_{model_name}.jsonl")
    with job_record.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"batch_job_id": batch_job.id, "model": model_name, "summary_csv": str(summary_csv)}) + "\n")

    return batch_job.id
