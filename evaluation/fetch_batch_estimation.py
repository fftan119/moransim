from __future__ import annotations

import json
import os
from pathlib import Path

import dotenv
from openai import OpenAI


def fetch_estimation_batches(
    batch_ids_jsonl: str | Path,
    output_dir: str | Path,
    *,
    verbose: bool = True,
) -> list[Path]:
    dotenv.load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    client = OpenAI(api_key=api_key)

    batch_ids_jsonl = Path(batch_ids_jsonl)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    if not batch_ids_jsonl.exists():
        raise FileNotFoundError(f"Batch IDs file not found: {batch_ids_jsonl}")

    with batch_ids_jsonl.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]

    for row in rows:
        batch_id = row["batch_job_id"]
        batch = client.batches.retrieve(batch_id)
        if verbose:
            print(
                f"batch_id={batch_id} status={batch.status} "
                f"output_file_id={getattr(batch, 'output_file_id', None)}"
            )
        if batch.status == "completed" and getattr(batch, "output_file_id", None):
            content = client.files.content(batch.output_file_id)
            out_path = output_dir / f"{batch_id}_estimation_output.jsonl"
            out_path.write_bytes(content.content)
            outputs.append(out_path)
        elif batch.status in {"failed", "expired", "cancelled"} and getattr(batch, "error_file_id", None):
            err = client.files.content(batch.error_file_id)
            (output_dir / f"{batch_id}_estimation_errors.jsonl").write_bytes(err.content)

    return outputs
