from __future__ import annotations

import csv
import json
from pathlib import Path
from .moran import MoranRun

OBSERVABLE_TRACE_COLUMNS = [
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


def write_run_trace_csv(run: MoranRun, csv_path: str | Path) -> Path:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OBSERVABLE_TRACE_COLUMNS)
        writer.writeheader()
        for ev in run.steps:
            writer.writerow(
                {
                    "step": ev.step,
                    "event": ev.event,
                    "birth_index": ev.birth_index,
                    "birth_type": ev.birth_type,
                    "death_index": ev.death_index,
                    "death_type": ev.death_type,
                    "mutants_before": ev.mutants_before,
                    "mutants_after": ev.mutants_after,
                    "N": ev.N,
                }
            )
    return csv_path


def write_run_metadata_json(run: MoranRun, json_path: str | Path) -> Path:
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run.run_id,
        "true_r": run.true_r,
        "true_N": run.true_N,
        "true_i0": run.true_i0,
        "absorbed_type": run.absorbed_type,
        "num_events": len(run.steps),
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return json_path
