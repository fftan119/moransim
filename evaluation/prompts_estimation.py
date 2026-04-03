from __future__ import annotations

import csv
from pathlib import Path

ALLOWED_COLUMNS = [
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


def build_system_prompt() -> str:
    return (
        "You are given a single replicate event history from a Moran birth-death process. "
        "In this process, at each step one individual is chosen to reproduce proportional to fitness, "
        "and one individual is chosen to die uniformly at random. "
        "Mutants (type A) have relative fitness r compared to non-mutants (type B) which have fitness 1. "
        "Your task is to estimate r from the observable trace alone. "
        "You are not given r, i0, or any other hidden parameters. "
        "To estimate r, reason about how often mutants are chosen for birth relative to their "
        "current population fraction at each step — a mutant birth rate consistently above the "
        "mutant fraction suggests r > 1, below suggests r < 1, equal suggests r = 1. "
        "Do not round to the nearest 0.1 or 0.5. "
        "Report r as a continuous value to 2 decimal places. "
        "A value like 1.37 or 0.83 is expected and correct. "
        "Return valid JSON with exactly one key: estimated_r (a positive float to 2 decimal places). "
        "Do not output any other text outside the JSON."
    )


def _observable_csv_text(csv_path: Path) -> str:
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    for forbidden in ("true_i0", "true_r", "true_N", "meta_json"):
        if forbidden in (reader.fieldnames or []):
            raise ValueError(f"Forbidden ground-truth column found in trace CSV: {forbidden}")

    present_columns = [c for c in ALLOWED_COLUMNS if c in (reader.fieldnames or [])]
    output_lines = [",".join(present_columns)]
    for row in rows:
        output_lines.append(",".join(str(row[col]) for col in present_columns))
    return "\n".join(output_lines)


def build_user_prompt_from_csv(csv_path: str | Path) -> str:
    csv_path = Path(csv_path)
    return (
        f"Trace file: {csv_path.name}\n"
        "Below is the observable Moran-process event history in CSV format. "
        "Estimate the relative fitness r of the mutant type from this trace alone.\n\n"
        f"{_observable_csv_text(csv_path)}"
    )
