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
        "Mutants (type A) may have a different fitness than non-mutants (type B). "
        "Your task is to classify whether the mutant type is less likely or more likely to ultimately "
        "fix (take over) the entire population. "
        "If you believe the mutant is less likely to fix than not, output O. "
        "If you believe the mutant is more likely to fix than not, output X. "
        "Base your judgment on the observable trace only: consider how often mutants are chosen "
        "for birth relative to their current population share, and how the mutant count evolves. "
        "Return valid JSON with exactly one key: label (either the string X or the string O). "
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
        "Classify whether the fixation probability rho is greater than 0.5 (X) "
        "or less than 0.5 (O).\n\n"
        f"{_observable_csv_text(csv_path)}"
    )