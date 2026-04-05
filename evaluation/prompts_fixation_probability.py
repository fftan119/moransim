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
        "You are given a partial event history from a Moran birth-death process. "
        "At each step, one individual reproduces proportional to fitness and one dies uniformly at random. "
        "Mutants (type A) have relative fitness r versus non-mutants (type B) which have fitness 1. "
        "r may be above, equal to, or below 1 with equal prior probability — do not assume any direction. "
        "The trace has been truncated and does not show the final outcome. "
        "Your task is to decide whether the fixation probability rho of the mutant type is above or below 0.5. "
        "To do this, examine the birth events in the trace: "
        "At each step, compute whether the observed birth type matches what would be expected "
        "under neutrality given the current mutant fraction. "
        "Steps where the same individual is chosen for both birth and death carry no information — ignore these null events. "
        "Integrate this evidence across all non-null steps to form a judgment. "
        "Both outcomes are equally plausible a priori. "
        "A single trace is noisy — base your decision on the overall pattern, not any single event. "
        "Return valid JSON with exactly one key: label (either the string X if rho > 0.5, or O if rho < 0.5). "
        "Do not output any other text outside the JSON."
    )


def _truncate_rows(rows: list[dict], fraction: float = 0.20) -> list[dict]:
    """Remove the last `fraction` of rows to hide the absorption outcome."""
    n = len(rows)
    keep = max(1, n - int(n * fraction))
    return rows[:keep]


def _observable_csv_text(csv_path: Path) -> str:
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    for forbidden in ("true_i0", "true_r", "true_N", "meta_json"):
        if forbidden in (reader.fieldnames or []):
            raise ValueError(f"Forbidden ground-truth column found in trace CSV: {forbidden}")

    rows = _truncate_rows(rows, fraction=0.20)

    present_columns = [c for c in ALLOWED_COLUMNS if c in (reader.fieldnames or [])]
    output_lines = [",".join(present_columns)]
    for row in rows:
        output_lines.append(",".join(str(row[col]) for col in present_columns))
    return "\n".join(output_lines)


def build_user_prompt_from_csv(csv_path: str | Path) -> str:
    csv_path = Path(csv_path)
    return (
        f"Trace file: {csv_path.name}\n"
        "Below is a partial Moran-process event history in CSV format. "
        "Decide whether the fixation probability rho of the mutant type is above or below 0.5.\n\n"
        f"{_observable_csv_text(csv_path)}"
    )