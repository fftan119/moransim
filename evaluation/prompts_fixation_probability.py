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
        "In this process, at each step one individual is chosen to reproduce (proportional to fitness) "
        "and one individual is chosen to die (uniformly at random). "
        "Mutants (type A) may have a different fitness than non-mutants (type B). "
        "The process runs until one type fixes (takes over the entire population). "
        "Your task is to estimate rho: the probability that the mutant type ultimately fixes "
        "in the population, based solely on the observable trace. "
        "rho reflects how advantaged or disadvantaged the mutants are — "
        "a value close to 1 means mutants almost certainly fix, close to 0 means they almost certainly die out. "
        "Do not use any formula. Instead, reason intuitively from the trace: "
        "consider how often mutants are chosen for birth relative to their population share, "
        "how the mutant count evolves over time, and whether mutants seem to have a fitness advantage. "
        "Return valid JSON with exactly one key: rho_estimated (a float between 0 and 1, to 2 decimal places). "
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
        "Estimate the fixation probability rho of the mutant type from this trace.\n\n"
        f"{_observable_csv_text(csv_path)}"
    )