from __future__ import annotations

from dataclasses import dataclass, asdict
import csv
import json
import random
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MoranParams:
    r: float
    N: int
    i0: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _validate(params: MoranParams) -> None:
    if not (params.r > 0):
        raise ValueError("Relative fitness r must be positive.")
    if not (params.N >= 2):
        raise ValueError("Population size N must be at least 2.")
    if not (0 < params.i0 < params.N):
        raise ValueError("Initial mutant count i0 must satisfy 0 < i0 < N.")


def run_moran_process(
    params: MoranParams,
    run_id: str,
    output_dir: str | Path,
    seed: int | None = None,
) -> dict[str, Any]:
    """Simulate one Moran process trajectory and save a structured CSV.

    Each row contains a single birth-death event plus mutant counts before/after the event.
    This makes downstream analysis easier and avoids header/merging bugs.
    """
    _validate(params)
    rng = random.Random(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    population = ["A"] * params.i0 + ["B"] * (params.N - params.i0)
    i = params.i0
    step = 0
    rows: list[dict[str, Any]] = []

    while 0 < i < params.N:
        i_before = i
        weights = [params.r if individual == "A" else 1.0 for individual in population]
        birth_index = rng.choices(range(params.N), weights=weights, k=1)[0]
        birth_type = population[birth_index]

        death_index = rng.randrange(params.N)
        death_type = population[death_index]

        population[death_index] = birth_type
        i = population.count("A")

        rows.append(
            {
                "run_id": run_id,
                "step": step,
                "birth_index": birth_index,
                "birth_type": birth_type,
                "death_index": death_index,
                "death_type": death_type,
                "event": f"{birth_index}{birth_type}:{death_index}{death_type}",
                "mutants_before": i_before,
                "mutants_after": i,
                "N": params.N,
                "true_r": params.r,
                "true_i0": params.i0,
            }
        )
        step += 1

    csv_path = output_dir / f"{run_id}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [
            "run_id", "step", "birth_index", "birth_type", "death_index", "death_type",
            "event", "mutants_before", "mutants_after", "N", "true_r", "true_i0"
        ])
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "run_id": run_id,
        **params.to_dict(),
        "steps": len(rows),
        "absorbing_state": i,
        "seed": seed,
        "csv_path": str(csv_path),
    }
    meta_path = output_dir / f"{run_id}.meta.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
