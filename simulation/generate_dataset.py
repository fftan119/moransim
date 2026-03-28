from __future__ import annotations

import csv
import random
from pathlib import Path
from .io import write_run_metadata_json, write_run_trace_csv
from .moran import simulate_moran_run


def _sample_r(rng: random.Random) -> float:
    # Rounded grid keeps scoring sensible while still varying the parameter.
    return round(rng.uniform(0.6, 1.8), 2)


def generate_dataset(
    *,
    num_experiments: int,
    replicates: int,
    seed: int | None,
    base_dir: str | Path,
    fixed_r: float | None = None,
    fixed_N: int | None = None,
    fixed_i0: int | None = None,
) -> Path:
    base_dir = Path(base_dir)
    raw_dir = base_dir / "data" / "raw"
    results_dir = base_dir / "data" / "results"
    raw_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    summary_rows: list[dict[str, str | int | float]] = []

    if fixed_N is not None and fixed_N < 2:
        raise ValueError("N must be at least 2.")
    if fixed_i0 is not None and fixed_i0 < 1:
        raise ValueError("i0 must be at least 1.")
    if fixed_N is not None and fixed_i0 is not None and fixed_i0 >= fixed_N:
        raise ValueError("i0 must satisfy 1 <= i0 < N.")
    if fixed_r is not None and fixed_r <= 0:
        raise ValueError("r must be positive.")

    for exp_idx in range(1, num_experiments + 1):
        true_r = fixed_r if fixed_r is not None else _sample_r(rng)
        true_N = fixed_N if fixed_N is not None else rng.randint(15, 25)
        true_i0 = fixed_i0 if fixed_i0 is not None else rng.randint(1, true_N - 1)
        for rep_idx in range(1, replicates + 1):
            run_id = f"exp{exp_idx:03d}_run{rep_idx:02d}"
            run = simulate_moran_run(r=true_r, N=true_N, i0=true_i0, run_id=run_id, rng=rng)
            raw_trace_path = write_run_trace_csv(run, raw_dir / f"{run_id}.csv")
            meta_path = write_run_metadata_json(run, raw_dir / f"{run_id}.meta.json")
            summary_rows.append(
                {
                    "run_id": run_id,
                    "trace_csv": str(raw_trace_path),
                    "meta_json": str(meta_path),
                    "true_r": true_r,
                    "true_N": true_N,
                    "true_i0": true_i0,
                    "num_events_full": len(run.steps),
                }
            )

    summary_path = results_dir / "dataset_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
        fieldnames=["run_id", "trace_csv", "meta_json", "true_r", "true_N", "true_i0", "num_events_full"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    return summary_path