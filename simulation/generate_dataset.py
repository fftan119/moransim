from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any

from .crop import crop_trace
from .moran import MoranParams, run_moran_process


def generate_dataset(
    num_experiments: int,
    replicates_per_experiment: int,
    raw_dir: str | Path,
    cropped_dir: str | Path,
    summary_csv: str | Path,
    seed: int | None = None,
    r_min: float = 1.0,
    r_max: float = 1.5,
    N_min: int = 15,
    N_max: int = 25,
) -> Path:
    rng = random.Random(seed)
    raw_dir = Path(raw_dir)
    cropped_dir = Path(cropped_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    cropped_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = Path(summary_csv)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []

    for exp_idx in range(1, num_experiments + 1):
        N = rng.randint(N_min, N_max)
        params = MoranParams(
            r=round(rng.randint(int(r_min * 10), int(r_max * 10)) / 10.0, 1),
            N=N,
            i0=rng.randint(1, N - 1),
        )

        for rep_idx in range(1, replicates_per_experiment + 1):
            run_id = f"exp{exp_idx:03d}_run{rep_idx:02d}"
            run_seed = rng.randint(0, 10**9)
            metadata = run_moran_process(params, run_id=run_id, output_dir=raw_dir, seed=run_seed)
            raw_csv = Path(metadata["csv_path"])

            full_csv = cropped_dir / f"{run_id}__full.csv"
            prefix_csv = cropped_dir / f"{run_id}__prefix10.csv"
            suffix_csv = cropped_dir / f"{run_id}__suffix10.csv"
            stride_csv = cropped_dir / f"{run_id}__stride3.csv"

            crop_trace(raw_csv, full_csv, mode="full")
            crop_trace(raw_csv, prefix_csv, mode="prefix", k=10)
            crop_trace(raw_csv, suffix_csv, mode="suffix", k=10)
            crop_trace(raw_csv, stride_csv, mode="stride", stride=3)

            for crop_name, crop_path in {
                "full": full_csv,
                "prefix10": prefix_csv,
                "suffix10": suffix_csv,
                "stride3": stride_csv,
            }.items():
                summary_rows.append(
                    {
                        "experiment_id": f"exp{exp_idx:03d}",
                        "run_id": run_id,
                        "crop_name": crop_name,
                        "trace_csv": str(crop_path),
                        "true_r": params.r,
                        "true_N": params.N,
                        "true_i0": params.i0,
                        "seed": run_seed,
                        "steps": metadata["steps"],
                        "absorbing_state": metadata["absorbing_state"],
                    }
                )

    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()) if summary_rows else [
            "experiment_id", "run_id", "crop_name", "trace_csv", "true_r", "true_N", "true_i0",
            "seed", "steps", "absorbing_state"
        ])
        writer.writeheader()
        writer.writerows(summary_rows)

    manifest = summary_csv.with_suffix(".json")
    manifest.write_text(json.dumps({"rows": len(summary_rows), "summary_csv": str(summary_csv)}, indent=2), encoding="utf-8")
    return summary_csv
