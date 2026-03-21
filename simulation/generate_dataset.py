from __future__ import annotations

import csv
import random
from pathlib import Path
from .crop import make_crop_variants
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
) -> Path:
    base_dir = Path(base_dir)
    raw_dir = base_dir / "data" / "raw"
    cropped_dir = base_dir / "data" / "cropped"
    results_dir = base_dir / "data" / "results"
    raw_dir.mkdir(parents=True, exist_ok=True)
    cropped_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    summary_rows: list[dict[str, str | int | float]] = []

    for exp_idx in range(1, num_experiments + 1):
        true_r = _sample_r(rng)
        true_N = rng.randint(15, 25)
        true_i0 = rng.randint(1, true_N - 1)
        for rep_idx in range(1, replicates + 1):
            run_id = f"exp{exp_idx:03d}_run{rep_idx:02d}"
            run = simulate_moran_run(r=true_r, N=true_N, i0=true_i0, run_id=run_id, rng=rng)
            raw_trace_path = write_run_trace_csv(run, raw_dir / f"{run_id}.csv")
            meta_path = write_run_metadata_json(run, raw_dir / f"{run_id}.meta.json")
            crop_paths = make_crop_variants(raw_trace_path, cropped_dir)

            for crop_name, crop_path in crop_paths.items():
                summary_rows.append(
                    {
                        "run_id": run_id,
                        "crop_name": crop_name,
                        "trace_csv": str(crop_path),
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
            fieldnames=["run_id", "crop_name", "trace_csv", "meta_json", "true_r", "true_N", "true_i0", "num_events_full"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    return summary_path
