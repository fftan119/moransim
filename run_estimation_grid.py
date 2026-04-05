from __future__ import annotations

"""
Grid runner for r estimation pipeline.

Define your (r, i0) pairs in GRID, set N, REPLICATES, MODEL, then run:

    python run_estimation_grid.py --group my_experiment

Once batches complete on OpenAI:

    python run_estimation_grid.py --group my_experiment --fetch-parse-score

Then visualize:

    python visualize_estimation.py --group my_experiment
"""

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

# ── Configure your grid here ─────────────────────────────────────────────────
NUM = 7

GRID = [
    # (r,    i0)
    (0.25,   NUM),
    (0.5,    NUM),
    (0.75,   NUM),
    (1,      NUM),
    (1.25,   NUM),
    (1.5,    NUM),
    (1.75,   NUM),
]

N          = 20
REPLICATES = 20
MODEL      = "gpt-4o-mini"
SEED       = 42

# ─────────────────────────────────────────────────────────────────────────────

GROUPS_DIR = Path("data/groups")


def group_paths(group: str) -> dict[str, Path]:
    base = GROUPS_DIR / group
    return {
        "base":      base,
        "raw":       base / "raw",
        "summaries": base / "summaries",
        "batches":   base / "batches",
        "outputs":   base / "batches" / "outputs",
        "parsed":    base / "parsed",
        "results":   base / "results",
        "batch_ids": base / "batches" / f"estimation_batch_job_ids_{MODEL}.jsonl",
        "scored_csv":base / "results" / "estimation_scored.csv",
        "group_json":base / "group.json",
    }


def init_group(group: str) -> None:
    paths = group_paths(group)
    for key, p in paths.items():
        if key not in ("batch_ids", "scored_csv", "group_json"):
            p.mkdir(parents=True, exist_ok=True)

    gj = paths["group_json"]
    if not gj.exists():
        gj.write_text(json.dumps({
            "group": group,
            "pipeline": "estimation",
            "N": N,
            "replicates": REPLICATES,
            "model": MODEL,
            "points": [],
        }, indent=2))
        print(f"Created estimation group: {group}")
    else:
        print(f"Appending to existing estimation group: {group}")


def register_point(group: str, r: float, i0: int) -> None:
    paths = group_paths(group)
    gj = paths["group_json"]
    meta = json.loads(gj.read_text())
    entry = {"r": r, "i0": i0}
    if entry not in meta["points"]:
        meta["points"].append(entry)
    gj.write_text(json.dumps(meta, indent=2))


def run(cmd: list[str]) -> None:
    print(f"\n>>> {' '.join(str(c) for c in cmd)}")
    subprocess.run(cmd, check=True)


def simulate_and_send(group: str) -> None:
    init_group(group)
    paths = group_paths(group)

    print(f"\n{'='*60}")
    print(f"Estimation group: {group}")
    print(f"Running {len(GRID)} grid points | N={N}, replicates={REPLICATES}, model={MODEL}")
    print(f"{'='*60}")

    for r, i0 in GRID:
        print(f"\n── Point: r={r}, i0={i0} ──")

        # Clean shared temp dirs
        for d in [Path("data/raw"), Path("data/cropped")]:
            if d.exists():
                shutil.rmtree(d)
        for f in [Path("data/results/dataset_summary.csv")]:
            if f.exists():
                f.unlink()

        # Simulate
        run([sys.executable, "main.py", "simulate",
             "--num-experiments", "1",
             "--replicates", str(REPLICATES),
             "--N", str(N),
             "--r", str(r),
             "--i0", str(i0),
             "--seed", str(SEED),
        ])

        # Copy raw traces into group
        raw_point_dir = paths["raw"] / f"r{r}_i{i0}"
        raw_point_dir.mkdir(parents=True, exist_ok=True)
        for csv_file in Path("data/raw").glob("*.csv"):
            shutil.copy(csv_file, raw_point_dir / csv_file.name)

        # Write summary CSV with correct trace paths
        summary_path = paths["summaries"] / f"summary_r{r}_i{i0}.csv"
        fieldnames = ["run_id", "trace_csv", "meta_json", "true_r", "true_N", "true_i0", "num_events_full"]
        rows = []
        for rep in range(1, REPLICATES + 1):
            run_id = f"exp001_run{rep:02d}"
            trace = raw_point_dir / f"{run_id}.csv"
            rows.append({"run_id": run_id, "trace_csv": str(trace),
                         "meta_json": "", "true_r": r, "true_N": N,
                         "true_i0": i0, "num_events_full": ""})
        with summary_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        # Send estimation batch
        batch_jsonl = paths["batches"] / "estimation_batch.jsonl"
        run([sys.executable, "main.py", "estimation-send",
             "--summary-csv", str(summary_path),
             "--batch-jsonl", str(batch_jsonl),
             "--model", MODEL,
        ])

        # Move batch IDs into group
        default_ids = Path(f"data/batches/estimation_batch_job_ids_{MODEL}.jsonl")
        if default_ids.exists():
            with default_ids.open("r") as src, paths["batch_ids"].open("a") as dst:
                for line in src:
                    record = json.loads(line)
                    record["summary_csv"] = str(summary_path)
                    record["r"] = r
                    record["i0"] = i0
                    dst.write(json.dumps(record) + "\n")
            default_ids.unlink()

        register_point(group, r, i0)
        print(f"✓ r={r}, i0={i0} registered to estimation group '{group}'")

    print(f"\n{'='*60}")
    print(f"All batches submitted to group '{group}'.")
    print(f"Wait for OpenAI to complete, then run:")
    print(f"  python run_estimation_grid.py --group {group} --fetch-parse-score")
    print(f"{'='*60}")


def fetch_parse_score(group: str) -> None:
    paths = group_paths(group)

    print(f"\n{'='*60}")
    print(f"Fetching, parsing, and scoring for estimation group: {group}")
    print(f"{'='*60}")

    run([sys.executable, "main.py", "estimation-fetch",
         "--batch-ids-jsonl", str(paths["batch_ids"]),
         "--output-dir", str(paths["outputs"]),
    ])

    output_files = sorted(paths["outputs"].glob("*_estimation_output.jsonl"))
    if not output_files:
        print("\nNo completed output files found. Batches may still be in progress.")
        return

    print(f"\nFound {len(output_files)} output file(s) to process.")

    for output_jsonl in output_files:
        print(f"\n── Processing: {output_jsonl.name} ──")
        batch_id = output_jsonl.name.replace("_estimation_output.jsonl", "")
        summary_csv = _find_summary(paths["batch_ids"], batch_id)

        if summary_csv is None:
            print(f"   WARNING: No summary found for {batch_id}, skipping.")
            continue

        parsed_csv = paths["parsed"] / f"estimation_parsed_{batch_id}.csv"
        run([sys.executable, "main.py", "estimation-parse",
             "--output-jsonl", str(output_jsonl),
             "--parsed-csv", str(parsed_csv),
        ])
        run([sys.executable, "main.py", "estimation-score",
             "--parsed-csv", str(parsed_csv),
             "--summary-csv", str(summary_csv),
             "--scored-csv", str(paths["scored_csv"]),
        ])

    print(f"\n{'='*60}")
    print(f"Done. Visualize with:")
    print(f"  python visualize_estimation.py --group {group}")
    print(f"{'='*60}")


def _find_summary(batch_ids_file: Path, batch_id: str) -> Path | None:
    if not batch_ids_file.exists():
        return None
    with batch_ids_file.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("batch_job_id") == batch_id:
                p = Path(record["summary_csv"])
                if p.exists():
                    return p
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run grid simulation and r estimation batches within a named group."
    )
    parser.add_argument("--group", required=True, help="Group name (e.g. r_estimation_N20)")
    parser.add_argument("--fetch-parse-score", dest="fetch_parse_score", action="store_true",
                        help="Fetch completed batches and process them")
    args = parser.parse_args()

    if args.fetch_parse_score:
        fetch_parse_score(args.group)
    else:
        simulate_and_send(args.group)


if __name__ == "__main__":
    main()
