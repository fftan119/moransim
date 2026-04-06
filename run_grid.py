from __future__ import annotations

"""
Grid runner for fixation probability classification.

Define your (r, i0) pairs in GRID, set N, REPLICATES, MODEL, then run:

    python run_grid.py --group my_experiment

This simulates and sends a classification batch for each point, storing
everything under data/groups/my_experiment/.

Once batches are complete on OpenAI:

    python run_grid.py --group my_experiment --fetch-parse-vote

Then visualize:

    python visualize_classify.py --group my_experiment
"""

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

# ── Configure your grid here ─────────────────────────────────────────────────

GRID = [
    (1.6666, 1),   # rho = 0.4000
    (1.2883, 2),   # rho = 0.4000
    (1.5810, 2),   # rho = 0.6000
    (1.1754, 3),   # rho = 0.4000
    (1.3557, 3),   # rho = 0.6000
    (1.1161, 4),   # rho = 0.4000
    (1.2522, 4),   # rho = 0.6000
    (1.0766, 5),   # rho = 0.4000
    (1.1904, 5),   # rho = 0.6000
    (1.0466, 6),   # rho = 0.4000
    (1.1473, 6),   # rho = 0.6000
    (1.0218, 7),   # rho = 0.4000
    (1.1143, 7),   # rho = 0.6000
    (1.0000, 8),   # rho = 0.4000
    (1.0870, 8),   # rho = 0.6000
    (0.9798, 9),   # rho = 0.4000
    (1.0631, 9),   # rho = 0.6000
    (0.9603, 10),   # rho = 0.4000
    (1.0414, 10),   # rho = 0.6000
    (0.9406, 11),   # rho = 0.4000
    (1.0206, 11),   # rho = 0.6000
    (0.9200, 12),   # rho = 0.4000
    (1.0000, 12),   # rho = 0.6000
    (0.8975, 13),   # rho = 0.4000
    (0.9786, 13),   # rho = 0.6000
    (0.8716, 14),   # rho = 0.4000
    (0.9555, 14),   # rho = 0.6000
    (0.8401, 15),   # rho = 0.4000
    (0.9289, 15),   # rho = 0.6000
    (0.7986, 16),   # rho = 0.4000
    (0.8960, 16),   # rho = 0.6000
    (0.7376, 17),   # rho = 0.4000
    (0.8508, 17),   # rho = 0.6000
    (0.6325, 18),   # rho = 0.4000
    (0.7762, 18),   # rho = 0.6000
    (0.4000, 19),   # rho = 0.4000
    (0.6000, 19),   # rho = 0.6000
]
N          = 20
REPLICATES = 20
MODEL      = "gpt-4o-mini"
SEED       = 17

# ─────────────────────────────────────────────────────────────────────────────

GROUPS_DIR = Path("data/groups")


def group_dir(group: str) -> Path:
    return GROUPS_DIR / group


def group_paths(group: str) -> dict[str, Path]:
    base = group_dir(group)
    return {
        "base":       base,
        "raw":        base / "raw",
        "summaries":  base / "summaries",
        "batches":    base / "batches",
        "outputs":    base / "batches" / "outputs",
        "parsed":     base / "parsed",
        "results":    base / "results",
        "batch_ids":  base / "batches" / f"classify_batch_job_ids_{MODEL}.jsonl",
        "voted_csv":  base / "results" / "classify_voted.csv",
        "group_json": base / "group.json",
    }


def init_group(group: str) -> None:
    paths = group_paths(group)
    for key, p in paths.items():
        if key not in ("batch_ids", "voted_csv", "group_json"):
            p.mkdir(parents=True, exist_ok=True)

    gj = paths["group_json"]
    if not gj.exists():
        gj.write_text(json.dumps({
            "group": group,
            "N": N,
            "replicates": REPLICATES,
            "model": MODEL,
            "points": [],
        }, indent=2))
        print(f"Created group: {group}")
    else:
        print(f"Appending to existing group: {group}")


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


def make_summary_csv(r: float, i0: int, summary_path: Path) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["run_id", "trace_csv", "meta_json", "true_r", "true_N", "true_i0", "num_events_full"]
    rows = [
        {"run_id": f"exp001_run{rep:02d}", "trace_csv": "", "meta_json": "",
         "true_r": r, "true_N": N, "true_i0": i0, "num_events_full": ""}
        for rep in range(1, REPLICATES + 1)
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def simulate_and_send(group: str) -> None:
    init_group(group)
    paths = group_paths(group)

    print(f"\n{'='*60}")
    print(f"Group: {group}")
    print(f"Running {len(GRID)} grid points | N={N}, replicates={REPLICATES}, model={MODEL}")
    print(f"{'='*60}")

    for r, i0 in GRID:
        print(f"\n── Point: r={r}, i0={i0} ──")

        # Clean shared temp dirs before each point
        for d in [Path("data/raw"), Path("data/cropped")]:
            if d.exists():
                shutil.rmtree(d)
        for f in [Path("data/results/dataset_summary.csv"),
                  Path("data/results/classify_parsed.csv"),
                  Path(f"data/batches/classify_batch.jsonl")]:
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

        # Build summary CSV pointing at group raw dir
        summary_path = paths["summaries"] / f"summary_r{r}_i{i0}.csv"
        # Write summary with correct trace_csv paths
        summary_path.parent.mkdir(parents=True, exist_ok=True)
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

        # Send classification batch using group summary
        run([sys.executable, "main.py", "classify-send",
             "--summary-csv", str(summary_path),
             "--batch-jsonl", str(paths["batches"] / "classify_batch.jsonl"),
             "--model", MODEL,
        ])

        # Move batch IDs into group
        default_ids = Path(f"data/batches/classify_batch_job_ids_{MODEL}.jsonl")
        if default_ids.exists():
            with default_ids.open("r") as src, paths["batch_ids"].open("a") as dst:
                for line in src:
                    # Rewrite summary_csv path to point at group summary
                    record = json.loads(line)
                    record["summary_csv"] = str(summary_path)
                    record["r"] = r
                    record["i0"] = i0
                    dst.write(json.dumps(record) + "\n")
            default_ids.unlink()

        register_point(group, r, i0)
        print(f"✓ r={r}, i0={i0} registered to group '{group}'")

    print(f"\n{'='*60}")
    print(f"All batches submitted to group '{group}'.")
    print(f"Wait for OpenAI to complete, then run:")
    print(f"  python run_grid.py --group {group} --fetch-parse-vote")
    print(f"{'='*60}")


def fetch_parse_vote(group: str) -> None:
    paths = group_paths(group)

    print(f"\n{'='*60}")
    print(f"Fetching, parsing, and voting for group: {group}")
    print(f"{'='*60}")

    # Fetch into group outputs dir
    run([sys.executable, "main.py", "classify-fetch",
         "--batch-ids-jsonl", str(paths["batch_ids"]),
         "--output-dir", str(paths["outputs"]),
    ])

    output_files = sorted(paths["outputs"].glob("*_classify_output.jsonl"))
    if not output_files:
        print("\nNo completed output files found. Batches may still be in progress.")
        return

    print(f"\nFound {len(output_files)} output file(s) to process.")

    for output_jsonl in output_files:
        print(f"\n── Processing: {output_jsonl.name} ──")
        batch_id = output_jsonl.name.replace("_classify_output.jsonl", "")
        summary_csv = _find_summary(paths["batch_ids"], batch_id)

        if summary_csv is None:
            print(f"   WARNING: No summary found for {batch_id}, skipping.")
            continue

        parsed_csv = paths["parsed"] / f"classify_parsed_{batch_id}.csv"
        run([sys.executable, "main.py", "classify-parse",
             "--output-jsonl", str(output_jsonl),
             "--parsed-csv", str(parsed_csv),
        ])
        run([sys.executable, "main.py", "classify-vote",
             "--parsed-csv", str(parsed_csv),
             "--summary-csv", str(summary_csv),
             "--voted-csv", str(paths["voted_csv"]),
        ])

    print(f"\n{'='*60}")
    print(f"Done. Visualize with:")
    print(f"  python visualize_classify.py --group {group}")
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
        description="Run grid simulation and classification batches within a named group."
    )
    parser.add_argument("--group", required=True, help="Group name (e.g. boundary_sweep_N20)")
    parser.add_argument("--fetch-parse-vote", dest="fetch_parse_vote", action="store_true",
                        help="Fetch completed batches and process them")
    args = parser.parse_args()

    if args.fetch_parse_vote:
        fetch_parse_vote(args.group)
    else:
        simulate_and_send(args.group)


if __name__ == "__main__":
    main()