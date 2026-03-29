from __future__ import annotations

"""
Grid runner for fixation probability classification.

Define your (r, i0) pairs in the GRID list below, set N, REPLICATES,
and MODEL, then run:

    python run_grid.py

This will simulate and send a classification batch for each point.
Once all batches are complete on OpenAI, run:

    python run_grid.py --fetch-parse-vote

to fetch, parse, and vote all points in one go. Then visualize:

    python visualize_classify.py
"""

import argparse
import subprocess
import sys
import json
from pathlib import Path

# ── Configure your grid here ─────────────────────────────────────────────────

GRID = [
    # (r,    i0)
    (1,   2),
    (2,  2),
    (1.5,    6),
    (0.75,   6),
    (1.45,   10),
    (1.25,  10),
    (1.3,    14),
    (0.6,   14),
    (1.25,    18),
    (0.15,   18),
]

N          = 20
REPLICATES = 20
MODEL      = "gpt-4o-mini"
SEED       = 42

# ─────────────────────────────────────────────────────────────────────────────

BATCH_IDS_FILE = Path("data/batches/classify_batch_job_ids_gpt-4o-mini.jsonl")
OUTPUTS_DIR    = Path("data/batches/outputs")
RESULTS_DIR    = Path("data/results")


def run(cmd: list[str]) -> None:
    print(f"\n>>> {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True)


def simulate_and_send() -> None:
    print(f"\n{'='*60}")
    print(f"Running {len(GRID)} grid points")
    print(f"N={N}, replicates={REPLICATES}, model={MODEL}")
    print(f"{'='*60}")

    for r, i0 in GRID:
        print(f"\n── Point: r={r}, i0={i0} ──")

        # Clean intermediate files from previous point (keep voted csv)
        run([sys.executable, "cleanup_classify.py"])
        run([sys.executable, "cleanup_run.py", "--delete-summary"])

        # Simulate
        run([
            sys.executable, "main.py", "simulate",
            "--num-experiments", "1",
            "--replicates", str(REPLICATES),
            "--N", str(N),
            "--r", str(r),
            "--i0", str(i0),
            "--seed", str(SEED),
        ])

        # Send classification batch
        run([
            sys.executable, "main.py", "classify-send",
            "--model", MODEL,
        ])

        print(f"✓ Batch sent for r={r}, i0={i0}")

    print(f"\n{'='*60}")
    print("All batches submitted. Wait for them to complete on OpenAI,")
    print("then run:  python run_grid.py --fetch-parse-vote")
    print(f"{'='*60}")


def fetch_parse_vote() -> None:
    print(f"\n{'='*60}")
    print("Fetching, parsing, and voting all completed batches")
    print(f"{'='*60}")

    # Fetch all completed batches
    run([
        sys.executable, "main.py", "classify-fetch",
        "--batch-ids-jsonl", str(BATCH_IDS_FILE),
        "--output-dir", str(OUTPUTS_DIR),
    ])

    # Find all downloaded classify output files
    output_files = sorted(OUTPUTS_DIR.glob("*_classify_output.jsonl"))
    if not output_files:
        print("\nNo completed output files found. Batches may still be in progress.")
        return

    print(f"\nFound {len(output_files)} output file(s) to process.")

    for output_jsonl in output_files:
        print(f"\n── Processing: {output_jsonl.name} ──")

        # Extract batch_id to find matching summary csv from batch ids file
        batch_id = output_jsonl.name.replace("_classify_output.jsonl", "")
        summary_csv = _find_summary_for_batch(batch_id)

        if summary_csv is None:
            print(f"  WARNING: Could not find summary CSV for {batch_id}, skipping.")
            continue

        parsed_csv = RESULTS_DIR / f"classify_parsed_{batch_id}.csv"

        run([
            sys.executable, "main.py", "classify-parse",
            "--output-jsonl", str(output_jsonl),
            "--parsed-csv", str(parsed_csv),
        ])

        run([
            sys.executable, "main.py", "classify-vote",
            "--parsed-csv", str(parsed_csv),
            "--summary-csv", str(summary_csv),
            "--voted-csv", str(RESULTS_DIR / "classify_voted.csv"),
        ])

    print(f"\n{'='*60}")
    print("All points processed. Run:  python visualize_classify.py")
    print(f"{'='*60}")


def _find_summary_for_batch(batch_id: str) -> Path | None:
    """Look up the summary CSV path recorded when the batch was sent."""
    if not BATCH_IDS_FILE.exists():
        return None
    with BATCH_IDS_FILE.open("r", encoding="utf-8") as f:
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
        description="Run simulate+send for all grid points, or fetch+parse+vote when done."
    )
    parser.add_argument(
        "--fetch-parse-vote", dest="fetch_parse_vote",
        action="store_true",
        help="Fetch completed batches and process them (run after batches complete)",
    )
    args = parser.parse_args()

    if args.fetch_parse_vote:
        fetch_parse_vote()
    else:
        simulate_and_send()


if __name__ == "__main__":
    main()