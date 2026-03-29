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
import shutil
import subprocess
import sys
import json
from pathlib import Path

# ── Configure your grid here ─────────────────────────────────────────────────

GRID = [
    (1.6666, 1),   # rho = 0.4
    (1.2883, 2),   # rho = 0.4
    (1.581 , 2),   # rho = 0.6
    (1.1754, 3),   # rho = 0.4
    (1.3557, 3),   # rho = 0.6
    (1.1161, 4),   # rho = 0.4
    (1.2522, 4),   # rho = 0.6
    (1.0766, 5),   # rho = 0.4
    (1.1904, 5),   # rho = 0.6
    (1.0466, 6),   # rho = 0.4
    (1.1473, 6),   # rho = 0.6
    (1.0218, 7),   # rho = 0.4
    (1.1143, 7),   # rho = 0.6
    (1.0   , 8),   # rho = 0.4
    (1.087 , 8),   # rho = 0.6
    (0.9798, 9),   # rho = 0.4
    (1.0631, 9),   # rho = 0.6
    (0.9603, 10),  # rho = 0.4
    (1.0414, 10),  # rho = 0.6
    (0.9406, 11),  # rho = 0.4
    (1.0206, 11),  # rho = 0.6
    (0.92  , 12),  # rho = 0.4
    (1.0   , 12),  # rho = 0.6
    (0.8975, 13),  # rho = 0.4
    (0.9786, 13),  # rho = 0.6
    (0.8716, 14),  # rho = 0.4
    (0.9555, 14),  # rho = 0.6
    (0.8401, 15),  # rho = 0.4
    (0.9289, 15),  # rho = 0.6
    (0.7986, 16),  # rho = 0.4
    (0.896 , 16),  # rho = 0.6
    (0.7376, 17),  # rho = 0.4
    (0.8508, 17),  # rho = 0.6
    (0.6325, 18),  # rho = 0.4
    (0.7762, 18),  # rho = 0.6
    (0.4   , 19),  # rho = 0.4
    (0.6   , 19),  # rho = 0.6
]

N          = 20
REPLICATES = 20
MODEL      = "gpt-4o-mini"
SEED       = 42

# ─────────────────────────────────────────────────────────────────────────────

BATCHES_DIR    = Path("data/batches")
OUTPUTS_DIR    = Path("data/batches/outputs")
RESULTS_DIR    = Path("data/results")
SUMMARIES_DIR  = RESULTS_DIR / "summaries"
BATCH_IDS_FILE = BATCHES_DIR / f"classify_batch_job_ids_{MODEL}.jsonl"


def run(cmd: list[str]) -> None:
    print(f"\n>>> {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def _cleanup_intermediates() -> None:
    """Remove per-point intermediate files but keep batch IDs and voted CSV."""
    for f in [
        RESULTS_DIR / "classify_parsed.csv",
        RESULTS_DIR / "dataset_summary.csv",
        BATCHES_DIR / "classify_batch.jsonl",
    ]:
        if f.exists():
            f.unlink()
    # Remove raw and cropped data dirs
    for d in [Path("data/raw"), Path("data/cropped")]:
        if d.exists():
            shutil.rmtree(d)


def simulate_and_send() -> None:
    SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Running {len(GRID)} grid points")
    print(f"N={N}, replicates={REPLICATES}, model={MODEL}")
    print(f"{'='*60}")

    for r, i0 in GRID:
        print(f"\n── Point: r={r}, i0={i0} ──")

        _cleanup_intermediates()

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

        # Save a permanent copy of this point's summary CSV
        src = RESULTS_DIR / "dataset_summary.csv"
        dst = SUMMARIES_DIR / f"summary_r{r}_i{i0}.csv"
        shutil.copy(src, dst)
        print(f"   Saved summary → {dst}")

        # Send classification batch, pointing at the permanent summary
        run([
            sys.executable, "main.py", "classify-send",
            "--summary-csv", str(dst),
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

    run([
        sys.executable, "main.py", "classify-fetch",
        "--batch-ids-jsonl", str(BATCH_IDS_FILE),
        "--output-dir", str(OUTPUTS_DIR),
    ])

    output_files = sorted(OUTPUTS_DIR.glob("*_classify_output.jsonl"))
    if not output_files:
        print("\nNo completed output files found. Batches may still be in progress.")
        return

    print(f"\nFound {len(output_files)} output file(s) to process.")

    for output_jsonl in output_files:
        print(f"\n── Processing: {output_jsonl.name} ──")

        batch_id = output_jsonl.name.replace("_classify_output.jsonl", "")
        summary_csv = _find_summary_for_batch(batch_id)

        if summary_csv is None:
            print(f"   WARNING: Could not find summary CSV for {batch_id}, skipping.")
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
    """Look up the permanent summary CSV recorded when the batch was sent."""
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