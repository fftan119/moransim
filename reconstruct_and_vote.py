from __future__ import annotations

"""
Reconstructs ground truth summary CSVs from known grid parameters and
votes all batches in order, appending results to classify_voted.csv.

Run once:
    python reconstruct_and_vote.py
"""

import csv
import json
from pathlib import Path

from evaluation.vote_fixation_probability import run_vote

# ── Your grid in the exact order batches were submitted ──────────────────────

GRID = [
    # (r,     i0)
    (1.0,     2),
    (2.0,     2),
    (1.5,     6),
    (0.75,    6),
    (1.45,   10),
    (1.25,   10),
    (1.3,    14),
    (0.6,    14),
    (1.25,   18),
    (0.15,   18),
]

N          = 20
REPLICATES = 20

# ── Paths ────────────────────────────────────────────────────────────────────

BATCH_IDS_FILE = Path("data/batches/classify_batch_job_ids_gpt-4o-mini.jsonl")
OUTPUTS_DIR    = Path("data/batches/outputs")
RESULTS_DIR    = Path("data/results")
VOTED_CSV      = RESULTS_DIR / "classify_voted.csv"

# ─────────────────────────────────────────────────────────────────────────────


def reconstruct_summary_csv(r: float, i0: int, N: int, replicates: int, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["run_id", "trace_csv", "meta_json", "true_r", "true_N", "true_i0", "num_events_full"]
    rows = []
    for rep in range(1, replicates + 1):
        run_id = f"exp001_run{rep:02d}"
        rows.append({
            "run_id": run_id,
            "trace_csv": "",
            "meta_json": "",
            "true_r": r,
            "true_N": N,
            "true_i0": i0,
            "num_events_full": "",
        })
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def load_batch_ids() -> list[str]:
    if not BATCH_IDS_FILE.exists():
        raise FileNotFoundError(f"Batch IDs file not found: {BATCH_IDS_FILE}")
    ids = []
    with BATCH_IDS_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ids.append(json.loads(line)["batch_job_id"])
    return ids


def _classify_rho(rho: float | None) -> str:
    if rho is None:
        return "O"
    return "X" if rho > 0.5 else "O"


def _parse_output(output_jsonl: Path, parsed_csv: Path) -> None:
    parsed_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with output_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            try:
                content = record["response"]["body"]["choices"][0]["message"]["content"]
            except Exception:
                content = None

            parsed = {}
            if content:
                try:
                    parsed = json.loads(content)
                except json.JSONDecodeError:
                    parsed = {}

            custom_id = record.get("custom_id", "")
            run_id = custom_id.removeprefix("classify__")
            exp_id = "_".join(run_id.split("_")[:-1]) if "_run" in run_id else run_id

            rho_raw = parsed.get("rho_estimated")
            try:
                rho_val = float(rho_raw) if rho_raw is not None else None
            except (TypeError, ValueError):
                rho_val = None

            label = _classify_rho(rho_val)

            rows.append({
                "custom_id": custom_id,
                "run_id": run_id,
                "exp_id": exp_id,
                "rho_estimated": rho_val,
                "label": label,
                "raw_content": content,
            })

    fieldnames = ["custom_id", "run_id", "exp_id", "rho_estimated", "label", "raw_content"]
    with parsed_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    batch_ids = load_batch_ids()

    if len(batch_ids) != len(GRID):
        print(f"WARNING: {len(batch_ids)} batch IDs found but {len(GRID)} grid points defined.")
        print("Proceeding with min(batch_ids, grid) pairs.\n")

    # Clear voted csv so we start fresh
    if VOTED_CSV.exists():
        VOTED_CSV.unlink()
        print(f"Cleared {VOTED_CSV}")

    pairs = list(zip(batch_ids, GRID))
    print(f"\nProcessing {len(pairs)} grid points:\n")

    for batch_id, (r, i0) in pairs:
        print(f"── r={r}, i0={i0}  →  {batch_id}")

        output_jsonl = OUTPUTS_DIR / f"{batch_id}_classify_output.jsonl"
        if not output_jsonl.exists():
            print(f"   SKIP: output file not found — fetch this batch first.\n")
            continue

        summary_csv = RESULTS_DIR / f"summary_r{r}_i{i0}.csv"
        reconstruct_summary_csv(r, i0, N, REPLICATES, summary_csv)

        parsed_csv = RESULTS_DIR / f"classify_parsed_{batch_id}.csv"
        _parse_output(output_jsonl, parsed_csv)

        run_vote(parsed_csv, summary_csv, VOTED_CSV)
        print(f"   ✓ Voted and appended to {VOTED_CSV}\n")

    print("Done. Run:  python visualize_classify.py")


if __name__ == "__main__":
    main()