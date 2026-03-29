from __future__ import annotations

"""
Run this before starting a new grid run to wipe all classification
intermediates and accumulated results.

    python pregrid_cleanup.py

Use --dry-run to preview what would be deleted without deleting anything.
"""

import argparse
import shutil
from pathlib import Path


def remove_file(path: Path, verbose: bool = True) -> None:
    if path.exists() and path.is_file():
        path.unlink()
        if verbose:
            print(f"Deleted file: {path}")


def remove_dir(path: Path, verbose: bool = True) -> None:
    if path.exists() and path.is_dir():
        shutil.rmtree(path)
        if verbose:
            print(f"Deleted directory: {path}")


def pregrid_cleanup(base_dir: str | Path = "data", dry_run: bool = False) -> None:
    base = Path(base_dir)

    batches_dir   = base / "batches"
    outputs_dir   = batches_dir / "outputs"
    results_dir   = base / "results"
    summaries_dir = results_dir / "summaries"
    raw_dir       = base / "raw"
    cropped_dir   = base / "cropped"

    files_to_remove = [
        # Batch IDs
        batches_dir / "classify_batch_job_ids_gpt-4o-mini.jsonl",
        batches_dir / "classify_batch.jsonl",
        # Accumulated results
        results_dir / "classify_voted.csv",
        results_dir / "classify_parsed.csv",
        results_dir / "dataset_summary.csv",
    ]

    glob_patterns = [
        (results_dir, "classify_parsed_batch_*.csv"),
        (outputs_dir, "*_classify_output.jsonl"),
        (outputs_dir, "*_classify_errors.jsonl"),
    ]

    dirs_to_remove = [summaries_dir, raw_dir, cropped_dir]

    if dry_run:
        print("Dry run — the following would be deleted:\n")
        for f in files_to_remove:
            if f.exists():
                print(f"  [FILE] {f}")
        for parent, pattern in glob_patterns:
            for f in parent.glob(pattern):
                print(f"  [FILE] {f}")
        for d in dirs_to_remove:
            if d.exists():
                print(f"  [DIR]  {d}")
        return

    for f in files_to_remove:
        remove_file(f)

    for parent, pattern in glob_patterns:
        for f in parent.glob(pattern):
            remove_file(f)

    for d in dirs_to_remove:
        remove_dir(d)

    print("\nPre-grid cleanup complete. Ready to run:  python run_grid.py")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Wipe all classification intermediates before a new grid run."
    )
    parser.add_argument(
        "--base-dir",
        default="data",
        help="Base data directory (default: data)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be deleted without deleting anything",
    )
    args = parser.parse_args()
    pregrid_cleanup(base_dir=args.base_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()