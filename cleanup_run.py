from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def remove_file(path: Path, verbose: bool = True) -> None:
    if path.exists() and path.is_file():
        path.unlink()
        if verbose:
            print(f"Deleted file: {path}")


def cleanup_classify(
    base_dir: str | Path = "data",
    reset_grid: bool = False,
    dry_run: bool = False,
) -> None:
    base = Path(base_dir)

    batches_dir = base / "batches"
    outputs_dir = batches_dir / "outputs"
    results_dir = base / "results"

    # Intermediate files cleared between every batch
    files_to_remove = [
        results_dir / "classify_parsed.csv",
        batches_dir / "classify_batch.jsonl",
    ]

    # Output JSONLs from previous classify fetch
    classify_output_files = list(outputs_dir.glob("*_classify_output.jsonl")) if outputs_dir.exists() else []

    # Only cleared if explicitly requested
    voted_csv = results_dir / "classify_voted.csv"

    if dry_run:
        print("Dry run mode. The following items would be removed:\n")
        for f in files_to_remove:
            if f.exists():
                print(f"[FILE] {f}")
        for f in classify_output_files:
            print(f"[FILE] {f}")
        if reset_grid and voted_csv.exists():
            print(f"[FILE] {voted_csv}  (grid reset)")
        return

    for f in files_to_remove:
        remove_file(f)

    for f in classify_output_files:
        remove_file(f)

    if reset_grid:
        remove_file(voted_csv)
        print("Grid reset: classify_voted.csv deleted.")

    print("\nClassification cleanup complete.")
    print(f"Grid data (classify_voted.csv) kept: {not reset_grid}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean up intermediate classification files between batches. "
                    "Keeps classify_voted.csv by default so grid points accumulate."
    )
    parser.add_argument(
        "--base-dir",
        default="data",
        help="Base data directory (default: data)",
    )
    parser.add_argument(
        "--reset-grid",
        action="store_true",
        help="Also delete classify_voted.csv (wipes all accumulated grid points)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without deleting anything",
    )

    args = parser.parse_args()
    cleanup_classify(
        base_dir=args.base_dir,
        reset_grid=args.reset_grid,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()