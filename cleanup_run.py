from __future__ import annotations

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


def cleanup_data(
    base_dir: str | Path = "data",
    keep_plots: bool = True,
    keep_summary: bool = True,
    dry_run: bool = False,
) -> None:
    base = Path(base_dir)

    raw_dir = base / "raw"
    cropped_dir = base / "cropped"
    batches_dir = base / "batches"
    batch_outputs_dir = batches_dir / "outputs"
    results_dir = base / "results"
    plots_dir = results_dir / "plots"

    files_to_remove = [
        results_dir / "parsed_predictions.csv",
        results_dir / "scored_predictions.csv",
    ]

    if not keep_summary:
        files_to_remove.extend(
            [
                results_dir / "dataset_summary.csv",
                results_dir / "dataset_summary.json",
            ]
        )

    if dry_run:
        print("Dry run mode. The following items would be removed:\n")

        for d in [raw_dir, cropped_dir]:
            if d.exists():
                print(f"[DIR]  {d}")

        if batches_dir.exists():
            for f in batches_dir.glob("*.jsonl"):
                print(f"[FILE] {f}")

        if batch_outputs_dir.exists():
            for f in batch_outputs_dir.glob("*.jsonl"):
                print(f"[FILE] {f}")

        for f in files_to_remove:
            if f.exists():
                print(f"[FILE] {f}")

        if not keep_plots and plots_dir.exists():
            print(f"[DIR]  {plots_dir}")

        return

    remove_dir(raw_dir)
    remove_dir(cropped_dir)

    if batches_dir.exists():
        for f in batches_dir.glob("*.jsonl"):
            remove_file(f)

    if batch_outputs_dir.exists():
        for f in batch_outputs_dir.glob("*.jsonl"):
            remove_file(f)

    for f in files_to_remove:
        remove_file(f)

    if not keep_plots:
        remove_dir(plots_dir)

    print("\nCleanup complete.")
    print(f"Kept plots: {keep_plots}")
    print(f"Kept dataset summary: {keep_summary}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean up intermediate Moran experiment files while keeping final plots."
    )
    parser.add_argument(
        "--base-dir",
        default="data",
        help="Base data directory (default: data)",
    )
    parser.add_argument(
        "--delete-plots",
        action="store_true",
        help="Also delete plot images",
    )
    parser.add_argument(
        "--delete-summary",
        action="store_true",
        help="Also delete dataset_summary.csv and dataset_summary.json",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without deleting anything",
    )

    args = parser.parse_args()

    cleanup_data(
        base_dir=args.base_dir,
        keep_plots=not args.delete_plots,
        keep_summary=not args.delete_summary,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()