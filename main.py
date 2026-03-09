from __future__ import annotations

import argparse
from pathlib import Path

from simulation.generate_dataset import generate_dataset
from evaluation.send_batch import send_batch
from evaluation.fetch_batch import fetch_completed_batches
from evaluation.parse_outputs import parse_batch_outputs
from evaluation.score import score_predictions
from evaluation.analyze import summarize_scores


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Moran process simulation and LLM evaluation pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    sim = sub.add_parser("simulate", help="Generate raw and cropped Moran traces")
    sim.add_argument("--num-experiments", type=int, default=10)
    sim.add_argument("--replicates", type=int, default=5)
    sim.add_argument("--seed", type=int, default=42)

    send = sub.add_parser("send", help="Build and submit a batch job")
    send.add_argument("--summary-csv", default="data/results/dataset_summary.csv")
    send.add_argument("--batch-jsonl", default="data/batches/batch.jsonl")
    send.add_argument("--model", default=None)

    fetch = sub.add_parser("fetch", help="Download completed batch outputs")
    fetch.add_argument("--batch-ids", default="data/batches/batch_job_ids_gpt-4o-mini.jsonl")
    fetch.add_argument("--output-dir", default="data/batches/outputs")

    parse = sub.add_parser("parse", help="Parse one batch output JSONL into CSV")
    parse.add_argument("--output-jsonl", required=True)
    parse.add_argument("--parsed-csv", default="data/results/parsed_predictions.csv")

    score = sub.add_parser("score", help="Score parsed predictions against ground truth")
    score.add_argument("--summary-csv", default="data/results/dataset_summary.csv")
    score.add_argument("--parsed-csv", default="data/results/parsed_predictions.csv")
    score.add_argument("--scored-csv", default="data/results/scored_predictions.csv")

    analyze = sub.add_parser("analyze", help="Print a small score summary")
    analyze.add_argument("--scored-csv", default="data/results/scored_predictions.csv")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    root = Path(__file__).resolve().parent

    if args.command == "simulate":
        summary_path = generate_dataset(
            num_experiments=args.num_experiments,
            replicates_per_experiment=args.replicates,
            raw_dir=root / "data/raw",
            cropped_dir=root / "data/cropped",
            summary_csv=root / "data/results/dataset_summary.csv",
            seed=args.seed,
        )
        print(f"Dataset summary written to {summary_path}")

    elif args.command == "send":
        batch_id = send_batch(root / args.summary_csv, root / args.batch_jsonl, model_name=args.model)
        print(f"Submitted batch job: {batch_id}")

    elif args.command == "fetch":
        outputs = fetch_completed_batches(root / args.batch_ids, root / args.output_dir)
        print("Downloaded outputs:")
        for path in outputs:
            print(path)

    elif args.command == "parse":
        parsed = parse_batch_outputs(root / args.output_jsonl, root / args.parsed_csv)
        print(f"Parsed predictions written to {parsed}")

    elif args.command == "score":
        scored = score_predictions(root / args.summary_csv, root / args.parsed_csv, root / args.scored_csv)
        print(f"Scored predictions written to {scored}")

    elif args.command == "analyze":
        print(summarize_scores(root / args.scored_csv))


if __name__ == "__main__":
    main()
