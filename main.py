from __future__ import annotations

import argparse
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Moran-process simulation and GPT evaluation pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    sim = sub.add_parser("simulate", help="Generate hidden-truth Moran runs and observable trace CSVs")
    sim.add_argument("--num-experiments", type=int, default=10)
    sim.add_argument("--replicates", type=int, default=5)
    sim.add_argument("--seed", type=int, default=42)
    sim.add_argument("--N", type=int, default=None, help="Fix the population size N for every experiment")
    sim.add_argument("--r", type=float, default=None, help="Fix the relative fitness r for every experiment")
    sim.add_argument("--i0", type=int, default=None, help="Fix the initial mutant count i0 for every experiment")

    send = sub.add_parser("send", help="Submit the observable traces to the OpenAI Batch API")
    send.add_argument("--summary-csv", default=str(BASE_DIR / "data" / "results" / "dataset_summary.csv"))
    send.add_argument("--batch-jsonl", default=str(BASE_DIR / "data" / "batches" / "batch.jsonl"))
    send.add_argument("--model", default="gpt-4o-mini")

    fetch = sub.add_parser("fetch", help="Fetch completed batch outputs")
    fetch.add_argument("--batch-ids-jsonl", default=str(BASE_DIR / "data" / "batches" / "batch_job_ids_gpt-4o-mini.jsonl"))
    fetch.add_argument("--output-dir", default=str(BASE_DIR / "data" / "batches" / "outputs"))

    parse = sub.add_parser("parse", help="Parse one output JSONL into a flat CSV")
    parse.add_argument("--output-jsonl", required=True)
    parse.add_argument("--parsed-csv", default=str(BASE_DIR / "data" / "results" / "parsed_predictions.csv"))

    score = sub.add_parser("score", help="Compare parsed predictions to hidden truth")
    score.add_argument("--summary-csv", default=str(BASE_DIR / "data" / "results" / "dataset_summary.csv"))
    score.add_argument("--parsed-csv", default=str(BASE_DIR / "data" / "results" / "parsed_predictions.csv"))
    score.add_argument("--scored-csv", default=str(BASE_DIR / "data" / "results" / "scored_predictions.csv"))

    analyze = sub.add_parser("analyze", help="Summarize scored performance by crop type")
    analyze.add_argument("--scored-csv", default=str(BASE_DIR / "data" / "results" / "scored_predictions.csv"))

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "simulate":
        from simulation.generate_dataset import generate_dataset
        summary = generate_dataset(
            num_experiments=args.num_experiments,
            replicates=args.replicates,
            seed=args.seed,
            base_dir=BASE_DIR,
            fixed_r=args.r,
            fixed_N=args.N,
            fixed_i0=args.i0,
        )
        print(f"Dataset summary written to {summary}")

    elif args.command == "send":
        from evaluation.send_batch import send_batch
        batch_id = send_batch(args.summary_csv, args.batch_jsonl, model_name=args.model)
        print(f"Submitted batch job: {batch_id}")

    elif args.command == "fetch":
        from evaluation.fetch_batch import fetch_completed_batches
        outputs = fetch_completed_batches(args.batch_ids_jsonl, args.output_dir, verbose=True)
        print("Downloaded outputs:")
        for path in outputs:
            print(f"- {path}")

    elif args.command == "parse":
        from evaluation.parse_outputs import parse_batch_outputs
        parsed = parse_batch_outputs(args.output_jsonl, args.parsed_csv)
        print(f"Parsed predictions written to {parsed}")

    elif args.command == "score":
        from evaluation.score import score_predictions
        scored = score_predictions(args.summary_csv, args.parsed_csv, args.scored_csv)
        print(f"Scored predictions written to {scored}")

    elif args.command == "analyze":
        from evaluation.analyze import summarize_scores
        print(summarize_scores(args.scored_csv))


if __name__ == "__main__":
    main()
