# Moran Process Reverse-Inference Project

This version separates the codebase into two stages:

- `simulation/`: generate Moran-process traces and cropped variants.
- `evaluation/`: send traces to the OpenAI API, retrieve outputs, parse predictions, and score results.

## Folder layout

```text
moran_project/
├─ simulation/
├─ evaluation/
├─ data/
│  ├─ raw/
│  ├─ cropped/
│  ├─ batches/
│  └─ results/
├─ config/
└─ main.py
```

## Why this rewrite helps

1. Raw simulation is now separate from API evaluation.
2. Traces are saved with explicit headers and metadata.
3. The prompt no longer hardcodes `N = 20`; it tells the model to use the `N` column from the trace.
4. Scoring is based on parsed outputs against a summary table of ground truth.
5. Cropped traces are generated systematically (`full`, `prefix10`, `suffix10`, `stride3`).

## Quick start

Create traces:

```bash
python main.py simulate --num-experiments 10 --replicates 5 --seed 42
```

Submit a batch:

```bash
python main.py send --summary-csv data/results/dataset_summary.csv --batch-jsonl data/batches/batch.jsonl --model gpt-4o-mini
```

Fetch completed outputs:

```bash
python main.py fetch --batch-ids data/batches/batch_job_ids_gpt-4o-mini.jsonl --output-dir data/batches/outputs
```

Parse one output file:

```bash
python main.py parse --output-jsonl data/batches/outputs/<batch_id>_output.jsonl --parsed-csv data/results/parsed_predictions.csv
```

Score predictions:

```bash
python main.py score --summary-csv data/results/dataset_summary.csv --parsed-csv data/results/parsed_predictions.csv --scored-csv data/results/scored_predictions.csv
```

Analyze:

```bash
python main.py analyze --scored-csv data/results/scored_predictions.csv
```

## Notes for the paper

This structure is better aligned with an experimental paper because it separates:

- stochastic data generation,
- partial-observability / cropping design,
- LLM prompting,
- prediction parsing,
- evaluation.

That makes it easier to argue whether failure comes from code issues, prompt design, or true non-identifiability.
