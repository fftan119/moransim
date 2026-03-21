# Moran reverse-inference pipeline

This project does four things cleanly:

1. Simulates Moran-process runs with hidden true parameters `(r, i0, N)`.
2. Saves **observable-only** trace CSVs for GPT.
3. Sends those traces to the OpenAI Batch API to estimate `r` and `i0`.
4. Parses and scores GPT's predictions against the hidden truth stored separately in metadata and the dataset summary.

## Important design rule

The trace CSV files sent to GPT **must not** contain `true_i0`, `true_r`, or other answer columns.
The prompt layer enforces this and raises an error if such columns appear in a trace CSV.

## Commands

Generate data:

```bash
python main.py simulate --num-experiments 10 --replicates 5 --seed 42
```

Submit a batch:

```bash
python main.py send --summary-csv data/results/dataset_summary.csv --batch-jsonl data/batches/batch.jsonl --model gpt-4o-mini
```

Fetch completed outputs:

```bash
python main.py fetch
```

Parse one downloaded output JSONL:

```bash
python main.py parse --output-jsonl data/batches/outputs/<batch_id>_output.jsonl
```

Score predictions:

```bash
python main.py score
```

Summarize by crop type:

```bash
python main.py analyze
```


Fixed-parameter simulation example:

```bash
python main.py simulate --num-experiments 3 --replicates 5 --N 20 --r 1.25 --i0 4 --seed 42
```

You can also fix only some parameters and let the others be sampled. For example:

```bash
python main.py simulate --num-experiments 10 --replicates 3 --N 20
```
