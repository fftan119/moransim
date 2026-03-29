from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path


def compute_true_rho(r: float, i0: int, N: int) -> float:
    """Moran fixation probability: rho = (1 - (1/r)^i0) / (1 - (1/r)^N)."""
    if r == 1.0:
        # Neutral case: fixation probability is i0/N
        return i0 / N
    inv_r = 1.0 / r
    return (1.0 - inv_r ** i0) / (1.0 - inv_r ** N)


def majority_vote(labels: list[str]) -> str:
    """Return X or O by majority. Ties go to O (conservative)."""
    counts = Counter(labels)
    if counts.get("X", 0) > counts.get("O", 0):
        return "X"
    return "O"


def run_vote(
    parsed_csv: str | Path,
    summary_csv: str | Path,
    voted_csv: str | Path,
) -> Path:
    parsed_csv = Path(parsed_csv)
    summary_csv = Path(summary_csv)
    voted_csv = Path(voted_csv)
    voted_csv.parent.mkdir(parents=True, exist_ok=True)

    # Load ground truth parameters keyed by run_id
    truth: dict[str, dict[str, str]] = {}
    with summary_csv.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            truth[row["run_id"]] = row

    # Group per-replicate labels by exp_id
    exp_labels: dict[str, list[str]] = {}
    exp_run_ids: dict[str, list[str]] = {}

    with parsed_csv.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            exp_id = row["exp_id"]
            label = row.get("label", "").strip().upper()
            if label not in ("X", "O"):
                label = "O"  # treat unparseable as O
            exp_labels.setdefault(exp_id, []).append(label)
            exp_run_ids.setdefault(exp_id, []).append(row["run_id"])

    result_rows: list[dict] = []
    for exp_id, labels in exp_labels.items():
        # Get parameters from the first run of this experiment
        first_run_id = exp_run_ids[exp_id][0]
        gold = truth.get(first_run_id, {})

        true_r = float(gold.get("true_r", 0))
        true_N = int(gold.get("true_N", 0))
        true_i0 = int(gold.get("true_i0", 0))
        rho_true = compute_true_rho(true_r, true_i0, true_N) if true_N > 0 else None
        true_label = "X" if (rho_true is not None and rho_true > 0.5) else "O"

        vote = majority_vote(labels)
        x_count = labels.count("X")
        o_count = labels.count("O")
        correct = int(vote == true_label)

        result_rows.append({
            "exp_id": exp_id,
            "true_r": true_r,
            "true_i0": true_i0,
            "true_N": true_N,
            "rho_true": round(rho_true, 6) if rho_true is not None else "",
            "true_label": true_label,
            "majority_vote": vote,
            "correct": correct,
            "x_count": x_count,
            "o_count": o_count,
            "n_replicates": len(labels),
            "per_replicate_labels": ",".join(labels),
        })

    fieldnames = [
        "exp_id", "true_r", "true_i0", "true_N",
        "rho_true", "true_label", "majority_vote", "correct",
        "x_count", "o_count", "n_replicates", "per_replicate_labels",
    ]
    with voted_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(result_rows)

    # Print summary
    n = len(result_rows)
    accuracy = sum(r["correct"] for r in result_rows) / n if n > 0 else 0
    print(f"\nClassification summary")
    print(f"- Experiments: {n}")
    print(f"- Accuracy (majority vote): {accuracy:.3f}")
    for row in result_rows:
        print(
            f"  {row['exp_id']}: true={row['true_label']} (rho={row['rho_true']}) "
            f"vote={row['majority_vote']} [{row['x_count']}X / {row['o_count']}O] "
            f"{'✓' if row['correct'] else '✗'}"
        )

    return voted_csv
