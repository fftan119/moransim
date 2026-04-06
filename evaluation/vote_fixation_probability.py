from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path


def compute_true_rho(r: float, i0: int, N: int) -> float:
    """Moran fixation probability: rho = (1 - (1/r)^i0) / (1 - (1/r)^N)."""
    if r == 1.0:
        return i0 / N
    inv_r = 1.0 / r
    return (1.0 - inv_r ** i0) / (1.0 - inv_r ** N)


def majority_vote(labels: list[str]) -> str:
    """Return X, O, or T (tie) based on majority."""
    counts = Counter(labels)
    x = counts.get("X", 0)
    o = counts.get("O", 0)
    if x > o:
        return "X"
    elif o > x:
        return "O"
    else:
        return "T"  # tie


def run_vote(
    parsed_csv: str | Path,
    summary_csv: str | Path,
    voted_csv: str | Path,
) -> Path:
    parsed_csv = Path(parsed_csv)
    summary_csv = Path(summary_csv)
    voted_csv = Path(voted_csv)
    voted_csv.parent.mkdir(parents=True, exist_ok=True)

    truth: dict[str, dict[str, str]] = {}
    with summary_csv.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            truth[row["run_id"]] = row

    exp_labels: dict[str, list[str]] = {}
    exp_run_ids: dict[str, list[str]] = {}

    with parsed_csv.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            exp_id = row["exp_id"]
            label = row.get("label", "").strip().upper()
            if label not in ("X", "O"):
                label = "O"
            exp_labels.setdefault(exp_id, []).append(label)
            exp_run_ids.setdefault(exp_id, []).append(row["run_id"])

    result_rows: list[dict] = []
    for exp_id, labels in exp_labels.items():
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

        # Correct if vote matches true label; ties count as incorrect
        correct = int(vote == true_label) if vote != "T" else 0

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
    file_exists = voted_csv.exists()
    with voted_csv.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(result_rows)

    n = len(result_rows)
    accuracy = sum(r["correct"] for r in result_rows) / n if n > 0 else 0
    ties = sum(1 for r in result_rows if r["majority_vote"] == "T")
    print(f"\nClassification summary")
    print(f"- Experiments: {n}")
    print(f"- Accuracy (majority vote, excl. ties): {accuracy:.3f}")
    print(f"- Ties: {ties}")
    for row in result_rows:
        symbol = "△" if row["majority_vote"] == "T" else ("✓" if row["correct"] else "✗")
        print(
            f"  {row['exp_id']}: true={row['true_label']} (rho={row['rho_true']}) "
            f"vote={row['majority_vote']} [{row['x_count']}X / {row['o_count']}O] {symbol}"
        )

    return voted_csv