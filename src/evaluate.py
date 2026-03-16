"""Evaluation utilities: Accuracy, Macro F1, per-class F1, confusion matrix."""

import csv
import sys
import json
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


def load_predictions(pred_path):
    """Load a TSV with columns: text, gold_label, pred_label."""
    golds, preds = [], []
    with open(pred_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            golds.append(int(row["gold_label"]))
            preds.append(int(row["pred_label"]))
    return golds, preds


def evaluate(golds, preds, label_names=("negative", "positive")):
    acc = accuracy_score(golds, preds)
    macro_f1 = f1_score(golds, preds, average="macro")
    report = classification_report(golds, preds, target_names=label_names, digits=4)
    cm = confusion_matrix(golds, preds)
    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "report": report,
        "confusion_matrix": cm.tolist(),
    }


def print_results(results, name=""):
    header = f"=== {name} ===" if name else "=== Results ==="
    print(header)
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Macro F1:  {results['macro_f1']:.4f}")
    print(results["report"])
    print("Confusion Matrix:")
    for row in results["confusion_matrix"]:
        print("  ", row)
    print()


def results_to_latex_row(name, results):
    """Return a LaTeX table row string."""
    return f"{name} & {results['accuracy']:.4f} & {results['macro_f1']:.4f} \\\\"


def main():
    if len(sys.argv) < 2:
        print("Usage: python evaluate.py <predictions.tsv> [name]")
        sys.exit(1)

    pred_path = sys.argv[1]
    name = sys.argv[2] if len(sys.argv) > 2 else Path(pred_path).stem

    golds, preds = load_predictions(pred_path)
    results = evaluate(golds, preds)
    print_results(results, name)
    print("LaTeX row:")
    print(results_to_latex_row(name, results))


if __name__ == "__main__":
    main()
