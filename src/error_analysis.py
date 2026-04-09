"""Error analysis: sample misclassified examples and categorize errors."""

import csv
import random
import sys
from pathlib import Path
from collections import Counter

SEED = 42
SAMPLE_SIZE = 50


def read_predictions(path):
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return list(reader)


def analyze_errors(pred_path, sample_size=SAMPLE_SIZE):
    rows = read_predictions(pred_path)
    errors = [r for r in rows if r["gold_label"] != r["pred_label"]]

    print(f"Total predictions: {len(rows)}")
    print(f"Total errors: {len(errors)} ({len(errors)/len(rows)*100:.1f}%)")

    rng = random.Random(SEED)
    # pick a few mistakes so we can inspect them by hand
    sample = rng.sample(errors, min(sample_size, len(errors)))

    print(f"\n--- Sampled {len(sample)} misclassified examples ---\n")
    for i, row in enumerate(sample, 1):
        text = row["text"][:120]
        print(f"{i:2d}. Gold={row['gold_label']} Pred={row['pred_label']} | {text}")

    gold_dist = Counter(r["gold_label"] for r in errors)
    pred_dist = Counter(r["pred_label"] for r in errors)
    print(f"\nError gold distribution: {dict(gold_dist)}")
    print(f"Error pred distribution: {dict(pred_dist)}")

    short = [r for r in errors if len(r["text"].split()) < 6]
    print(f"Short errors (<6 tokens): {len(short)}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python error_analysis.py <predictions.tsv>")
        sys.exit(1)
    analyze_errors(sys.argv[1])
