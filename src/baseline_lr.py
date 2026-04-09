"""TF-IDF + Logistic Regression baseline for sentiment classification.

Three training settings:
  zero_shot  -- train on monolingual Vietnamese only
  limited    -- train on small code-mixed subset (500 examples)
  augmented  -- train on monolingual Vietnamese + all code-mixed data
"""

import csv
import random
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
SEED = 42
LIMITED_SIZE = 500


def read_tsv(path):
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return list(reader)


def extract_xy(rows):
    texts = [r["text"] for r in rows]
    labels = [int(r["label"]) for r in rows]
    return texts, labels


def save_predictions(texts, golds, preds, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["text", "gold_label", "pred_label"])
        for t, g, p in zip(texts, golds, preds):
            writer.writerow([t, g, p])


def train_and_eval(train_texts, train_labels, test_texts, test_labels, name):
    vec = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), sublinear_tf=True)
    X_train = vec.fit_transform(train_texts)
    X_test = vec.transform(test_texts)

    clf = LogisticRegression(max_iter=1000, random_state=SEED, C=1.0)
    clf.fit(X_train, train_labels)
    preds = clf.predict(X_test)

    acc = accuracy_score(test_labels, preds)
    f1 = f1_score(test_labels, preds, average="macro")

    print(f"\n--- {name} ---")
    print(f"  Accuracy: {acc:.4f}  Macro F1: {f1:.4f}")
    print(classification_report(test_labels, preds, target_names=["negative", "positive"], digits=4))

    save_predictions(test_texts, test_labels, preds.tolist(), OUTPUT_DIR / f"predictions_lr_{name}.tsv")
    return acc, f1


def main():
    mono_vi_train = read_tsv(DATA_DIR / "mono_vi" / "train.tsv")
    cm_train = read_tsv(DATA_DIR / "codemixed" / "train.tsv")
    cm_test = read_tsv(DATA_DIR / "codemixed" / "test.tsv")

    # keep the same code-mixed test set for every setting
    test_texts, test_labels = extract_xy(cm_test)

    # Setting 1: Zero-shot (train on monolingual Vietnamese only)
    train_texts, train_labels = extract_xy(mono_vi_train)
    train_and_eval(train_texts, train_labels, test_texts, test_labels, "zero_shot")

    # Setting 2: Limited (train on 500 code-mixed examples)
    rng = random.Random(SEED)
    limited = rng.sample(cm_train, min(LIMITED_SIZE, len(cm_train)))
    train_texts, train_labels = extract_xy(limited)
    train_and_eval(train_texts, train_labels, test_texts, test_labels, "limited")

    # Setting 3: Augmented (monolingual Vietnamese + all code-mixed)
    combined = mono_vi_train + cm_train
    train_texts, train_labels = extract_xy(combined)
    train_and_eval(train_texts, train_labels, test_texts, test_labels, "augmented")


if __name__ == "__main__":
    main()
