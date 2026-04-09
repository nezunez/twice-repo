"""Fine-tune a transformer for binary sentiment on code-mixed data.

Usage:
  python src/train_transformer.py --model xlm-roberta-base --setting zero_shot
  python src/train_transformer.py --model vinai/phobert-base --setting limited
  python src/train_transformer.py --model bert-base-multilingual-cased --setting augmented

Settings:
  zero_shot  -- fine-tune on monolingual Vietnamese, eval on code-mixed test
  limited    -- fine-tune on 500 code-mixed examples
  augmented  -- fine-tune on monolingual Vietnamese + all code-mixed train

Designed to run on Google Colab (T4 GPU) or CPU.
"""

import argparse
import csv
import random
import os
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import accuracy_score, f1_score

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
SEED = 42
LIMITED_SIZE = 500
MAX_LEN = 128


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def read_tsv(path):
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return list(reader)


class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=MAX_LEN):
        # turn raw text into tensors the trainer can use
        self.encodings = tokenizer(
            texts, truncation=True, padding="max_length", max_length=max_len, return_tensors="pt"
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }


def get_train_data(setting):
    mono_vi = read_tsv(DATA_DIR / "mono_vi" / "train.tsv")
    cm_train = read_tsv(DATA_DIR / "codemixed" / "train.tsv")

    if setting == "zero_shot":
        rows = mono_vi
    elif setting == "limited":
        rng = random.Random(SEED)
        rows = rng.sample(cm_train, min(LIMITED_SIZE, len(cm_train)))
    elif setting == "augmented":
        rows = mono_vi + cm_train
    else:
        raise ValueError(f"Unknown setting: {setting}")

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        help="HuggingFace model name (e.g. xlm-roberta-base)")
    parser.add_argument("--setting", type=str, required=True,
                        choices=["zero_shot", "limited", "augmented"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    set_seed(SEED)
    model_short = args.model.replace("/", "_")
    run_name = f"{model_short}_{args.setting}"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Model: {args.model} | Setting: {args.setting}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)

    train_texts, train_labels = get_train_data(args.setting)
    print(f"Training examples: {len(train_texts)}")

    cm_test = read_tsv(DATA_DIR / "codemixed" / "test.tsv")
    test_texts = [r["text"] for r in cm_test]
    test_labels = [int(r["label"]) for r in cm_test]

    cm_dev = read_tsv(DATA_DIR / "codemixed" / "dev.tsv")
    dev_texts = [r["text"] for r in cm_dev]
    dev_labels = [int(r["label"]) for r in cm_dev]

    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer)
    eval_dataset = SentimentDataset(dev_texts, dev_labels, tokenizer)
    test_dataset = SentimentDataset(test_texts, test_labels, tokenizer)

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "checkpoints" / run_name),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=32,
        learning_rate=args.lr,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        seed=SEED,
        logging_steps=50,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    results = trainer.evaluate(test_dataset)
    print(f"\n=== Test Results ({run_name}) ===")
    print(f"  Accuracy: {results['eval_accuracy']:.4f}")
    print(f"  Macro F1: {results['eval_macro_f1']:.4f}")

    preds_output = trainer.predict(test_dataset)
    preds = np.argmax(preds_output.predictions, axis=-1).tolist()
    save_predictions(test_texts, test_labels, preds, OUTPUT_DIR / f"predictions_{run_name}.tsv")
    print(f"Predictions saved to output/predictions_{run_name}.tsv")


if __name__ == "__main__":
    main()
