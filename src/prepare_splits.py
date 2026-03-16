"""Harmonize labels to binary (positive=1, negative=0) and create
monolingual Vietnamese / English splits plus a held-out code-mixed test set.

UIT-VSFC: positive(2)->1, negative(0)->0, neutral(1)->dropped
SST-2:    label 1->1, label 0->0
IMDB:     label 1->1, label 0->0
"""

import csv
import random
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
SEED = 42


def read_tsv(path):
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return list(reader)


def write_tsv(rows, path, columns=("text", "label")):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(columns)
        for row in rows:
            writer.writerow([row[c] for c in columns])
    print(f"  Wrote {len(rows)} rows to {path}")


def prepare_uit_vsfc():
    """Binary Vietnamese splits from UIT-VSFC."""
    out_dir = DATA_DIR / "mono_vi"
    all_rows = []
    for split_name in ["train", "dev", "test"]:
        raw = read_tsv(RAW_DIR / "uit_vsfc" / f"{split_name}.tsv")
        for r in raw:
            enc = int(r["encoded_sentiment"])
            if enc == 2:
                all_rows.append({"text": r["text"], "label": "1", "split": split_name})
            elif enc == 0:
                all_rows.append({"text": r["text"], "label": "0", "split": split_name})

    train = [r for r in all_rows if r["split"] == "train"]
    dev = [r for r in all_rows if r["split"] == "dev"]
    test = [r for r in all_rows if r["split"] == "test"]

    write_tsv(train, out_dir / "train.tsv")
    write_tsv(dev, out_dir / "dev.tsv")
    write_tsv(test, out_dir / "test.tsv")
    return train, dev, test


def prepare_sst2():
    """Binary English splits from SST-2."""
    out_dir = DATA_DIR / "mono_en"
    train_raw = read_tsv(RAW_DIR / "sst2" / "train.tsv")
    val_raw = read_tsv(RAW_DIR / "sst2" / "validation.tsv")

    train = [{"text": r["text"], "label": r["label"]} for r in train_raw]
    random.seed(SEED)
    random.shuffle(train)
    dev_size = min(len(val_raw), 1000)
    dev = train[:dev_size]
    train = train[dev_size:]

    test = [{"text": r["text"], "label": r["label"]} for r in val_raw]

    write_tsv(train, out_dir / "train.tsv")
    write_tsv(dev, out_dir / "dev.tsv")
    write_tsv(test, out_dir / "test.tsv")
    return train, dev, test


def main():
    print("=== Preparing Vietnamese (UIT-VSFC) ===")
    vi_train, vi_dev, vi_test = prepare_uit_vsfc()
    print(f"  Total: train={len(vi_train)} dev={len(vi_dev)} test={len(vi_test)}")

    print("\n=== Preparing English (SST-2) ===")
    en_train, en_dev, en_test = prepare_sst2()
    print(f"  Total: train={len(en_train)} dev={len(en_dev)} test={len(en_test)}")


if __name__ == "__main__":
    main()
