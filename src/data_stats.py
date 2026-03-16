"""Compute and print dataset statistics for all splits."""

import csv
from pathlib import Path
from collections import Counter

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def read_tsv(path):
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return list(reader)


def stats_for_split(rows, has_switch_rate=False):
    labels = [r["label"] for r in rows]
    counter = Counter(labels)
    lengths = [len(r["text"].split()) for r in rows]
    avg_len = sum(lengths) / max(len(lengths), 1)

    s = {
        "count": len(rows),
        "label_dist": dict(counter),
        "avg_tokens": round(avg_len, 1),
    }

    if has_switch_rate:
        rates = [float(r.get("switch_rate", 0)) for r in rows]
        s["avg_switch_rate"] = round(sum(rates) / max(len(rates), 1), 3)

    return s


def print_stats(name, stats):
    print(f"\n--- {name} ---")
    print(f"  Count:       {stats['count']}")
    print(f"  Labels:      {stats['label_dist']}")
    print(f"  Avg tokens:  {stats['avg_tokens']}")
    if "avg_switch_rate" in stats:
        print(f"  Avg switch:  {stats['avg_switch_rate']}")


def main():
    datasets = {
        "Vietnamese (mono_vi)": ("mono_vi", False),
        "English (mono_en)": ("mono_en", False),
        "Code-Mixed": ("codemixed", True),
    }

    for dataset_name, (subdir, has_sr) in datasets.items():
        print(f"\n{'='*50}")
        print(f"  {dataset_name}")
        print(f"{'='*50}")
        base = DATA_DIR / subdir
        for split in ["train", "dev", "test"]:
            path = base / f"{split}.tsv"
            if not path.exists():
                print(f"\n--- {split} --- NOT FOUND")
                continue
            rows = read_tsv(path)
            s = stats_for_split(rows, has_sr)
            print_stats(split, s)


if __name__ == "__main__":
    main()
