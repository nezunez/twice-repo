"""Download source datasets: UIT-VSFC, SST-2, IMDB.

UIT-VSFC is loaded from a public GitHub release.
SST-2 and IMDB come from Hugging Face datasets library.
VLSP 2018 requires a license agreement -- not automated here.
"""

import os
import csv
import json
import urllib.request
import zipfile
from pathlib import Path

from datasets import load_dataset

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"


def download_uit_vsfc():
    """Download UIT-VSFC from Hugging Face."""
    out_dir = RAW_DIR / "uit_vsfc"
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("tridm/UIT-VSFC")
    split_map = {"train": "train", "validation": "dev", "test": "test"}
    for hf_split, local_name in split_map.items():
        out_path = out_dir / f"{local_name}.tsv"
        if out_path.exists():
            print(f"  Already exists: {out_path}")
            continue
        split_data = ds[hf_split]
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["text", "sentiment", "topic", "encoded_sentiment"])
            for row in split_data:
                writer.writerow([row["Sentence"], row["Sentiment"], row["Topic"], row["Encoded_sentiment"]])
        print(f"  Saved {local_name} ({len(split_data)} rows) to {out_path}")

    print(f"UIT-VSFC saved to {out_dir}")
    return out_dir


def download_sst2():
    """Download SST-2 via Hugging Face datasets."""
    out_dir = RAW_DIR / "sst2"
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("stanfordnlp/sst2")
    for split_name in ["train", "validation"]:
        out_path = out_dir / f"{split_name}.tsv"
        if out_path.exists():
            print(f"  Already exists: {out_path}")
            continue
        split_data = ds[split_name]
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["text", "label"])
            for row in split_data:
                writer.writerow([row["sentence"], row["label"]])
        print(f"  Saved {split_name} ({len(split_data)} rows) to {out_path}")

    print(f"SST-2 saved to {out_dir}")
    return out_dir


def download_imdb():
    """Download IMDB via Hugging Face datasets."""
    out_dir = RAW_DIR / "imdb"
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("stanfordnlp/imdb")
    for split_name in ["train", "test"]:
        out_path = out_dir / f"{split_name}.tsv"
        if out_path.exists():
            print(f"  Already exists: {out_path}")
            continue
        split_data = ds[split_name]
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["text", "label"])
            for row in split_data:
                writer.writerow([row["text"], row["label"]])
        print(f"  Saved {split_name} ({len(split_data)} rows) to {out_path}")

    print(f"IMDB saved to {out_dir}")
    return out_dir


if __name__ == "__main__":
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print("=== Downloading UIT-VSFC ===")
    download_uit_vsfc()
    print("\n=== Downloading SST-2 ===")
    download_sst2()
    print("\n=== Downloading IMDB ===")
    download_imdb()
    print("\nDone. Note: VLSP 2018 must be requested manually from vlsp.resources@gmail.com")
