"""Build synthetic Vietnamese-English code-mixed corpus from Vietnamese monolingual data.

For each Vietnamese sentence, randomly replace a fraction of words with
English translations from the bilingual dictionary. Outputs train/dev/test TSVs.
"""

import csv
import random
import re
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DICT_PATH = DATA_DIR / "bilingual_dict" / "vi_en_dict.tsv"
SEED = 42

MIN_TOKENS = 4
DEFAULT_SWITCH_RATE = 0.35


def load_bilingual_dict(path=DICT_PATH):
    """Load Vietnamese -> English word mapping."""
    vi_en = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            vi_word = row["vietnamese"].strip().lower()
            en_word = row["english"].strip().lower()
            if vi_word and en_word:
                vi_en[vi_word] = en_word
    sorted_keys = sorted(vi_en.keys(), key=len, reverse=True)
    return vi_en, sorted_keys


def code_mix_sentence(text, vi_en, sorted_keys, target_rate=DEFAULT_SWITCH_RATE, rng=None):
    """Replace Vietnamese phrases/words with English equivalents.

    Returns (mixed_text, actual_switch_rate) or (None, 0) if too short.
    """
    if rng is None:
        rng = random.Random(SEED)

    tokens = text.strip().split()
    if len(tokens) < MIN_TOKENS:
        return None, 0.0

    lower_text = text.lower()
    replacements = []
    used_positions = set()

    for vi_phrase in sorted_keys:
        if vi_phrase not in lower_text:
            continue
        if rng.random() > target_rate:
            continue

        pattern = re.compile(re.escape(vi_phrase), re.IGNORECASE)
        for m in pattern.finditer(text):
            start, end = m.start(), m.end()
            if any(p in used_positions for p in range(start, end)):
                continue
            replacements.append((start, end, vi_en[vi_phrase]))
            for p in range(start, end):
                used_positions.add(p)

    if not replacements:
        return None, 0.0

    replacements.sort(key=lambda x: x[0])
    result = []
    prev_end = 0
    for start, end, en_word in replacements:
        result.append(text[prev_end:start])
        result.append(en_word)
        prev_end = end
    result.append(text[prev_end:])
    mixed = "".join(result).strip()

    n_replaced_chars = sum(e - s for s, e, _ in replacements)
    total_chars = max(len(text.strip()), 1)
    actual_rate = n_replaced_chars / total_chars

    return mixed, round(actual_rate, 3)


def read_mono(path):
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return list(reader)


def build_codemixed_split(rows, vi_en, sorted_keys, target_rate, rng):
    """Generate code-mixed version of monolingual Vietnamese rows."""
    out = []
    for row in rows:
        mixed, rate = code_mix_sentence(row["text"], vi_en, sorted_keys, target_rate, rng)
        if mixed is not None:
            out.append({
                "text": mixed,
                "label": row["label"],
                "source": "synthetic",
                "switch_rate": str(rate),
            })
    return out


def write_cm_tsv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["text", "label", "source", "switch_rate"])
        for r in rows:
            writer.writerow([r["text"], r["label"], r["source"], r["switch_rate"]])
    print(f"  Wrote {len(rows)} rows to {path}")


def main(target_rate=DEFAULT_SWITCH_RATE):
    vi_en, sorted_keys = load_bilingual_dict()
    print(f"Loaded {len(vi_en)} bilingual entries")

    rng = random.Random(SEED)
    cm_dir = DATA_DIR / "codemixed"

    for split in ["train", "dev", "test"]:
        mono_path = DATA_DIR / "mono_vi" / f"{split}.tsv"
        rows = read_mono(mono_path)
        cm_rows = build_codemixed_split(rows, vi_en, sorted_keys, target_rate, rng)
        write_cm_tsv(cm_rows, cm_dir / f"{split}.tsv")

    print("Done building code-mixed corpus.")


if __name__ == "__main__":
    main()
