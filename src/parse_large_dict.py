"""Parse the large English-Vietnamese.csv dictionary and extract clean word pairs.

Converts from English->Vietnamese format to Vietnamese->English format
suitable for code-mixing (replacing Vietnamese words with English).
"""

import csv
import re
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "bilingual_dict"
INPUT_FILE = DATA_DIR / "English - Vietnamese.csv"
OUTPUT_FILE = DATA_DIR / "vi_en_dict_large.tsv"
MERGED_FILE = DATA_DIR / "vi_en_dict.tsv"


def extract_vietnamese_words(definition_text):
    """Extract clean Vietnamese words/phrases from a dictionary definition."""
    vietnamese_words = []
    
    # Pattern for Vietnamese diacritics
    vn_chars = r'[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]'
    
    # Try splitting by |- first
    if "|-" in definition_text:
        parts = definition_text.split("|-")
        text_parts = parts[1:]  # Skip part of speech
    else:
        # No |- delimiter, use the whole text
        text_parts = [definition_text]
    
    for part in text_parts:
        # Remove examples (|= ... |+)
        part = re.sub(r'\|=.*?(\||$)', '', part)
        part = re.sub(r'\|\+.*?(\||$)', '', part)
        part = re.sub(r'\|[*@#].*?(\||$)', '', part)
        
        # Remove parenthetical notes
        part = re.sub(r'\([^)]*\)', '', part)
        
        # Remove technical markers
        part = re.sub(r'<[^>]*>', '', part)
        
        # Remove part of speech labels
        part = re.sub(r'\b(danh từ|động từ|tính từ|phó từ|giới từ|liên từ|thán từ|ngoại động từ|nội động từ)\b', '', part)
        
        # Split by common separators
        candidates = re.split(r'[,;]', part)
        
        for candidate in candidates:
            word = candidate.strip()
            # Filter criteria:
            # - Not empty
            # - Reasonable length (2-15 chars for single words)
            # - Contains Vietnamese characters
            # - Not just punctuation or numbers
            if (word 
                and 2 <= len(word) <= 15
                and not word.startswith('-')
                and not word.isdigit()
                and not re.match(r'^[\W\d]+$', word)
                and not any(c in word for c in ['|', '=', '+', '/', '(', ')'])
                and re.search(vn_chars, word, re.IGNORECASE)  # Must contain Vietnamese chars
                ):
                vietnamese_words.append(word.lower())
    
    return vietnamese_words


def clean_english_word(word):
    """Clean and validate English word for code-mixing."""
    word = word.strip().lower()
    
    # Remove phonetic notation [...]
    word = re.sub(r'\s*\[[^\]]*\]', '', word)
    word = word.strip()
    
    # Skip if:
    # - Too short or too long
    # - Contains numbers or special chars after cleaning
    # - Is an abbreviation pattern
    # - Starts/ends with hyphen
    if len(word) < 2 or len(word) > 20:
        return None
    if re.search(r'[\d{}()<>|=+/\\]', word):
        return None
    if word.startswith('-') or word.endswith('-'):
        return None
    if not re.match(r'^[a-z\-]+$', word):
        return None
    return word


def parse_dictionary():
    """Parse the large dictionary and extract word pairs."""
    vi_to_en = defaultdict(set)
    
    print(f"Reading {INPUT_FILE}...")
    
    with open(INPUT_FILE, encoding='utf-16') as f:
        for i, line in enumerate(f):
            if i == 0:  # Skip header
                continue
            
            if i % 10000 == 0:
                print(f"  Processed {i} lines...")
            
            # Split on first comma
            parts = line.strip().split(',', 1)
            if len(parts) != 2:
                continue
            
            raw_english = parts[0].strip()
            vietnamese_def = parts[1].strip()
            
            english_word = clean_english_word(raw_english)
            if not english_word:
                continue
            
            # Extract Vietnamese translations
            vi_words = extract_vietnamese_words(vietnamese_def)
            
            for vi_word in vi_words:
                if vi_word and len(vi_word) >= 2:
                    vi_to_en[vi_word].add(english_word)
    
    return vi_to_en


def load_existing_dict():
    """Load the existing small dictionary."""
    existing = {}
    old_file = DATA_DIR / "vi_en_dict.tsv"
    if old_file.exists():
        with open(old_file, encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                vi = row.get('vietnamese', '').strip().lower()
                en = row.get('english', '').strip().lower()
                if vi and en:
                    existing[vi] = en
    return existing


def is_clean_vietnamese(word):
    """Filter out noisy Vietnamese words."""
    if not word:
        return False
    # Remove entries starting with brackets or numbers
    if word[0] in '[]{}()<>0123456789#@':
        return False
    # Remove entries with brackets anywhere
    if '[' in word or ']' in word:
        return False
    # Must be at least 2 chars
    if len(word) < 2:
        return False
    return True


def is_clean_english(word):
    """Filter out noisy English words."""
    if not word:
        return False
    # Must be at least 2 chars
    if len(word) < 2:
        return False
    # Remove very short nonsense patterns
    if len(word) <= 3 and not word.isalpha():
        return False
    # Must be mostly alphabetic (allow hyphens)
    alpha_count = sum(1 for c in word if c.isalpha())
    if alpha_count < len(word) * 0.8:
        return False
    # Remove obvious OCR errors (uncommon consonant clusters)
    noise_patterns = ['xf', 'qx', 'zx', 'xq', 'wq', 'qw', 'vx', 'xv', 'bx', 'xb']
    word_lower = word.lower()
    for pattern in noise_patterns:
        if pattern in word_lower:
            return False
    return True


def save_dictionary(vi_to_en, path):
    """Save Vietnamese->English dictionary to TSV."""
    clean_entries = {}
    for vi_word, en_words in vi_to_en.items():
        if is_clean_vietnamese(vi_word):
            # Filter and pick best English word
            clean_en = [e for e in en_words if is_clean_english(e)]
            if clean_en:
                best_en = min(clean_en, key=len)
                clean_entries[vi_word] = best_en
    
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['vietnamese', 'english'])
        
        for vi_word in sorted(clean_entries.keys()):
            writer.writerow([vi_word, clean_entries[vi_word]])
    
    print(f"Saved {len(clean_entries)} entries to {path}")


def main():
    # Parse the large dictionary
    vi_to_en = parse_dictionary()
    print(f"Extracted {len(vi_to_en)} Vietnamese->English pairs from large dictionary")
    
    # Save large dictionary separately
    save_dictionary(vi_to_en, OUTPUT_FILE)
    
    # Load and merge with existing small dictionary
    existing = load_existing_dict()
    print(f"Existing dictionary has {len(existing)} entries")
    
    # Merge: existing takes priority, then add new ones
    merged = {}
    for vi, en in existing.items():
        if is_clean_vietnamese(vi) and is_clean_english(en):
            merged[vi] = en
    
    for vi, en_set in vi_to_en.items():
        if vi not in merged and is_clean_vietnamese(vi):
            clean_en = [e for e in en_set if is_clean_english(e)]
            if clean_en:
                merged[vi] = min(clean_en, key=len)
    
    # Save merged dictionary
    save_dictionary({k: {v} for k, v in merged.items()}, MERGED_FILE)
    print(f"Merged dictionary: {len(merged)} total entries")


if __name__ == "__main__":
    main()
