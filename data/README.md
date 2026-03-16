# Data Directory

## Label Harmonization

All datasets are harmonized to binary sentiment:
- **1** = positive
- **0** = negative

### UIT-VSFC (Vietnamese)
- Original labels: 0=negative, 1=neutral, 2=positive
- Mapping: 2 -> 1 (positive), 0 -> 0 (negative), 1 -> dropped (neutral)

### SST-2 (English)
- Original labels: 0=negative, 1=positive
- No mapping needed.

### IMDB (English, backup)
- Original labels: 0=negative, 1=positive
- No mapping needed.

## Directory Structure

- `raw/` -- downloaded datasets as-is
- `mono_vi/` -- binary Vietnamese train/dev/test from UIT-VSFC
- `mono_en/` -- binary English train/dev/test from SST-2
- `codemixed/` -- synthetic code-mixed train/dev/test
- `bilingual_dict/` -- Vietnamese-English word list for substitution
