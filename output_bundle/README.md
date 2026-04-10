# Reproducing reported test-set metrics

This archive contains **model predictions on the held-out test split** of our synthetic Vietnamese–English code-mixed corpus (one TSV per model and training setting), plus the **exact evaluation script** used to compute accuracy and macro F1 in the write-up.

## Prediction file format

Each `predictions_*.tsv` is tab-separated with a header row:

| Column       | Description                                      |
|-------------|---------------------------------------------------|
| `text`      | Input sentence                                    |
| `gold_label`| Reference label: `0` = negative, `1` = positive   |
| `pred_label`| Model prediction: `0` or `1`                      |

File naming: `predictions_<model>_<setting>.tsv`

- **Models:** `lr` (TF-IDF + logistic regression), `vinai_phobert-base`, `xlm-roberta-base`, `bert-base-multilingual-cased`  
- **Settings:** `zero_shot`, `limited`, `augmented` (as defined in `project.ipynb`)

## Dependencies (evaluation only)

Evaluation uses **scikit-learn** only (no PyTorch or transformers required):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-eval.txt
```

## How to run

From the directory that contains `evaluate.py` and the `predictions_*.tsv` files:

```bash
python evaluate.py predictions_lr_zero_shot.tsv "LR zero-shot"
```

Repeat for any other TSV. The script prints accuracy, macro F1, per-class report, confusion matrix, and a LaTeX table row.

### Check against the write-up

The notebook results table was generated with this same `evaluate.py` logic. For example:

| Run (in write-up)     | File to evaluate                                      |
|----------------------|--------------------------------------------------------|
| LR, zero-shot        | `predictions_lr_zero_shot.tsv`                         |
| PhoBERT, augmented   | `predictions_vinai_phobert-base_augmented.tsv`         |
| XLM-R, zero-shot     | `predictions_xlm-roberta-base_zero_shot.tsv`          |
| mBERT, limited       | `predictions_bert-base-multilingual-cased_limited.tsv` |

**Note:** Some runs may have a different number of test examples if a subset was used; the counts printed by `evaluate.py` (support per class) match the rows in that TSV only.

## Files in this archive

- `evaluate.py` — loads TSV, computes sklearn `accuracy_score`, `f1_score(..., average="macro")`, classification report, confusion matrix  
- `predictions_*.tsv` — test outputs only (no training data, no checkpoints)  
- `requirements-eval.txt` — minimal pip deps for evaluation  
- `REFERENCES.md` — dataset and tooling citations  

Large trained weights are **not** included; only predictions on the test split are provided here.
