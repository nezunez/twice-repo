# References for data and evaluation

## Primary sentiment corpus (Vietnamese)

- **UIT-VSFC** — Vietnamese student feedback corpus used as the source of monolingual Vietnamese labels before code-mixed synthesis.  
  - Nguyen, D. Q., Vu, T. D., Nguyen, A. T., Dras, M., & Johnson, M. (2020). *UIT-VSFC: Vietnamese Students’ Feedback Corpus for Sentiment Analysis.* Association for Computational Linguistics (ACL).  
  - Dataset (public): search for UIT-VSFC / VSFC on Hugging Face or the authors’ release pages.

## Code-mixed test set

- The **test predictions** in this archive are for our **synthetic Vietnamese–English code-mixed** split, built by dictionary-based substitution on UIT-VSFC-derived monolingual splits (see `project.ipynb` and source archive `src/build_codemixed.py`, `prepare_splits.py`).

## Evaluation software

- **scikit-learn** — `accuracy_score`, `f1_score` with `average="macro"`, `classification_report`, `confusion_matrix`.  
  - Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python.* JMLR 12, 2825–2830.  
  - https://scikit-learn.org/stable/modules/model_evaluation.html  

## Models (for context; weights not in this zip)

- **PhoBERT:** `vinai/phobert-base` — Nguyen, D. Q., & Nguyen, A. T. (2020). *PhoBERT: Pre-trained language models for Vietnamese.* Findings of EMNLP.  
- **XLM-RoBERTa:** `xlm-roberta-base` — Conneau, A. et al. (2020). *Unsupervised Cross-lingual Representation Learning at Scale.* ACL.  
- **mBERT:** `bert-base-multilingual-cased` — Devlin, J. et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* NAACL.
