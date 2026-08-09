# External data sources

Downloaded into `data/external/` for lexicon/annotation expansion. Do **not** treat imports as gold without spot-checking; confidence is marked `medium`.

## 1. Koharu24 MyUWD (German / Japanese untranslatable terms)

- **Repo:** https://github.com/Koharu24/mBERT-Unaligned-fine-tuning-for-a-cross-lingual-RD-of-untranslatable-terms  
- **Files:** `MyUWD/Dataset De +.xlsx`, `MyUWD/Dataset Jpn +.xlsx`  
- **What it is:** Untranslatable-word reverse-dictionary style sheets (lemma, English definition/synonyms, examples, English sentence translations).  
- **How we use it:** Seed lexicon expansion + English host sentences where a synonym/trigger span can be aligned.  
- **Local copies:** `uwd_de.xlsx`, `uwd_ja.xlsx`

## 2. INK-USC Untranslatability Corpus

- **HF dataset:** https://huggingface.co/datasets/INK-USC/untranslatability-corpus  
- **License:** CC BY 4.0  
- **What it is:** Source sentences (ES/JA) with multiple English compensation translations for untranslatability types (customs, history, religion, etc.).  
- **How we use it:** Japanese `cu/hi/re/pn/ph` rows → English host lines that conserve cultural terms; span heuristics over conserved loans / named cultural practices.  
- **Local copy:** `untranslatability.parquet`

## Import command

```bash
python scripts/expand_from_external.py
python -m training.prepare_data
python -m training.train_span_classifier --epochs 20
```

## Ethics note

Imported cultural explanations are short and may essentialize. Prefer the curated seed lexicon (`annotator: seed-v0.1`) for high-confidence UI copy; treat imports as training signal.
