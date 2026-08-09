# Cultural Nuance Translation (EN→DE / EN→JA)

Machine translation often finds a *denotational* match and stops there. This project surfaces **cultural asymmetry**: cases where an English word and its German or Japanese rendering look equivalent, but the cultural frames diverge (individual liberty vs. communal *Freiheit* discourse; English “dependence” vs. Japanese *amae* 甘え).

**Architecture**

1. **Frozen NLLB-200** (Hugging Face / PyTorch MPS) for translation quality  
2. **MLX-trained span head** on frozen XLM-R features for culturally loaded English spans  
3. **Research-backed explanation store** for clickable info boxes in Gradio  

Cultural asymmetry ≠ translation error. The detector flags spans; humans (and cited sources) explain them.

## Why MLX

This repo targets Apple Silicon. Training uses a hybrid that is practical on Mac:

- **Frozen XLM-RoBERTa encoder** (Transformers on MPS) for contextual features  
- **MLX MLP token-classification head** trained with `mlx.nn` + AdamW  

So the learnable detector is trained in MLX without needing a full fragile XLM-R port. Translation still uses frozen **NLLB-200** via Transformers/PyTorch (MPS). An experimental full-MLX RoBERTa port lives in `src/nuance_mt/mlx_roberta.py` for future work.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .

# Prepare BIO tags from annotations
python -m training.prepare_data

# Train span detector with MLX (Apple Silicon)
python -m training.train_span_classifier --epochs 8

# Evaluate vs lexicon baseline
python -m training.evaluate

# Local HTML UI (translate + review medium imports)
python -m nuance_mt.ui.server
# http://127.0.0.1:7860/        → translator
# http://127.0.0.1:7860/review  → human-clean medium rows

# Optional Gradio demo
python -m nuance_mt.ui.app
```

Human-cleaning guide: [`data/guidelines/REVIEW.md`](data/guidelines/REVIEW.md).

## Dataset

See [`data/guidelines/`](data/guidelines/) for taxonomy, annotation guidelines, and datasheet. Seed lexicons live in [`data/seed_lexicon/`](data/seed_lexicon/). Annotated JSONL splits are in [`data/annotations/`](data/annotations/).

Every retained span includes an `evidence` field (citation). LLM drafts are allowed only as proposals under human review.

**External expansion** (see [`data/external/SOURCES.md`](data/external/SOURCES.md)):

- [Koharu24 MyUWD](https://github.com/Koharu24/mBERT-Unaligned-fine-tuning-for-a-cross-lingual-RD-of-untranslatable-terms) DE/JA untranslatable-term sheets  
- [INK-USC/untranslatability-corpus](https://huggingface.co/datasets/INK-USC/untranslatability-corpus) (CC BY 4.0) Japanese cultural rows  

```bash
python scripts/expand_from_external.py
python -m training.prepare_data
python -m training.train_span_classifier --epochs 20
```

## Project layout

```
src/nuance_mt/          # translate, detect (MLX), explain, pipeline, Gradio UI
training/               # prepare_data, MLX train loop, evaluate, lexicon baseline
data/                   # guidelines, seed lexicons, annotations, explanations
models/                 # trained MLX weights (gitignored)
notebooks/              # error analysis
```

## License

MIT — dataset annotations are research prototypes; verify citations before academic reuse.
