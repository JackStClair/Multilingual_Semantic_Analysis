# How to human-clean `medium` imports

Imported rows (UWD / untranslatability corpus / synth contexts) are marked `"confidence": "medium"`. Cleaning means a human either **promotes**, **edits**, or **rejects** each one before you trust it as gold.

## What “clean” means

For each row, decide:

1. **Is the span real cultural asymmetry?** Not just an ordinary hard word or a bad heuristic match.
2. **Is the span boundary minimal and correct?** Prefer the trigger phrase, not the whole sentence.
3. **Is `nuance_type` right?** Use [`TAXONOMY.md`](TAXONOMY.md).
4. **Are `en_frame` / `target_frame` / `explanation` accurate?** Fix hedging; remove essentializing junk.
5. **Does `evidence` point somewhere real?** Keep the source id; add a better citation if you have one.

Outcomes:

| Decision | Effect |
|----------|--------|
| **accept** | `confidence` → `high`, `annotator` gets your id |
| **edit** | your field overrides saved, then treat as accept |
| **reject** | row removed from all splits |

## Fastest path (local review UI)

```bash
source .venv/bin/activate
python -m nuance_mt.ui.server
# open http://127.0.0.1:7860/review
```

Work through the queue (DE/JA filterable). You can **rewrite the English sentence**
when context is too thin, then click **Find span in sentence** so offsets update.
Decisions append to `data/annotations/review_decisions.jsonl`.

Apply them:

```bash
python scripts/apply_review.py
python -m training.prepare_data
python -m training.train_span_classifier --epochs 20
```

## Manual path (editor)

```bash
# list medium rows
rg '"confidence": "medium"' data/annotations/train.jsonl | less
```

Edit JSONL in place only if you know what you’re doing; prefer the review UI so decisions are auditable.

## Priority order

1. Reject obvious junk (wrong span, English with no cultural load, Spanish leftovers, broken heuristics).
2. Accept clear wins with minor wording tweaks.
3. Leave uncertain items as `medium` — better than false `high`.

Aim for batches of 50–100 per sitting. You do **not** need to clear all ~1k medium rows before the demo works; clean the ones that show up in the UI first.
