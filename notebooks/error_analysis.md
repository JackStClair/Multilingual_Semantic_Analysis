# Error analysis notes

After training:

```bash
python -m training.evaluate --split test
```

Inspect failures by category:

1. **Lexicon hits, model misses** — span boundary / BIO fragmentation  
2. **Model hits, lexicon misses** — generalization beyond seed triggers (goal)  
3. **Wrong type** — taxonomy confusions (`affect_atmosphere` vs `untranslatable_lexical`)  
4. **False positives** — ordinary uses of triggers (*education* as schooling only)

Log representative errors here as the dataset grows.
