"""Span-level F1 metrics + lexicon baseline comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from nuance_mt.detect import LexiconDetector
from nuance_mt.labels import LABELS
from nuance_mt.mlx_classifier import load_head

ROOT = Path(__file__).resolve().parents[1]


def bio_to_spans(tags: list[str]) -> set[tuple[int, int, str]]:
    spans: set[tuple[int, int, str]] = set()
    i = 0
    while i < len(tags):
        tag = tags[i]
        if tag.startswith("B-"):
            ntype = tag[2:]
            j = i + 1
            while j < len(tags) and tags[j] == f"I-{ntype}":
                j += 1
            spans.add((i, j, ntype))
            i = j
        else:
            i += 1
    return spans


def span_f1_from_bio(gold: list[list[str]], pred: list[list[str]]) -> dict:
    strict_tp = strict_fp = strict_fn = 0
    partial_tp = partial_fp = partial_fn = 0
    per_type: dict = {}

    for g_tags, p_tags in zip(gold, pred):
        g_spans = bio_to_spans(g_tags)
        p_spans = bio_to_spans(p_tags)
        strict_tp += len(g_spans & p_spans)
        strict_fp += len(p_spans - g_spans)
        strict_fn += len(g_spans - p_spans)

        matched_g = set()
        matched_p = set()
        for ps in p_spans:
            for gs in g_spans:
                if gs in matched_g:
                    continue
                if ps[2] == gs[2] and max(ps[0], gs[0]) < min(ps[1], gs[1]):
                    matched_p.add(ps)
                    matched_g.add(gs)
                    break
        partial_tp += len(matched_p)
        partial_fp += len(p_spans) - len(matched_p)
        partial_fn += len(g_spans) - len(matched_g)

        for _, _, t in g_spans:
            per_type.setdefault(t, {"tp": 0, "fp": 0, "fn": 0})
        for _, _, t in p_spans:
            per_type.setdefault(t, {"tp": 0, "fp": 0, "fn": 0})
        for s in g_spans & p_spans:
            per_type[s[2]]["tp"] += 1
        for s in p_spans - g_spans:
            per_type[s[2]]["fp"] += 1
        for s in g_spans - p_spans:
            per_type[s[2]]["fn"] += 1

    def f1(tp, fp, fn):
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        return 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)

    type_f1 = {t: f1(v["tp"], v["fp"], v["fn"]) for t, v in sorted(per_type.items())}
    return {
        "strict_f1": f1(strict_tp, strict_fp, strict_fn),
        "partial_f1": f1(partial_tp, partial_fp, partial_fn),
        "strict_tp": strict_tp,
        "strict_fp": strict_fp,
        "strict_fn": strict_fn,
        "per_type_f1": type_f1,
    }


def char_spans_to_token_bio(
    text: str,
    spans: list[dict],
    tokenizer,
    lang_pair: str,
    max_length: int = 256,
) -> list[str]:
    from nuance_mt.labels import LANG_PAIR_TOKENS, bio_label

    prefix = LANG_PAIR_TOKENS[lang_pair]
    enc = tokenizer(
        prefix,
        text,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
    )
    offsets = enc["offset_mapping"]
    seq_ids = enc.sequence_ids()
    tags = ["O"] * len(offsets)
    for span in spans:
        started = False
        for i, (sid, (a, b)) in enumerate(zip(seq_ids, offsets)):
            if sid != 1:
                continue
            if b <= span["start"] or a >= span["end"]:
                continue
            if not started:
                tags[i] = bio_label("B", span["nuance_type"])
                started = True
            else:
                tags[i] = bio_label("I", span["nuance_type"])
    out = []
    for sid, tag in zip(seq_ids, tags):
        if sid != 1:
            continue
        out.append(tag)
    return out


def lexicon_baseline_on_annotations(split: str = "test") -> dict:
    from training.prepare_data import group_by_sentence, load_jsonl

    ann = load_jsonl(ROOT / "data" / "annotations" / f"{split}.jsonl")
    grouped = group_by_sentence(ann)
    detector = LexiconDetector(
        [
            ROOT / "data" / "seed_lexicon" / "en-de.json",
            ROOT / "data" / "seed_lexicon" / "en-ja.json",
        ]
    )
    tok_path = ROOT / "data" / "processed" / "tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(
        tok_path if tok_path.exists() else "xlm-roberta-base"
    )
    if not tok_path.exists():
        from nuance_mt.labels import LANG_PAIR_TOKENS

        tokenizer.add_special_tokens(
            {"additional_special_tokens": list(LANG_PAIR_TOKENS.values())}
        )

    gold_bios, pred_bios = [], []
    for g in grouped:
        gold_bios.append(
            char_spans_to_token_bio(
                g["source_text"], g["spans"], tokenizer, g["lang_pair"]
            )
        )
        hits = detector.predict(g["source_text"], g["lang_pair"])
        pred_spans = [
            {"start": h.start, "end": h.end, "nuance_type": h.nuance_type} for h in hits
        ]
        pred_bios.append(
            char_spans_to_token_bio(
                g["source_text"], pred_spans, tokenizer, g["lang_pair"]
            )
        )
        L = min(len(gold_bios[-1]), len(pred_bios[-1]))
        gold_bios[-1] = gold_bios[-1][:L]
        pred_bios[-1] = pred_bios[-1][:L]

    metrics = span_f1_from_bio(gold_bios, pred_bios)
    metrics["system"] = "lexicon_baseline"
    metrics["split"] = split
    return metrics


def evaluate_trained(split: str = "test", weights: Path | None = None) -> dict:
    import mlx.core as mx

    from training.train_span_classifier import (
        batches,
        cache_split_features,
        device_torch,
        evaluate_head,
        load_split,
    )

    proc = ROOT / "data" / "processed" / f"{split}.jsonl"
    data = load_split(proc)
    weights = weights or (ROOT / "models" / "span_classifier.safetensors")
    tok = AutoTokenizer.from_pretrained(ROOT / "data" / "processed" / "tokenizer")
    encoder = AutoModel.from_pretrained("xlm-roberta-base")
    encoder.resize_token_embeddings(len(tok))
    device = device_torch()
    encoder.to(device)
    encoder.eval()
    cached = cache_split_features(split, data, encoder, device, batch_size=8)
    del encoder
    head = load_head(weights, cached[0]["hidden"].shape[-1], num_labels=len(LABELS))
    metrics = evaluate_head(head, cached, batch_size=8)
    metrics["system"] = "mlx_head_frozen_xlmr"
    metrics["split"] = split
    return metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test")
    parser.add_argument("--weights", default=None)
    parser.add_argument(
        "--skip-model",
        action="store_true",
        help="Only run lexicon baseline (no MLX weights required)",
    )
    args = parser.parse_args(argv)

    lex = lexicon_baseline_on_annotations(args.split)
    print("=== Lexicon baseline ===")
    print(json.dumps({k: v for k, v in lex.items() if k != "per_type_f1"}, indent=2))
    print("per_type_f1:", json.dumps(lex["per_type_f1"], indent=2))

    if not args.skip_model:
        weights = (
            Path(args.weights)
            if args.weights
            else ROOT / "models" / "span_classifier.safetensors"
        )
        if not weights.exists():
            print("\nNo trained weights found; skipping model eval.")
            return
        model_metrics = evaluate_trained(args.split, weights)
        print("\n=== MLX head + frozen XLM-R ===")
        print(
            json.dumps(
                {k: v for k, v in model_metrics.items() if k != "per_type_f1"}, indent=2
            )
        )
        print("per_type_f1:", json.dumps(model_metrics["per_type_f1"], indent=2))
        print(
            f"\nDelta strict_f1 (model - lexicon): "
            f"{model_metrics['strict_f1'] - lex['strict_f1']:+.3f}"
        )


if __name__ == "__main__":
    main()
