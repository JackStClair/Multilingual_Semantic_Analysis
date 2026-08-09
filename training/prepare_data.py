"""Convert JSONL span annotations to token-level BIO examples."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from transformers import AutoTokenizer

from nuance_mt.labels import LABEL2ID, LABELS, LANG_PAIR_TOKENS, bio_label

ROOT = Path(__file__).resolve().parents[1]
ANN = ROOT / "data" / "annotations"
OUT = ROOT / "data" / "processed"


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def group_by_sentence(rows: list[dict]) -> list[dict]:
    """Merge multiple spans that share the same source_text + lang_pair."""
    groups: dict[tuple[str, str], dict] = {}
    for r in rows:
        key = (r["lang_pair"], r["source_text"])
        if key not in groups:
            groups[key] = {
                "lang_pair": r["lang_pair"],
                "source_text": r["source_text"],
                "spans": [],
            }
        groups[key]["spans"].append(
            {
                "start": r["src_char_start"],
                "end": r["src_char_end"],
                "nuance_type": r["nuance_type"],
            }
        )
    return list(groups.values())


def encode_example(ex: dict, tokenizer, max_length: int = 256) -> dict:
    prefix = LANG_PAIR_TOKENS[ex["lang_pair"]]
    text = ex["source_text"]
    enc = tokenizer(
        prefix,
        text,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
        return_attention_mask=True,
    )
    offsets = enc["offset_mapping"]
    # For pair encoding, sequence_ids mark which segment
    seq_ids = enc.sequence_ids()
    labels = [LABEL2ID["O"]] * len(offsets)

    spans = sorted(ex["spans"], key=lambda s: s["start"])
    for span in spans:
        span_started = False
        for i, (seq_id, (start, end)) in enumerate(zip(seq_ids, offsets)):
            if seq_id != 1:
                # only label the English text segment
                continue
            if end <= span["start"] or start >= span["end"]:
                continue
            # token overlaps span
            if not span_started:
                labels[i] = LABEL2ID[bio_label("B", span["nuance_type"])]
                span_started = True
            else:
                labels[i] = LABEL2ID[bio_label("I", span["nuance_type"])]

    # special tokens stay -100 for loss ignore? We use O on specials / prefix
    for i, seq_id in enumerate(seq_ids):
        if seq_id is None:
            labels[i] = -100

    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "labels": labels,
        "lang_pair": ex["lang_pair"],
        "source_text": text,
        "spans": ex["spans"],
    }


def prepare_split(name: str, tokenizer, max_length: int) -> Path:
    rows = load_jsonl(ANN / f"{name}.jsonl")
    grouped = group_by_sentence(rows)
    encoded = [encode_example(g, tokenizer, max_length) for g in grouped]
    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / f"{name}.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for e in encoded:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"{name}: {len(rows)} spans → {len(encoded)} sequences → {out_path}")
    return out_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="xlm-roberta-base")
    parser.add_argument("--max-length", type=int, default=256)
    args = parser.parse_args(argv)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.add_special_tokens(
        {"additional_special_tokens": list(LANG_PAIR_TOKENS.values())}
    )
    tok_dir = OUT / "tokenizer"
    tok_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(tok_dir)

    meta = {"labels": LABELS, "label2id": LABEL2ID, "model_name": args.model_name}
    (OUT / "label_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    for split in ("train", "val", "test"):
        prepare_split(split, tokenizer, args.max_length)


if __name__ == "__main__":
    main()
