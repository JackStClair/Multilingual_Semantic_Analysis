#!/usr/bin/env python3
"""Expand seed lexicon + annotations from external research resources.

Sources (see data/external/SOURCES.md):
  - Koharu24 MyUWD German / Japanese untranslatable-term sheets
  - INK-USC/untranslatability-corpus (Japanese cultural/historical/religious rows)

Outputs merge into data/seed_lexicon and rebuild annotations via build_annotations.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
EXT = ROOT / "data" / "external"
LEX = ROOT / "data" / "seed_lexicon"
ANN = ROOT / "data" / "annotations"
EXP = ROOT / "data" / "explanations" / "store.jsonl"

# Map UWD semantic fields → our closed taxonomy
FIELD_MAP = {
    "nature": "untranslatable_lexical",
    "feeling": "affect_atmosphere",
    "feelings": "affect_atmosphere",
    "emotion": "affect_atmosphere",
    "behaviour": "self_other",
    "behavior": "self_other",
    "appearance": "untranslatable_lexical",
    "society": "political_social",
    "social": "political_social",
    "culture": "untranslatable_lexical",
    "food": "untranslatable_lexical",
    "time": "untranslatable_lexical",
    "philosophy": "untranslatable_lexical",
    "religion": "institutional_legal",
    "work": "institutional_legal",
    "communication": "politeness_stance",
    "relationship": "self_other",
    "relationships": "self_other",
}

# Extra English host contexts to multiply coverage for each lexicon item
EXTRA_TEMPLATES = [
    "In translation class we debated how to render {trigger} without flattening it.",
    "The essay hinges on {trigger}, a word whose usual gloss feels too thin.",
    "She kept returning to {trigger} as the only honest label for what she felt.",
    "Tourists hear {trigger} as a slogan; locals hear a denser cultural package.",
    "His diary used {trigger} in a way English can only paraphrase at length.",
]


def _norm_field(val) -> str:
    if pd.isna(val):
        return ""
    return str(val).strip().lower()


def _split_triggers(raw) -> list[str]:
    if pd.isna(raw) or not str(raw).strip():
        return []
    parts = re.split(r"[,;/]| or ", str(raw))
    out = []
    for p in parts:
        t = p.strip().strip("\"'").lower()
        t = re.sub(r"\s+", " ", t)
        if 2 <= len(t) <= 48 and not t.startswith("to "):
            out.append(t)
        elif t.startswith("to ") and 5 <= len(t) <= 48:
            out.append(t[3:].strip())
    # unique preserve order
    seen = set()
    uniq = []
    for t in out:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq[:6]


def _find_span(text: str, triggers: list[str]) -> tuple[str, int, int] | None:
    low = text.lower()
    for t in sorted(triggers, key=len, reverse=True):
        idx = low.find(t.lower())
        if idx >= 0:
            return text[idx : idx + len(t)], idx, idx + len(t)
    return None


def _taxonomy(field: str) -> str:
    f = _norm_field(field)
    for key, lab in FIELD_MAP.items():
        if key in f:
            return lab
    return "untranslatable_lexical"


def load_base_lexicon(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def merge_lex_item(lex: dict, item: dict) -> None:
    """Merge by tgt_span; extend triggers if existing."""
    for existing in lex["items"]:
        if existing["tgt_span"] == item["tgt_span"] and existing["nuance_type"] == item[
            "nuance_type"
        ]:
            for t in item["en_triggers"]:
                if t not in existing["en_triggers"]:
                    existing["en_triggers"].append(t)
            return
    lex["items"].append(item)


def import_uwd_de() -> tuple[list[dict], list[dict]]:
    path = EXT / "uwd_de.xlsx"
    df = pd.read_excel(path)
    lex_items, ann_rows = [], []
    # collapse senses: first row per Word
    for word, group in df.groupby("Word", sort=False):
        row = group.iloc[0]
        triggers = _split_triggers(row.get("Synonyms"))
        # pull a short phrase from definition if needed
        defin = str(row.get("Definition ENG") or "")
        if not triggers and defin:
            # crude: first 1–4 content words
            m = re.match(r"^((?:[A-Za-z]+[ \-]){0,3}[A-Za-z]+)", defin)
            if m:
                triggers = [m.group(1).lower()]
        if not triggers:
            continue
        ntype = _taxonomy(row.get("Semantic field"))
        item = {
            "id": f"de-uwd-{re.sub(r'[^a-z0-9]+', '-', str(word).lower()).strip('-')}",
            "en_triggers": triggers,
            "tgt_span": str(word),
            "nuance_type": ntype,
            "en_frame": defin[:180],
            "target_frame": f"German lexical package: {word}",
            "explanation": (
                f"English usually paraphrases what German packs into '{word}'. "
                f"{defin[:240]}"
            ),
            "evidence": (
                "Koharu24 MyUWD Dataset De + (untranslatable-term reverse-dictionary resource); "
                "see data/external/SOURCES.md"
            ),
        }
        lex_items.append(item)

        for _, r in group.iterrows():
            en = r.get("English sentence translation")
            if pd.isna(en) or not str(en).strip():
                continue
            text = re.sub(r"\s+", " ", str(en)).strip()
            hit = _find_span(text, triggers)
            if not hit:
                continue
            span, start, end = hit
            ann_rows.append(
                {
                    "id": f"{item['id']}:{abs(hash(text)) % 10_000_000:07d}",
                    "lang_pair": "en-de",
                    "source_text": text,
                    "translation": str(r.get("Example") or f"[de] focus:{word}"),
                    "src_char_start": start,
                    "src_char_end": end,
                    "src_span": span,
                    "tgt_span": str(word),
                    "nuance_type": ntype,
                    "en_frame": item["en_frame"],
                    "target_frame": item["target_frame"],
                    "explanation": item["explanation"],
                    "evidence": item["evidence"],
                    "annotator": "import-uwd-de",
                    "confidence": "medium",
                    "lexicon_id": item["id"],
                }
            )
    return lex_items, ann_rows


def import_uwd_ja() -> tuple[list[dict], list[dict]]:
    path = EXT / "uwd_ja.xlsx"
    df = pd.read_excel(path)
    lex_items, ann_rows = [], []
    for (romaji, ideogram), group in df.groupby(["Word (Romaji)", "Ideogram"], sort=False):
        row = group.iloc[0]
        triggers = _split_triggers(row.get("English synonyms"))
        defin = str(row.get("Definition ENG") or "")
        if not triggers and defin:
            m = re.match(r"^((?:[A-Za-z]+[ \-]){0,3}[A-Za-z]+)", defin)
            if m:
                triggers = [m.group(1).lower()]
        # also allow romaji as trigger when used as loan in English hosts
        if romaji and str(romaji).lower() not in triggers:
            triggers.append(str(romaji).lower())
        if not triggers:
            continue
        ntype = _taxonomy(row.get("Semantic field"))
        tgt = str(ideogram) if not pd.isna(ideogram) else str(romaji)
        item = {
            "id": f"ja-uwd-{re.sub(r'[^a-z0-9]+', '-', str(romaji).lower()).strip('-')}",
            "en_triggers": triggers[:6],
            "tgt_span": tgt,
            "nuance_type": ntype,
            "en_frame": defin[:180],
            "target_frame": f"Japanese lexical package: {romaji} ({tgt})",
            "explanation": (
                f"English paraphrases what Japanese packs into '{romaji}' ({tgt}). "
                f"{defin[:240]}"
            ),
            "evidence": (
                "Koharu24 MyUWD Dataset Jpn + (untranslatable-term reverse-dictionary resource); "
                "see data/external/SOURCES.md"
            ),
        }
        lex_items.append(item)
        for _, r in group.iterrows():
            en = r.get("English sentence translation")
            if pd.isna(en) or not str(en).strip():
                continue
            text = re.sub(r"\s+", " ", str(en)).strip()
            hit = _find_span(text, triggers)
            if not hit:
                continue
            span, start, end = hit
            ann_rows.append(
                {
                    "id": f"{item['id']}:{abs(hash(text)) % 10_000_000:07d}",
                    "lang_pair": "en-ja",
                    "source_text": text,
                    "translation": str(r.get("Example") or f"[ja] focus:{tgt}"),
                    "src_char_start": start,
                    "src_char_end": end,
                    "src_span": span,
                    "tgt_span": tgt,
                    "nuance_type": ntype,
                    "en_frame": item["en_frame"],
                    "target_frame": item["target_frame"],
                    "explanation": item["explanation"],
                    "evidence": item["evidence"],
                    "annotator": "import-uwd-ja",
                    "confidence": "medium",
                    "lexicon_id": item["id"],
                }
            )
    return lex_items, ann_rows


def import_untranslatability_ja() -> list[dict]:
    """Use English translations that conserve Japanese cultural terms as hosts."""
    df = pd.read_parquet(EXT / "untranslatability.parquet")
    ja = df[df["id"].str.startswith("ja-")].copy()
    # Prefer customs / history / religion / proper-nameish cultural rows
    keep = ja["id"].str.match(r"ja-(cu|hi|re|pn|ph)-")
    ja = ja[keep]
    rows = []
    type_map = {
        "cu": "untranslatable_lexical",
        "hi": "political_social",
        "re": "institutional_legal",
        "pn": "untranslatable_lexical",
        "ph": "untranslatable_lexical",
    }
    for _, r in ja.iterrows():
        utype = r["id"].split("-")[1]
        ntype = type_map.get(utype, "untranslatable_lexical")
        # Prefer annotated English that keeps the cultural term + explanation
        an = str(r.get("an") or "")
        no = str(r.get("no") or "")
        text = an.split("(")[0].strip() if an else no.strip()
        if not text or len(text) < 8:
            continue
        # Heuristic: capitalized multi-word or known loan-ish tokens as span
        candidates = re.findall(
            r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}|[a-z]*soba|[a-z]*bento|"
            r"hanami|setsubun|matsuri|omamori|omikuji|obon|seppuku|wabi|sabi|"
            r"ikebana|origami|kimono|yukata|tatami|futon|sensei|senpai|kohai|"
            r"kaizen|ikigai|otaku|emoji|tsunami|typhoon)\b",
            text,
        )
        # also quoted terms
        candidates += re.findall(r"[\"“]([^\"”]{2,40})[\"”]", an)
        candidates = [c.strip() for c in candidates if len(c.strip()) >= 3]
        if not candidates:
            # fall back: last noun-ish chunk before period
            continue
        trigger = candidates[0]
        hit = _find_span(text, [trigger])
        if not hit:
            continue
        span, start, end = hit
        note = ""
        if "(" in an and ")" in an:
            note = an[an.find("(") + 1 : an.rfind(")")]
        rows.append(
            {
                "id": f"ja-utc-{r['id']}",
                "lang_pair": "en-ja",
                "source_text": text,
                "translation": str(r["sentence"]),
                "src_char_start": start,
                "src_char_end": end,
                "src_span": span,
                "tgt_span": span,
                "nuance_type": ntype,
                "en_frame": "English paraphrase / conserved cultural term",
                "target_frame": note[:200] or "Japanese culturally specific expression",
                "explanation": note[:400]
                or "English cannot fully preserve the Japanese cultural load without paraphrase or borrowing.",
                "evidence": (
                    "INK-USC/untranslatability-corpus (CC BY 4.0); "
                    "see data/external/SOURCES.md"
                ),
                "annotator": "import-utc-ja",
                "confidence": "medium",
                "lexicon_id": f"ja-utc-{utype}",
            }
        )
    return rows


def synthesize_extra_contexts(lex: dict, lang_pair: str, limit_per_item: int = 3) -> list[dict]:
    rows = []
    for item in lex["items"]:
        for trigger in item["en_triggers"][:2]:
            for tmpl in EXTRA_TEMPLATES[:limit_per_item]:
                text = tmpl.format(trigger=trigger)
                start = text.lower().find(trigger.lower())
                if start < 0:
                    continue
                end = start + len(trigger)
                rows.append(
                    {
                        "id": f"{item['id']}:syn:{abs(hash(text)) % 10_000_000:07d}",
                        "lang_pair": lang_pair,
                        "source_text": text,
                        "translation": f"[{lang_pair}] focus:{item['tgt_span']}",
                        "src_char_start": start,
                        "src_char_end": end,
                        "src_span": text[start:end],
                        "tgt_span": item["tgt_span"],
                        "nuance_type": item["nuance_type"],
                        "en_frame": item["en_frame"],
                        "target_frame": item["target_frame"],
                        "explanation": item["explanation"],
                        "evidence": item["evidence"],
                        "annotator": "synth-context-v1",
                        "confidence": "medium",
                        "lexicon_id": item["id"],
                    }
                )
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    de_lex = load_base_lexicon(LEX / "en-de.json")
    ja_lex = load_base_lexicon(LEX / "en-ja.json")

    de_items, de_anns = import_uwd_de()
    ja_items, ja_anns = import_uwd_ja()
    utc_anns = import_untranslatability_ja()

    for it in de_items:
        merge_lex_item(de_lex, it)
    for it in ja_items:
        merge_lex_item(ja_lex, it)

    de_lex["version"] = "0.2.0-external"
    ja_lex["version"] = "0.2.0-external"
    (LEX / "en-de.json").write_text(
        json.dumps(de_lex, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (LEX / "en-ja.json").write_text(
        json.dumps(ja_lex, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    import importlib

    import scripts.build_annotations as ba

    importlib.reload(ba)
    seed_rows = ba.build_rows()
    synth_de = synthesize_extra_contexts(de_lex, "en-de", limit_per_item=2)
    synth_ja = synthesize_extra_contexts(ja_lex, "en-ja", limit_per_item=2)

    all_rows = seed_rows + de_anns + ja_anns + utc_anns + synth_de + synth_ja
    seen = set()
    deduped = []
    for r in all_rows:
        key = (r["lang_pair"], r["source_text"], r["src_char_start"], r["src_char_end"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)

    train, val, test = ba.split_rows(deduped)
    ba.write_jsonl(ANN / "train.jsonl", train)
    ba.write_jsonl(ANN / "val.jsonl", val)
    ba.write_jsonl(ANN / "test.jsonl", test)
    ba.write_explanations(deduped)

    print(
        f"Lexicon sizes: en-de={len(de_lex['items'])} en-ja={len(ja_lex['items'])}\n"
        f"Imported UWD DE anns={len(de_anns)} JA anns={len(ja_anns)} UTC anns={len(utc_anns)}\n"
        f"Synth contexts: de={len(synth_de)} ja={len(synth_ja)}\n"
        f"Total unique spans={len(deduped)} → train={len(train)} val={len(val)} test={len(test)}"
    )


if __name__ == "__main__":
    main()
