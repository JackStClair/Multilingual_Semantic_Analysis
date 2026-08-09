"""Closed nuance taxonomy and BIO label helpers."""

from __future__ import annotations

NUANCE_TYPES: list[str] = [
    "political_social",
    "self_other",
    "affect_atmosphere",
    "politeness_stance",
    "institutional_legal",
    "untranslatable_lexical",
]

LANG_PAIRS = ("en-de", "en-ja")
LANG_PAIR_TOKENS = {
    "en-de": "<en-de>",
    "en-ja": "<en-ja>",
}

# BIO over nuance types
LABELS: list[str] = ["O"] + [
    f"{prefix}-{nt}" for nt in NUANCE_TYPES for prefix in ("B", "I")
]
LABEL2ID: dict[str, int] = {label: i for i, label in enumerate(LABELS)}
ID2LABEL: dict[int, str] = {i: label for label, i in LABEL2ID.items()}


def bio_label(prefix: str, nuance_type: str) -> str:
    if nuance_type not in NUANCE_TYPES:
        raise ValueError(f"Unknown nuance_type: {nuance_type}")
    if prefix not in {"B", "I"}:
        raise ValueError(f"prefix must be B or I, got {prefix}")
    return f"{prefix}-{nuance_type}"
