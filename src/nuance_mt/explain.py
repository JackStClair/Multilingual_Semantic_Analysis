"""Explanation store: join detected spans to research-backed glosses."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass
class Explanation:
    lemma: str
    lang_pair: str
    nuance_type: str
    en_frame: str
    target_frame: str
    explanation: str
    evidence: str
    tgt_span: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "lemma": self.lemma,
            "lang_pair": self.lang_pair,
            "nuance_type": self.nuance_type,
            "en_frame": self.en_frame,
            "target_frame": self.target_frame,
            "explanation": self.explanation,
            "evidence": self.evidence,
            "tgt_span": self.tgt_span,
        }


class ExplanationStore:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self._by_key: dict[tuple[str, str], list[Explanation]] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        with self.path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                exp = Explanation(
                    lemma=row["lemma"].lower(),
                    lang_pair=row["lang_pair"],
                    nuance_type=row["nuance_type"],
                    en_frame=row.get("en_frame", ""),
                    target_frame=row.get("target_frame", ""),
                    explanation=row["explanation"],
                    evidence=row.get("evidence", ""),
                    tgt_span=row.get("tgt_span", ""),
                )
                key = (exp.lang_pair, exp.lemma)
                self._by_key.setdefault(key, []).append(exp)

    def lookup(
        self,
        span_text: str,
        lang_pair: str,
        nuance_type: Optional[str] = None,
    ) -> Optional[Explanation]:
        key = (lang_pair, span_text.lower().strip())
        cands = self._by_key.get(key, [])
        if not cands:
            # substring / lemma containment
            span_l = span_text.lower()
            for (lp, lemma), items in self._by_key.items():
                if lp != lang_pair:
                    continue
                if lemma in span_l or span_l in lemma:
                    cands = items
                    break
        if not cands:
            return None
        if nuance_type:
            typed = [c for c in cands if c.nuance_type == nuance_type]
            if typed:
                return typed[0]
        return cands[0]


def default_store_path() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "explanations" / "store.jsonl"
