"""End-to-end: translate + detect + explain."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Optional

from nuance_mt.detect import DetectedSpan, LexiconDetector, NuanceDetector
from nuance_mt.explain import ExplanationStore, default_store_path
from nuance_mt.translate import Translator

TargetLang = Literal["de", "ja"]
LANG_PAIR = {"de": "en-de", "ja": "en-ja"}


@dataclass
class NuanceHit:
    start: int
    end: int
    text: str
    nuance_type: str
    score: float
    en_frame: str = ""
    target_frame: str = ""
    explanation: str = ""
    evidence: str = ""
    tgt_span: str = ""


@dataclass
class PipelineResult:
    source: str
    translation: str
    target: TargetLang
    hits: list[NuanceHit]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "translation": self.translation,
            "target": self.target,
            "hits": [asdict(h) for h in self.hits],
        }


class NuancePipeline:
    def __init__(
        self,
        weights_path: Optional[Path | str] = None,
        use_lexicon_fallback: bool = True,
        repo_root: Optional[Path] = None,
    ):
        root = repo_root or Path(__file__).resolve().parents[2]
        self.translator = Translator()
        self.detector = NuanceDetector(weights_path=weights_path)
        self.store = ExplanationStore(default_store_path())
        lex_paths = [
            root / "data" / "seed_lexicon" / "en-de.json",
            root / "data" / "seed_lexicon" / "en-ja.json",
        ]
        self.lexicon = LexiconDetector(lex_paths) if use_lexicon_fallback else None

    def run(self, text: str, target: TargetLang) -> PipelineResult:
        lang_pair = LANG_PAIR[target]
        translation = self.translator.translate(text, target)
        spans = self.detector.predict(text, lang_pair)
        if (not spans) and self.lexicon is not None:
            spans = self.lexicon.predict(text, lang_pair)
        hits = [self._enrich(s, lang_pair) for s in spans]
        return PipelineResult(
            source=text, translation=translation, target=target, hits=hits
        )

    def _enrich(self, span: DetectedSpan, lang_pair: str) -> NuanceHit:
        exp = self.store.lookup(span.text, lang_pair, span.nuance_type)
        return NuanceHit(
            start=span.start,
            end=span.end,
            text=span.text,
            nuance_type=span.nuance_type,
            score=span.score,
            en_frame=exp.en_frame if exp else "",
            target_frame=exp.target_frame if exp else "",
            explanation=exp.explanation if exp else "",
            evidence=exp.evidence if exp else "",
            tgt_span=exp.tgt_span if exp else "",
        )


def highlight_html(text: str, hits: list[NuanceHit]) -> str:
    """Render source text with clickable nuance spans."""
    if not hits:
        return _esc(text)
    parts: list[str] = []
    cursor = 0
    for i, h in enumerate(sorted(hits, key=lambda x: x.start)):
        if h.start < cursor:
            continue
        parts.append(_esc(text[cursor : h.start]))
        tip = _esc(h.nuance_type)
        parts.append(
            f'<mark class="nuance" data-idx="{i}" title="{tip}">'
            f"{_esc(text[h.start:h.end])}</mark>"
        )
        cursor = h.end
    parts.append(_esc(text[cursor:]))
    return "".join(parts)


def _esc(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
