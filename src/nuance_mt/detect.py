"""Span detector: frozen XLM-R features + MLX classification head."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from nuance_mt.labels import ID2LABEL, LANG_PAIR_TOKENS, LABELS
from nuance_mt.mlx_classifier import load_head, softmax_np


@dataclass
class DetectedSpan:
    start: int
    end: int
    text: str
    nuance_type: str
    score: float


def _torch_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@lru_cache(maxsize=1)
def _load_encoder(model_name: str, tokenizer_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir or model_name)
    if tokenizer_dir == "":
        tokenizer.add_special_tokens(
            {"additional_special_tokens": list(LANG_PAIR_TOKENS.values())}
        )
    encoder = AutoModel.from_pretrained(model_name)
    encoder.resize_token_embeddings(len(tokenizer))
    device = _torch_device()
    encoder.to(device)
    encoder.eval()
    return tokenizer, encoder, device


class NuanceDetector:
    def __init__(
        self,
        weights_path: Optional[Path | str] = None,
        model_name: str = "xlm-roberta-base",
        max_length: int = 256,
        confidence_threshold: float = 0.35,
        repo_root: Optional[Path] = None,
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.confidence_threshold = confidence_threshold
        root = repo_root or Path(__file__).resolve().parents[2]
        tok_dir = root / "data" / "processed" / "tokenizer"
        self.tokenizer, self.encoder, self.device = _load_encoder(
            model_name, str(tok_dir) if tok_dir.exists() else ""
        )
        self.hidden_size = self.encoder.config.hidden_size
        self.weights_path = Path(weights_path) if weights_path else _default_weights(root)
        self.head = None
        self._has_trained_weights = self.weights_path.exists()
        if self._has_trained_weights:
            self.head = load_head(
                self.weights_path, self.hidden_size, num_labels=len(LABELS)
            )

    @torch.no_grad()
    def predict(self, text: str, lang_pair: str) -> list[DetectedSpan]:
        if self.head is None:
            return []
        if lang_pair not in LANG_PAIR_TOKENS:
            raise ValueError(f"lang_pair must be one of {list(LANG_PAIR_TOKENS)}")
        prefix = LANG_PAIR_TOKENS[lang_pair]
        enc = self.tokenizer(
            prefix,
            text,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        offsets = enc.pop("offset_mapping")[0].tolist()
        seq_ids = self.tokenizer(
            prefix,
            text,
            return_offsets_mapping=True,
            truncation=True,
            max_length=self.max_length,
        ).sequence_ids()

        ids = enc["input_ids"].to(self.device)
        mask = enc["attention_mask"].to(self.device)
        hidden = self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state
        hidden_mx = __import__("mlx.core", fromlist=["array"]).array(
            hidden.cpu().numpy().astype(np.float32)
        )
        logits = self.head(hidden_mx, training=False)[0]
        probs = softmax_np(np.array(logits))
        pred_ids = probs.argmax(axis=-1)
        scores = probs.max(axis=-1)

        spans: list[DetectedSpan] = []
        i = 0
        while i < len(pred_ids):
            if seq_ids[i] != 1:
                i += 1
                continue
            label = ID2LABEL[int(pred_ids[i])]
            if label.startswith("B-"):
                ntype = label[2:]
                start_char, end_char = offsets[i]
                score = float(scores[i])
                j = i + 1
                while j < len(pred_ids):
                    if seq_ids[j] != 1:
                        break
                    lab_j = ID2LABEL[int(pred_ids[j])]
                    if lab_j == f"I-{ntype}":
                        if offsets[j][1] > 0:
                            end_char = offsets[j][1]
                        score = min(score, float(scores[j]))
                        j += 1
                    else:
                        break
                if end_char > start_char and score >= self.confidence_threshold:
                    snippet = text[start_char:end_char]
                    if snippet.strip():
                        spans.append(
                            DetectedSpan(
                                start=int(start_char),
                                end=int(end_char),
                                text=snippet,
                                nuance_type=ntype,
                                score=score,
                            )
                        )
                i = j
            else:
                i += 1
        return _merge_overlaps(spans)


def _merge_overlaps(spans: list[DetectedSpan]) -> list[DetectedSpan]:
    if not spans:
        return []
    spans = sorted(spans, key=lambda s: (s.start, -(s.end - s.start)))
    out: list[DetectedSpan] = []
    for s in spans:
        if out and s.start < out[-1].end:
            if s.score > out[-1].score:
                out[-1] = s
            continue
        out.append(s)
    return out


def _default_weights(root: Path) -> Path:
    for name in ("span_classifier.safetensors", "span_classifier.npz"):
        candidate = root / "models" / name
        if candidate.exists():
            return candidate
    return root / "models" / "span_classifier.safetensors"


class LexiconDetector:
    """Keyword baseline used when MLX weights are absent or for comparison."""

    def __init__(self, lexicon_paths: list[Path | str]):
        self.entries: dict[str, list[dict]] = {"en-de": [], "en-ja": []}
        for p in lexicon_paths:
            path = Path(p)
            import json

            data = json.loads(path.read_text(encoding="utf-8"))
            lang_pair = data["lang_pair"]
            for item in data["items"]:
                self.entries[lang_pair].append(item)

    def predict(self, text: str, lang_pair: str) -> list[DetectedSpan]:
        text_l = text.lower()
        spans: list[DetectedSpan] = []
        for item in self.entries.get(lang_pair, []):
            for trigger in item.get("en_triggers", []):
                t = trigger.lower()
                start = 0
                while True:
                    idx = text_l.find(t, start)
                    if idx < 0:
                        break
                    spans.append(
                        DetectedSpan(
                            start=idx,
                            end=idx + len(t),
                            text=text[idx : idx + len(t)],
                            nuance_type=item["nuance_type"],
                            score=1.0,
                        )
                    )
                    start = idx + len(t)
        return _merge_overlaps(spans)
