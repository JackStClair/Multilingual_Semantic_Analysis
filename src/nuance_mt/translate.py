"""Frozen NLLB-200 translation wrapper (PyTorch / MPS on Mac)."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

TargetLang = Literal["de", "ja"]

NLLB_LANG = {
    "en": "eng_Latn",
    "de": "deu_Latn",
    "ja": "jpn_Jpan",
}


@lru_cache(maxsize=1)
def _load_nllb(model_name: str = "facebook/nllb-200-distilled-600M"):
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return tokenizer, model, device


class Translator:
    def __init__(self, model_name: str = "facebook/nllb-200-distilled-600M"):
        self.model_name = model_name
        self._tokenizer = None
        self._model = None
        self._device = None

    def _ensure(self) -> None:
        if self._model is None:
            self._tokenizer, self._model, self._device = _load_nllb(self.model_name)

    def translate(self, text: str, target: TargetLang, max_new_tokens: int = 256) -> str:
        import torch

        self._ensure()
        assert self._tokenizer is not None and self._model is not None
        tokenizer = self._tokenizer
        tokenizer.src_lang = NLLB_LANG["en"]
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        forced_bos = tokenizer.convert_tokens_to_ids(NLLB_LANG[target])
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                forced_bos_token_id=forced_bos,
                max_new_tokens=max_new_tokens,
            )
        return tokenizer.batch_decode(out, skip_special_tokens=True)[0]
