"""MLX token-classification head trained on frozen XLM-R features."""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np


class TokenClassificationHead(nn.Module):
    """Deeper MLP head over frozen encoder hidden states.

    Layout: LN → Linear → GELU → Dropout → Linear → GELU → Dropout → Linear
    with a residual from the first projection into the second block.
    """

    def __init__(
        self,
        hidden_size: int,
        num_labels: int,
        dropout: float = 0.2,
        intermediate_size: int | None = None,
    ):
        super().__init__()
        mid = intermediate_size or hidden_size
        self.ln = nn.LayerNorm(hidden_size)
        self.dense1 = nn.Linear(hidden_size, mid)
        self.dense2 = nn.Linear(mid, mid)
        self.classifier = nn.Linear(mid, num_labels)
        self.drop = nn.Dropout(dropout)

    def __call__(self, hidden: mx.array, training: bool = False) -> mx.array:
        x = self.ln(hidden)
        h = nn.gelu(self.dense1(x))
        if training:
            h = self.drop(h)
        h2 = nn.gelu(self.dense2(h) + h)
        if training:
            h2 = self.drop(h2)
        return self.classifier(h2)


def save_head(model: TokenClassificationHead, path: str | Path) -> None:
    path = Path(path)
    if path.suffix != ".safetensors":
        path = path.with_suffix(".safetensors")
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_weights(str(path))


def load_head(
    path: str | Path,
    hidden_size: int,
    num_labels: int,
    dropout: float = 0.2,
) -> TokenClassificationHead:
    path = Path(path)
    if path.suffix == ".npz":
        path = path.with_suffix(".safetensors")
    model = TokenClassificationHead(hidden_size, num_labels, dropout=dropout)
    model.load_weights(str(path))
    mx.eval(model.parameters())
    return model


def softmax_np(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)
