"""MLX token-classification head trained on frozen XLM-R features."""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np


class TokenClassificationHead(nn.Module):
    """Two-layer MLP head over frozen encoder hidden states."""

    def __init__(self, hidden_size: int, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.drop = nn.Dropout(dropout)

    def __call__(self, hidden: mx.array, training: bool = False) -> mx.array:
        x = nn.gelu(self.dense(hidden))
        if training:
            x = self.drop(x)
        return self.classifier(x)


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
) -> TokenClassificationHead:
    path = Path(path)
    if path.suffix == ".npz":
        path = path.with_suffix(".safetensors")
    model = TokenClassificationHead(hidden_size, num_labels)
    model.load_weights(str(path))
    mx.eval(model.parameters())
    return model


def softmax_np(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)
