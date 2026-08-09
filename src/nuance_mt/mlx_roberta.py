"""XLM-RoBERTa token classifier implemented in MLX."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np


@dataclass
class RobertaConfig:
    vocab_size: int = 250002
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    max_position_embeddings: int = 514
    type_vocab_size: int = 1
    layer_norm_eps: float = 1e-5
    pad_token_id: int = 1
    num_labels: int = 13


class RobertaEmbeddings(nn.Module):
    def __init__(self, config: RobertaConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pad_token_id = config.pad_token_id

    def __call__(self, input_ids: mx.array) -> mx.array:
        seq_len = input_ids.shape[1]
        position_ids = mx.arange(seq_len)[None, :] + self.pad_token_id + 1
        # RoBERTa positions start at padding_idx+1
        position_ids = mx.broadcast_to(position_ids, input_ids.shape)
        token_type_ids = mx.zeros(input_ids.shape, dtype=input_ids.dtype)

        x = (
            self.word_embeddings(input_ids)
            + self.position_embeddings(position_ids)
            + self.token_type_embeddings(token_type_ids)
        )
        return self.layer_norm(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, config: RobertaConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scale = math.sqrt(self.head_dim)
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)

    def _split(self, x: mx.array) -> mx.array:
        b, t, _ = x.shape
        x = x.reshape(b, t, self.num_heads, self.head_dim)
        return x.transpose(0, 2, 1, 3)

    def __call__(self, x: mx.array, attention_mask: Optional[mx.array] = None) -> mx.array:
        q = self._split(self.q_proj(x))
        k = self._split(self.k_proj(x))
        v = self._split(self.v_proj(x))
        scores = (q @ k.transpose(0, 1, 3, 2)) / self.scale
        if attention_mask is not None:
            # attention_mask: (B, T) with 1=keep, 0=pad → additive mask
            mask = (1.0 - attention_mask[:, None, None, :].astype(mx.float32)) * -1e4
            scores = scores + mask
        weights = mx.softmax(scores, axis=-1)
        ctx = (weights @ v).transpose(0, 2, 1, 3)
        b, t, h, d = ctx.shape
        ctx = ctx.reshape(b, t, h * d)
        return self.out_proj(ctx)


class FeedForward(nn.Module):
    def __init__(self, config: RobertaConfig):
        super().__init__()
        self.dense1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.dense2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        return self.dense2(nn.gelu(self.dense1(x)))


class TransformerLayer(nn.Module):
    def __init__(self, config: RobertaConfig):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.ln_attn = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ffn = FeedForward(config)
        self.ln_ffn = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def __call__(self, x: mx.array, attention_mask: Optional[mx.array] = None) -> mx.array:
        # HF RoBERTa uses post-LN: attn on x, residual, LN; ffn on that, residual, LN
        attn = self.attention(x, attention_mask)
        x = self.ln_attn(x + attn)
        x = self.ln_ffn(x + self.ffn(x))
        return x


class RobertaEncoder(nn.Module):
    def __init__(self, config: RobertaConfig):
        super().__init__()
        self.layers = [TransformerLayer(config) for _ in range(config.num_hidden_layers)]

    def __call__(self, x: mx.array, attention_mask: Optional[mx.array] = None) -> mx.array:
        for layer in self.layers:
            x = layer(x, attention_mask)
        return x


class RobertaForTokenClassification(nn.Module):
    def __init__(self, config: RobertaConfig):
        super().__init__()
        self.config = config
        self.embeddings = RobertaEmbeddings(config)
        self.encoder = RobertaEncoder(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        x = self.embeddings(input_ids)
        x = self.encoder(x, attention_mask)
        return self.classifier(x)


def _np(t) -> np.ndarray:
    if hasattr(t, "detach"):
        return t.detach().cpu().numpy()
    return np.asarray(t)


def load_xlmr_token_classifier(
    model_name: str = "xlm-roberta-base",
    num_labels: int = 13,
) -> tuple[RobertaForTokenClassification, RobertaConfig]:
    """Load HF XLM-R weights into the MLX token classifier."""
    from transformers import AutoConfig, AutoModelForTokenClassification

    hf_config = AutoConfig.from_pretrained(model_name)
    config = RobertaConfig(
        vocab_size=hf_config.vocab_size,
        hidden_size=hf_config.hidden_size,
        num_hidden_layers=hf_config.num_hidden_layers,
        num_attention_heads=hf_config.num_attention_heads,
        intermediate_size=hf_config.intermediate_size,
        max_position_embeddings=hf_config.max_position_embeddings,
        type_vocab_size=getattr(hf_config, "type_vocab_size", 1),
        layer_norm_eps=hf_config.layer_norm_eps,
        pad_token_id=hf_config.pad_token_id,
        num_labels=num_labels,
    )
    model = RobertaForTokenClassification(config)

    hf = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
    )
    sd = hf.state_dict()

    def set_linear(mlx_linear: nn.Linear, weight_key: str, bias_key: str) -> None:
        mlx_linear.weight = mx.array(_np(sd[weight_key]))
        if bias_key in sd:
            mlx_linear.bias = mx.array(_np(sd[bias_key]))

    def set_ln(mlx_ln: nn.LayerNorm, weight_key: str, bias_key: str) -> None:
        mlx_ln.weight = mx.array(_np(sd[weight_key]))
        mlx_ln.bias = mx.array(_np(sd[bias_key]))

    # Embeddings
    model.embeddings.word_embeddings.weight = mx.array(
        _np(sd["roberta.embeddings.word_embeddings.weight"])
    )
    model.embeddings.position_embeddings.weight = mx.array(
        _np(sd["roberta.embeddings.position_embeddings.weight"])
    )
    model.embeddings.token_type_embeddings.weight = mx.array(
        _np(sd["roberta.embeddings.token_type_embeddings.weight"])
    )
    set_ln(
        model.embeddings.layer_norm,
        "roberta.embeddings.LayerNorm.weight",
        "roberta.embeddings.LayerNorm.bias",
    )

    for i, layer in enumerate(model.encoder.layers):
        p = f"roberta.encoder.layer.{i}"
        # HF packs QKV separately
        set_linear(layer.attention.q_proj, f"{p}.attention.self.query.weight", f"{p}.attention.self.query.bias")
        set_linear(layer.attention.k_proj, f"{p}.attention.self.key.weight", f"{p}.attention.self.key.bias")
        set_linear(layer.attention.v_proj, f"{p}.attention.self.value.weight", f"{p}.attention.self.value.bias")
        set_linear(
            layer.attention.out_proj,
            f"{p}.attention.output.dense.weight",
            f"{p}.attention.output.dense.bias",
        )
        set_ln(
            layer.ln_attn,
            f"{p}.attention.output.LayerNorm.weight",
            f"{p}.attention.output.LayerNorm.bias",
        )
        set_linear(layer.ffn.dense1, f"{p}.intermediate.dense.weight", f"{p}.intermediate.dense.bias")
        set_linear(layer.ffn.dense2, f"{p}.output.dense.weight", f"{p}.output.dense.bias")
        set_ln(layer.ln_ffn, f"{p}.output.LayerNorm.weight", f"{p}.output.LayerNorm.bias")

    # Classifier is randomly initialized (new task head)
    mx.eval(model.parameters())
    return model, config


def save_weights(model: nn.Module, path: str) -> None:
    from pathlib import Path

    p = Path(path)
    if p.suffix == ".npz":
        p = p.with_suffix(".safetensors")
    elif p.suffix != ".safetensors":
        p = Path(str(p) + ".safetensors")
    p.parent.mkdir(parents=True, exist_ok=True)
    model.save_weights(str(p))


def load_weights(model: nn.Module, path: str) -> None:
    from pathlib import Path

    p = Path(path)
    candidates = [p]
    if p.suffix == ".npz":
        candidates.append(p.with_suffix(".safetensors"))
    elif p.suffix != ".safetensors":
        candidates.append(Path(str(p) + ".safetensors"))
        candidates.append(p.with_suffix(".safetensors"))
    for c in candidates:
        if c.exists():
            model.load_weights(str(c))
            mx.eval(model.parameters())
            return
    raise FileNotFoundError(f"No weights found for {path}")
