"""Train a cultural-nuance BIO head with MLX on frozen XLM-R features.

Apple Silicon path:
  - Encoder: Hugging Face XLM-RoBERTa (torch MPS/CPU), frozen
  - Head: MLX MLP token classifier (trained)

This keeps training on MLX while avoiding a fragile full-transformer port.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from nuance_mt.labels import LABELS, LANG_PAIR_TOKENS
from nuance_mt.mlx_classifier import TokenClassificationHead, save_head
from training.evaluate import span_f1_from_bio

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"


def device_torch() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_split(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


@torch.no_grad()
def encode_features(
    encoder: AutoModel,
    batch: list[dict],
    device: torch.device,
) -> tuple[mx.array, mx.array, mx.array]:
    max_len = max(len(b["input_ids"]) for b in batch)
    pad_id = 1
    input_ids, attention_mask, labels = [], [], []
    for b in batch:
        pad = max_len - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_id] * pad)
        attention_mask.append(b["attention_mask"] + [0] * pad)
        labels.append(b["labels"] + [-100] * pad)

    ids = torch.tensor(input_ids, dtype=torch.long, device=device)
    mask = torch.tensor(attention_mask, dtype=torch.long, device=device)
    out = encoder(input_ids=ids, attention_mask=mask)
    hidden = out.last_hidden_state.detach().cpu().numpy().astype(np.float32)
    return (
        mx.array(hidden),
        mx.array(np.array(attention_mask, dtype=np.int32)),
        mx.array(np.array(labels, dtype=np.int32)),
    )


def batches(data: list[dict], batch_size: int, shuffle: bool, seed: int):
    idx = np.arange(len(data))
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
    for start in range(0, len(data), batch_size):
        yield [data[i] for i in idx[start : start + batch_size]]


def cross_entropy_token(logits: mx.array, labels: mx.array) -> mx.array:
    b, t, c = logits.shape
    logits = logits.reshape(b * t, c)
    labels = labels.reshape(b * t)
    valid = labels >= 0
    safe = mx.where(valid, labels, mx.zeros_like(labels))
    log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    nll = -mx.take_along_axis(log_probs, safe[:, None], axis=1).squeeze(-1)
    nll = mx.where(valid, nll, mx.zeros_like(nll))
    denom = mx.maximum(valid.sum(), mx.array(1))
    return nll.sum() / denom


def evaluate_head(
    head: TokenClassificationHead,
    encoder: AutoModel,
    data: list[dict],
    batch_size: int,
    device: torch.device,
) -> dict:
    all_pred, all_gold = [], []
    for batch in batches(data, batch_size, shuffle=False, seed=0):
        hidden, attention_mask, labels = encode_features(encoder, batch, device)
        logits = head(hidden, training=False)
        pred = np.array(mx.argmax(logits, axis=-1))
        gold = np.array(labels)
        mask = np.array(attention_mask)
        for p_row, g_row, m_row in zip(pred, gold, mask):
            p_tags, g_tags = [], []
            for p, g, m in zip(p_row, g_row, m_row):
                if m == 0 or g == -100:
                    continue
                p_tags.append(LABELS[int(p)])
                g_tags.append(LABELS[int(g)])
            all_pred.append(p_tags)
            all_gold.append(g_tags)
    return span_f1_from_bio(all_gold, all_pred)


def train(args: argparse.Namespace) -> None:
    train_path = PROC / "train.jsonl"
    val_path = PROC / "val.jsonl"
    if not train_path.exists():
        raise SystemExit("Run `python -m training.prepare_data` first.")

    train_data = load_split(train_path)
    val_data = load_split(val_path)
    print(f"Train sequences: {len(train_data)} | Val: {len(val_data)}")

    device = device_torch()
    print(f"Frozen encoder device: {device} | Head trainer: MLX")

    tokenizer = AutoTokenizer.from_pretrained(PROC / "tokenizer")
    encoder = AutoModel.from_pretrained(args.model_name)
    encoder.resize_token_embeddings(len(tokenizer))
    encoder.to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    hidden_size = encoder.config.hidden_size
    head = TokenClassificationHead(hidden_size, num_labels=len(LABELS))
    mx.eval(head.parameters())

    optimizer = optim.AdamW(learning_rate=args.lr, weight_decay=args.weight_decay)

    def loss_fn(model, hidden, labels):
        logits = model(hidden, training=True)
        return cross_entropy_token(logits, labels)

    loss_and_grad = nn.value_and_grad(head, loss_fn)

    best_f1 = -1.0
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        losses = []
        for batch in batches(
            train_data, args.batch_size, shuffle=True, seed=args.seed + epoch
        ):
            warm = max(1, args.warmup_steps)
            if global_step < warm:
                optimizer.learning_rate = args.lr * (global_step + 1) / warm

            hidden, _, labels = encode_features(encoder, batch, device)
            loss, grads = loss_and_grad(head, hidden, labels)
            optimizer.update(head, grads)
            mx.eval(head.parameters(), optimizer.state)
            losses.append(float(loss))
            global_step += 1

        metrics = evaluate_head(head, encoder, val_data, args.batch_size, device)
        mean_loss = float(np.mean(losses)) if losses else 0.0
        print(
            f"epoch {epoch}/{args.epochs}  loss={mean_loss:.4f}  "
            f"val_strict_f1={metrics['strict_f1']:.3f}  "
            f"val_partial_f1={metrics['partial_f1']:.3f}  "
            f"time={time.time() - t0:.1f}s"
        )
        if metrics["strict_f1"] >= best_f1:
            best_f1 = metrics["strict_f1"]
            save_head(head, out_path)
            meta = {
                "model_name": args.model_name,
                "labels": LABELS,
                "hidden_size": hidden_size,
                "best_strict_f1": best_f1,
                "epoch": epoch,
                "trainer": "mlx_head_on_frozen_xlmr",
                "lang_pair_tokens": LANG_PAIR_TOKENS,
            }
            meta_path = out_path.with_name(out_path.stem + ".meta.json")
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            print(f"  saved best → {out_path} (strict_f1={best_f1:.3f})")

    print(f"Done. Best val strict F1={best_f1:.3f}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="MLX span head training")
    parser.add_argument("--model-name", default="xlm-roberta-base")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--output",
        default=str(MODELS / "span_classifier.safetensors"),
    )
    args = parser.parse_args(argv)
    train(args)


if __name__ == "__main__":
    main()
