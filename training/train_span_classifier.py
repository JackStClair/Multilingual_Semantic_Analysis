"""Train a cultural-nuance BIO head with MLX on frozen XLM-R features.

Apple Silicon path:
  - Encoder: Hugging Face XLM-RoBERTa (torch MPS/CPU), frozen (features cached)
  - Head: MLX MLP token classifier with class-weighted CE + cosine LR
"""

from __future__ import annotations

import argparse
import json
import math
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
CACHE = PROC / "feature_cache"


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        hidden,
        np.array(attention_mask, dtype=np.int32),
        np.array(labels, dtype=np.int32),
    )


def cache_split_features(
    name: str,
    data: list[dict],
    encoder: AutoModel,
    device: torch.device,
    batch_size: int,
    force: bool = False,
) -> list[dict]:
    """Precompute frozen encoder states once; train the head off disk/RAM."""
    import pickle

    CACHE.mkdir(parents=True, exist_ok=True)
    meta_path = CACHE / f"{name}_meta.json"
    arr_path = CACHE / f"{name}_hidden.pkl"
    model_name = getattr(encoder.config, "_name_or_path", args_model_name(encoder))
    if arr_path.exists() and meta_path.exists() and not force:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("n") == len(data) and meta.get("model") == model_name:
            print(f"Loading cached features: {arr_path}")
            with arr_path.open("rb") as f:
                return pickle.load(f)

    print(f"Caching frozen features for {name} ({len(data)} seqs)…")
    items = []
    for batch in batches(data, batch_size, shuffle=False, seed=0):
        hidden, mask, labels = encode_features(encoder, batch, device)
        for i, ex in enumerate(batch):
            L = int(mask[i].sum())
            items.append(
                {
                    "hidden": hidden[i, :L].astype(np.float32),
                    "labels": labels[i, :L].astype(np.int32),
                    "lang_pair": ex.get("lang_pair", ""),
                }
            )
    with arr_path.open("wb") as f:
        pickle.dump(items, f, protocol=pickle.HIGHEST_PROTOCOL)
    meta_path.write_text(
        json.dumps(
            {
                "n": len(data),
                "model": model_name,
                "hidden_size": int(items[0]["hidden"].shape[-1]) if items else 0,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"  wrote {arr_path}")
    return items


def args_model_name(encoder) -> str:
    return str(getattr(encoder.config, "name_or_path", "") or "xlm-roberta-base")


def batches(data: list, batch_size: int, shuffle: bool, seed: int):
    idx = np.arange(len(data))
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
    for start in range(0, len(data), batch_size):
        yield [data[i] for i in idx[start : start + batch_size]]


def collate_cached(batch: list[dict]) -> tuple[mx.array, mx.array, mx.array]:
    max_len = max(b["hidden"].shape[0] for b in batch)
    hid_dim = batch[0]["hidden"].shape[-1]
    hidden = np.zeros((len(batch), max_len, hid_dim), dtype=np.float32)
    labels = np.full((len(batch), max_len), -100, dtype=np.int32)
    mask = np.zeros((len(batch), max_len), dtype=np.int32)
    for i, b in enumerate(batch):
        L = b["hidden"].shape[0]
        hidden[i, :L] = b["hidden"]
        labels[i, :L] = b["labels"]
        mask[i, :L] = 1
    return mx.array(hidden), mx.array(mask), mx.array(labels)


def compute_class_weights(data: list[dict], num_labels: int) -> mx.array:
    counts = np.zeros(num_labels, dtype=np.float64)
    for ex in data:
        for y in ex["labels"]:
            if y >= 0:
                counts[int(y)] += 1
    # inverse-frequency, smooth; cap O vs rare BIO tags
    counts = np.maximum(counts, 1.0)
    inv = counts.sum() / (num_labels * counts)
    # soften extreme weights
    inv = np.sqrt(inv)
    inv = inv / inv.mean()
    print("class weights:", {LABELS[i]: round(float(inv[i]), 3) for i in range(num_labels)})
    return mx.array(inv.astype(np.float32))


def weighted_ce(
    logits: mx.array,
    labels: mx.array,
    class_weights: mx.array,
    label_smoothing: float = 0.05,
) -> mx.array:
    """Token CE with class weights + label smoothing; ignore labels < 0."""
    b, t, c = logits.shape
    logits = logits.reshape(b * t, c)
    labels = labels.reshape(b * t)
    valid = labels >= 0
    safe = mx.where(valid, labels, mx.zeros_like(labels))

    log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    # one-hot with smoothing
    eye = mx.eye(c)
    oh = eye[safe]
    if label_smoothing > 0:
        oh = (1.0 - label_smoothing) * oh + label_smoothing / c
    nll = -(oh * log_probs).sum(axis=-1)
    w = class_weights[safe]
    nll = nll * w
    nll = mx.where(valid, nll, mx.zeros_like(nll))
    denom = mx.maximum(valid.astype(mx.float32).sum(), mx.array(1.0))
    return nll.sum() / denom


def cosine_lr(step: int, total: int, base_lr: float, warmup: int) -> float:
    if step < warmup:
        return base_lr * float(step + 1) / float(max(1, warmup))
    progress = (step - warmup) / float(max(1, total - warmup))
    progress = min(max(progress, 0.0), 1.0)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def evaluate_head(
    head: TokenClassificationHead,
    data: list[dict],
    batch_size: int,
) -> dict:
    all_pred, all_gold = [], []
    for batch in batches(data, batch_size, shuffle=False, seed=0):
        hidden, attention_mask, labels = collate_cached(batch)
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

    train_raw = load_split(train_path)
    val_raw = load_split(val_path)
    print(f"Train sequences: {len(train_raw)} | Val: {len(val_raw)}")

    device = device_torch()
    print(f"Frozen encoder device: {device} | Head trainer: MLX")

    tokenizer = AutoTokenizer.from_pretrained(PROC / "tokenizer")
    encoder = AutoModel.from_pretrained(args.model_name)
    encoder.resize_token_embeddings(len(tokenizer))
    encoder.to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    train_data = cache_split_features(
        "train", train_raw, encoder, device, args.batch_size, force=args.recache
    )
    val_data = cache_split_features(
        "val", val_raw, encoder, device, args.batch_size, force=args.recache
    )
    # encoder no longer needed for training loop
    del encoder
    if device.type == "mps":
        torch.mps.empty_cache()

    hidden_size = int(train_data[0]["hidden"].shape[-1])
    head = TokenClassificationHead(
        hidden_size,
        num_labels=len(LABELS),
        dropout=args.dropout,
        intermediate_size=args.intermediate_size,
    )
    mx.eval(head.parameters())

    class_weights = compute_class_weights(train_data, len(LABELS))
    optimizer = optim.AdamW(learning_rate=args.lr, weight_decay=args.weight_decay)

    def loss_fn(model, hidden, labels):
        logits = model(hidden, training=True)
        return weighted_ce(
            logits, labels, class_weights, label_smoothing=args.label_smoothing
        )

    loss_and_grad = nn.value_and_grad(head, loss_fn)

    steps_per_epoch = max(1, math.ceil(len(train_data) / args.batch_size))
    total_steps = steps_per_epoch * args.epochs
    best_f1 = -1.0
    best_partial = -1.0
    patience_left = args.patience
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        losses = []
        for batch in batches(
            train_data, args.batch_size, shuffle=True, seed=args.seed + epoch
        ):
            lr = cosine_lr(global_step, total_steps, args.lr, args.warmup_steps)
            optimizer.learning_rate = lr
            hidden, _, labels = collate_cached(batch)
            loss, grads = loss_and_grad(head, hidden, labels)
            optimizer.update(head, grads)
            mx.eval(head.parameters(), optimizer.state)
            losses.append(float(loss))
            global_step += 1

        metrics = evaluate_head(head, val_data, args.batch_size)
        mean_loss = float(np.mean(losses)) if losses else 0.0
        print(
            f"epoch {epoch}/{args.epochs}  loss={mean_loss:.4f}  lr={lr:.2e}  "
            f"val_strict_f1={metrics['strict_f1']:.3f}  "
            f"val_partial_f1={metrics['partial_f1']:.3f}  "
            f"time={time.time() - t0:.1f}s"
        )
        improved = metrics["strict_f1"] > best_f1 + 1e-4 or (
            abs(metrics["strict_f1"] - best_f1) <= 1e-4
            and metrics["partial_f1"] > best_partial + 1e-4
        )
        if improved:
            best_f1 = metrics["strict_f1"]
            best_partial = metrics["partial_f1"]
            patience_left = args.patience
            save_head(head, out_path)
            meta = {
                "model_name": args.model_name,
                "labels": LABELS,
                "hidden_size": hidden_size,
                "best_strict_f1": best_f1,
                "best_partial_f1": best_partial,
                "epoch": epoch,
                "trainer": "mlx_head_v2_weighted_cached",
                "dropout": args.dropout,
                "label_smoothing": args.label_smoothing,
                "lang_pair_tokens": LANG_PAIR_TOKENS,
            }
            meta_path = out_path.with_name(out_path.stem + ".meta.json")
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            print(f"  saved best → {out_path} (strict_f1={best_f1:.3f})")
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stop at epoch {epoch} (patience={args.patience})")
                break

    print(f"Done. Best val strict F1={best_f1:.3f}  partial F1={best_partial:.3f}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="MLX span head training v2")
    parser.add_argument("--model-name", default="xlm-roberta-base")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=80)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--intermediate-size", type=int, default=768)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--recache", action="store_true")
    parser.add_argument(
        "--output",
        default=str(MODELS / "span_classifier.safetensors"),
    )
    args = parser.parse_args(argv)
    train(args)


if __name__ == "__main__":
    main()
