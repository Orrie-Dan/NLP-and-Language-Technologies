"""Transformer encoder over MFCC frame sequences. Ownership: Person 4.

The recurrent model (``src/models/recurrent.py``, Person 2) carries a hidden
state forward in time. This model instead lets every frame attend to every
other frame in one step (multi-head self-attention), with order injected by a
fixed sinusoidal positional encoding. Padded frames are excluded with a key
padding mask, so they never receive attention and never enter the pooled
summary.

All neural experiments must consume the shared split under ``data/splits/``.
They must not build a separate train/validation/test division.
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.evaluation.metrics import classification_metrics
from src.models.recurrent import (
    TrainResult,
    _batches,
    _collate,
    predict_proba,
    set_torch_seed,
)


@dataclass(frozen=True)
class TransformerConfig:
    name: str
    d_model: int = 128
    num_heads: int = 4
    num_layers: int = 2
    ffn_dim: int = 256
    dropout: float = 0.1
    pooling: str = "mean"
    feature_streams: int = 3
    frame_stack: int = 3
    learning_rate: float = 5e-4
    weight_decay: float = 1e-2
    warmup_epochs: int = 3
    batch_size: int = 32
    max_epochs: int = 80
    patience: int = 12
    grad_clip: float = 1.0

    def architecture_key(self) -> tuple:
        """Every field except the run name; identical keys are the same experiment."""
        values = asdict(self)
        values.pop("name")
        return tuple(sorted(values.items()))

    def to_dict(self) -> dict:
        return asdict(self)


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sine/cosine position code (Vaswani et al., 2017), added to the input."""

    def __init__(self, d_model: int, max_len: int = 4096) -> None:
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        table = torch.zeros(max_len, d_model)
        table[:, 0::2] = torch.sin(position * div_term)
        table[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("table", table, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.table[: x.shape[1]].unsqueeze(0)


class SequentialTransformerClassifier(nn.Module):
    def __init__(self, n_features: int, n_classes: int, config: TransformerConfig) -> None:
        super().__init__()
        if config.pooling not in ("mean", "cls"):
            raise ValueError(f"pooling must be 'mean' or 'cls', got {config.pooling!r}")
        if config.d_model % config.num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        self.config = config
        self.input_proj = nn.Linear(n_features, config.d_model)
        self.position = SinusoidalPositionalEncoding(config.d_model)
        self.cls_token = (
            nn.Parameter(torch.zeros(1, 1, config.d_model)) if config.pooling == "cls" else None
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.ffn_dim,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=config.num_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.head = nn.Linear(config.d_model, n_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        lengths = lengths.to(x.device)
        h = self.position(self.input_proj(x))
        steps = torch.arange(x.shape[1], device=x.device)
        padding = steps[None, :] >= lengths[:, None]  # True marks a padded frame
        if self.cls_token is not None:
            # The CLS token is never padded and carries no positional code.
            h = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), h], dim=1)
            padding = torch.cat([padding.new_zeros(x.shape[0], 1), padding], dim=1)
        h = self.norm(self.encoder(h, src_key_padding_mask=padding))
        if self.cls_token is not None:
            summary = h[:, 0]
        else:
            keep = (~padding).unsqueeze(-1)
            summary = (h * keep).sum(dim=1) / lengths[:, None]
        return self.head(self.dropout(summary))


def train_transformer(
    config: TransformerConfig,
    train_sequences: list[np.ndarray],
    y_train: np.ndarray,
    val_sequences: list[np.ndarray],
    y_val: np.ndarray,
    n_classes: int,
    seed: int,
    device: torch.device,
    log_every: int = 5,
) -> TrainResult:
    """Train with AdamW, linear warmup, and early stopping on validation macro-F1.

    Transformers are unstable at full learning rate from random init, so the
    rate ramps linearly over ``warmup_epochs``; after warmup it is halved on a
    validation plateau, as in the recurrent model. The weights from the best
    validation epoch are restored before returning. Only the training and
    validation splits are passed in.
    """
    set_torch_seed(seed)
    n_features = train_sequences[0].shape[1]
    model = SequentialTransformerClassifier(n_features, n_classes, config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=max(1, config.patience // 3)
    )
    steps_per_epoch = math.ceil(len(train_sequences) / config.batch_size)
    warmup_steps = config.warmup_epochs * steps_per_epoch
    loss_fn = nn.CrossEntropyLoss()
    rng = np.random.default_rng(seed)
    y_train_t = torch.from_numpy(y_train.astype(np.int64))

    best_f1, best_epoch, best_state, best_metrics = -1.0, 0, None, {}
    rows = []
    step = 0
    start_time = time.time()
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        total_loss, correct = 0.0, 0
        for index in _batches(len(train_sequences), config.batch_size, True, rng):
            step += 1
            if step <= warmup_steps:
                for group in optimizer.param_groups:
                    group["lr"] = config.learning_rate * step / warmup_steps
            x, lengths = _collate(train_sequences, index, device)
            target = y_train_t[index].to(device)
            optimizer.zero_grad()
            logits = model(x, lengths)
            loss = loss_fn(logits, target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            total_loss += float(loss.item()) * len(index)
            correct += int((logits.argmax(dim=1) == target).sum().item())

        val_proba = predict_proba(model, val_sequences, device)
        val_loss = float(
            nn.functional.nll_loss(
                torch.log(torch.from_numpy(val_proba).clamp_min(1e-12)),
                torch.from_numpy(y_val.astype(np.int64)),
            )
        )
        val_metrics = classification_metrics(y_val, val_proba.argmax(axis=1))
        if epoch > config.warmup_epochs:
            scheduler.step(val_metrics["macro_f1"])
        rows.append({
            "epoch": epoch,
            "train_loss": total_loss / len(train_sequences),
            "train_accuracy": correct / len(train_sequences),
            "val_loss": val_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        })
        if val_metrics["macro_f1"] > best_f1:
            best_f1, best_epoch = val_metrics["macro_f1"], epoch
            best_state = copy.deepcopy(model.state_dict())
            best_metrics = val_metrics
        if log_every and (epoch % log_every == 0 or epoch == 1):
            r = rows[-1]
            print(
                f"  [{config.name} seed={seed}] epoch {epoch:3d} "
                f"train_loss={r['train_loss']:.3f} val_loss={r['val_loss']:.3f} "
                f"val_macro_f1={r['val_macro_f1']:.3f} (best {best_f1:.3f} @ {best_epoch})",
                flush=True,
            )
        if epoch - best_epoch >= config.patience:
            break

    model.load_state_dict(best_state)
    return TrainResult(
        model=model,
        history=pd.DataFrame(rows),
        best_epoch=best_epoch,
        best_val_metrics=best_metrics,
        seconds=time.time() - start_time,
        n_parameters=sum(p.numel() for p in model.parameters()),
    )
