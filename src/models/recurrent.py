"""Recurrent sequence classifier (BiGRU / BiLSTM). Ownership: Person 2.

The network reads the MFCC frame sequence in time order. Variable lengths are
handled with packed sequences, so padded steps never update the hidden state
and never enter the pooled summary.
"""

from __future__ import annotations

import copy
import random
import time
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

from src.evaluation.metrics import classification_metrics


@dataclass(frozen=True)
class RecurrentConfig:
    name: str
    cell: str = "gru"
    bidirectional: bool = True
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.3
    pooling: str = "mean"
    feature_streams: int = 3
    frame_stack: int = 1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 32
    max_epochs: int = 60
    patience: int = 10
    grad_clip: float = 1.0

    def architecture_key(self) -> tuple:
        """Every field except the run name; identical keys are the same experiment."""
        values = asdict(self)
        values.pop("name")
        return tuple(sorted(values.items()))

    def to_dict(self) -> dict:
        return asdict(self)


class RecurrentClassifier(nn.Module):
    def __init__(self, n_features: int, n_classes: int, config: RecurrentConfig) -> None:
        super().__init__()
        if config.cell not in ("gru", "lstm"):
            raise ValueError(f"cell must be 'gru' or 'lstm', got {config.cell!r}")
        if config.pooling not in ("mean", "last"):
            raise ValueError(f"pooling must be 'mean' or 'last', got {config.pooling!r}")
        rnn_class = nn.GRU if config.cell == "gru" else nn.LSTM
        self.config = config
        self.rnn = rnn_class(
            input_size=n_features,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            bidirectional=config.bidirectional,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
        )
        n_directions = 2 if config.bidirectional else 1
        self.dropout = nn.Dropout(config.dropout)
        self.head = nn.Linear(config.hidden_size * n_directions, n_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        output, hidden = self.rnn(packed)
        if self.config.pooling == "last":
            h_n = hidden[0] if isinstance(hidden, tuple) else hidden
            # Top layer only: the forward state ends at the true last frame and
            # the backward state ends at the first frame.
            summary = (
                torch.cat([h_n[-2], h_n[-1]], dim=1) if self.config.bidirectional else h_n[-1]
            )
        else:
            padded, _ = pad_packed_sequence(output, batch_first=True)
            steps = torch.arange(padded.shape[1], device=padded.device)
            mask = (steps[None, :] < lengths.to(padded.device)[:, None]).unsqueeze(-1)
            summary = (padded * mask).sum(dim=1) / lengths.to(padded.device)[:, None]
        return self.head(self.dropout(summary))


def set_torch_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def pick_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _batches(n: int, batch_size: int, shuffle: bool, rng: np.random.Generator):
    order = rng.permutation(n) if shuffle else np.arange(n)
    for start in range(0, n, batch_size):
        yield order[start : start + batch_size]


def _collate(sequences: list[np.ndarray], index: np.ndarray, device: torch.device):
    tensors = [torch.from_numpy(sequences[i]) for i in index]
    lengths = torch.tensor([t.shape[0] for t in tensors], dtype=torch.long)
    return pad_sequence(tensors, batch_first=True).to(device), lengths


@torch.no_grad()
def predict_proba(
    model: RecurrentClassifier,
    sequences: list[np.ndarray],
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    model.eval()
    outputs = []
    for index in _batches(len(sequences), batch_size, False, np.random.default_rng(0)):
        x, lengths = _collate(sequences, index, device)
        outputs.append(torch.softmax(model(x, lengths), dim=1).cpu().numpy())
    return np.concatenate(outputs, axis=0)


@dataclass
class TrainResult:
    model: RecurrentClassifier
    history: pd.DataFrame
    best_epoch: int
    best_val_metrics: dict
    seconds: float
    n_parameters: int


def train_recurrent(
    config: RecurrentConfig,
    train_sequences: list[np.ndarray],
    y_train: np.ndarray,
    val_sequences: list[np.ndarray],
    y_val: np.ndarray,
    n_classes: int,
    seed: int,
    device: torch.device,
    log_every: int = 5,
) -> TrainResult:
    """Train with Adam and early stopping on validation macro-F1.

    The weights from the best validation epoch are restored before returning.
    Only the training and validation splits are passed in.
    """
    set_torch_seed(seed)
    n_features = train_sequences[0].shape[1]
    model = RecurrentClassifier(n_features, n_classes, config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=max(1, config.patience // 3)
    )
    loss_fn = nn.CrossEntropyLoss()
    rng = np.random.default_rng(seed)
    y_train_t = torch.from_numpy(y_train.astype(np.int64))

    best_f1, best_epoch, best_state, best_metrics = -1.0, 0, None, {}
    rows = []
    start_time = time.time()
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        total_loss, correct = 0.0, 0
        for index in _batches(len(train_sequences), config.batch_size, True, rng):
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
