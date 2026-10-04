"""Transformer Encoder sequence classifier. Ownership: Person 4.

The network reads MFCC frame sequences and uses self-attention to model
relationships between different positions in the acoustic sequence.

Variable-length sequences are padded within each batch. A padding mask is
passed to the Transformer so padded time steps are ignored. Masked mean
pooling is then used to create one representation per audio clip.
"""

from __future__ import annotations

import copy
import math
import random
import time
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from src.evaluation.metrics import classification_metrics


@dataclass(frozen=True)
class TransformerConfig:
    """Configuration for one Transformer experiment."""

    name: str = "transformer_baseline"
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 256
    dropout: float = 0.2
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 32
    max_epochs: int = 30
    patience: int = 8
    grad_clip: float = 1.0

    def architecture_key(self) -> tuple:
        """Every field except the run name; identical keys are one experiment."""
        values = asdict(self)
        values.pop("name")
        return tuple(sorted(values.items()))

    def to_dict(self) -> dict:
        return asdict(self)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer input sequences."""

    def __init__(self, d_model: int, max_len: int = 4096) -> None:
        super().__init__()

        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )

        encoding = torch.zeros(max_len, d_model)
        encoding[:, 0::2] = torch.sin(position * div_term)

        # For odd d_model values, there is one fewer odd column.
        encoding[:, 1::2] = torch.cos(
            position * div_term[: encoding[:, 1::2].shape[1]]
        )

        self.register_buffer("encoding", encoding.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional information to a (batch, time, feature) tensor."""
        return x + self.encoding[:, : x.shape[1]]


class TransformerClassifier(nn.Module):
    """Transformer Encoder classifier for variable-length acoustic sequences."""

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        config: TransformerConfig,
    ) -> None:
        super().__init__()

        if config.d_model % config.nhead != 0:
            raise ValueError(
                f"d_model={config.d_model} must be divisible by "
                f"nhead={config.nhead}"
            )

        self.config = config

        # Convert the acoustic feature dimension into the Transformer
        # representation dimension.
        self.input_projection = nn.Linear(n_features, config.d_model)

        self.position = PositionalEncoding(config.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers,
            # The nested-tensor fast path is unavailable with norm_first=True anyway;
            # turning it off explicitly avoids a warning every time a model is built.
            enable_nested_tensor=False,
        )

        self.dropout = nn.Dropout(config.dropout)

        self.head = nn.Linear(config.d_model, n_classes)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Return class logits for a padded batch of sequences."""

        # x shape:
        # (batch, time, n_features)

        x = self.input_projection(x)

        x = self.position(x)

        # Create a padding mask.
        #
        # Transformer expects:
        # True  -> ignore this position
        # False -> real sequence position
        steps = torch.arange(
            x.shape[1],
            device=x.device,
        )

        padding_mask = (
            steps.unsqueeze(0)
            >= lengths.to(x.device).unsqueeze(1)
        )

        x = self.encoder(
            x,
            src_key_padding_mask=padding_mask,
        )

        # Masked mean pooling.
        # Padding positions must not contribute to the clip representation.
        valid_mask = (~padding_mask).unsqueeze(-1)

        x = x.masked_fill(~valid_mask, 0.0)

        lengths_float = lengths.to(x.device).float().unsqueeze(1)

        summary = x.sum(dim=1) / lengths_float

        return self.head(self.dropout(summary))


def set_torch_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def pick_device(requested: str = "auto") -> torch.device:
    """Choose CUDA, then Apple MPS, then CPU unless a device is explicitly requested."""

    if requested != "auto":
        return torch.device(requested)

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def _batches(
    n: int,
    batch_size: int,
    shuffle: bool,
    rng: np.random.Generator,
):
    """Yield arrays of example indices."""

    order = rng.permutation(n) if shuffle else np.arange(n)

    for start in range(0, n, batch_size):
        yield order[start : start + batch_size]


def _collate(
    sequences: list[np.ndarray],
    index: np.ndarray,
    device: torch.device,
):
    """Pad variable-length sequences into one batch."""

    tensors = [
        torch.from_numpy(sequences[i])
        for i in index
    ]

    lengths = torch.tensor(
        [tensor.shape[0] for tensor in tensors],
        dtype=torch.long,
    )

    padded = pad_sequence(
        tensors,
        batch_first=True,
    )

    return padded.to(device), lengths.to(device)


@torch.no_grad()
def predict_proba(
    model: TransformerClassifier,
    sequences: list[np.ndarray],
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    """Return class probabilities for a collection of sequences."""

    model.eval()

    outputs = []

    for index in _batches(
        len(sequences),
        batch_size,
        False,
        np.random.default_rng(0),
    ):
        x, lengths = _collate(
            sequences,
            index,
            device,
        )

        logits = model(x, lengths)

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

        outputs.append(
            probabilities.cpu().numpy()
        )

    return np.concatenate(outputs, axis=0)


@dataclass
class TrainResult:
    """Results returned after training."""

    model: TransformerClassifier
    history: pd.DataFrame
    best_epoch: int
    best_val_metrics: dict
    seconds: float
    n_parameters: int


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
    """Train Transformer with early stopping on validation macro-F1.

    Only the training and validation splits are used during model fitting.
    The best validation checkpoint is restored before returning.
    """

    set_torch_seed(seed)

    n_features = train_sequences[0].shape[1]

    model = TransformerClassifier(
        n_features=n_features,
        n_classes=n_classes,
        config=config,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=max(1, config.patience // 3),
    )

    loss_fn = nn.CrossEntropyLoss()

    rng = np.random.default_rng(seed)

    y_train_t = torch.from_numpy(
        y_train.astype(np.int64)
    )

    best_f1 = -1.0
    best_epoch = 0
    best_state = None
    best_metrics = {}

    rows = []

    start_time = time.time()

    for epoch in range(1, config.max_epochs + 1):

        model.train()

        total_loss = 0.0
        correct = 0

        for index in _batches(
            len(train_sequences),
            config.batch_size,
            True,
            rng,
        ):
            x, lengths = _collate(
                train_sequences,
                index,
                device,
            )

            target = y_train_t[index].to(device)

            optimizer.zero_grad()

            logits = model(
                x,
                lengths,
            )

            loss = loss_fn(
                logits,
                target,
            )

            loss.backward()

            nn.utils.clip_grad_norm_(
                model.parameters(),
                config.grad_clip,
            )

            optimizer.step()

            total_loss += (
                float(loss.item()) * len(index)
            )

            correct += int(
                (
                    logits.argmax(dim=1) == target
                ).sum().item()
            )

        # Validation
        val_proba = predict_proba(
            model,
            val_sequences,
            device,
        )

        val_loss = float(
            nn.functional.nll_loss(
                torch.log(
                    torch.from_numpy(
                        val_proba
                    ).clamp_min(1e-12)
                ),
                torch.from_numpy(
                    y_val.astype(np.int64)
                ),
            )
        )

        val_metrics = classification_metrics(
            y_val,
            val_proba.argmax(axis=1),
            val_proba,
        )

        scheduler.step(
            val_metrics["macro_f1"]
        )

        rows.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / len(train_sequences),
                "train_accuracy": correct / len(train_sequences),
                "val_loss": val_loss,
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )

        # Select ONLY using validation macro-F1.
        if val_metrics["macro_f1"] > best_f1:

            best_f1 = val_metrics["macro_f1"]

            best_epoch = epoch

            best_state = copy.deepcopy(
                model.state_dict()
            )

            best_metrics = val_metrics

        if log_every and (
            epoch % log_every == 0
            or epoch == 1
        ):
            row = rows[-1]

            print(
                f"  [{config.name} seed={seed}] "
                f"epoch {epoch:3d} "
                f"train_loss={row['train_loss']:.3f} "
                f"val_loss={row['val_loss']:.3f} "
                f"val_macro_f1={row['val_macro_f1']:.3f} "
                f"(best {best_f1:.3f} @ {best_epoch})",
                flush=True,
            )

        # Early stopping.
        if epoch - best_epoch >= config.patience:
            break

    if best_state is None:
        raise RuntimeError(
            "Training finished without producing a validation checkpoint."
        )

    model.load_state_dict(best_state)

    return TrainResult(
        model=model,
        history=pd.DataFrame(rows),
        best_epoch=best_epoch,
        best_val_metrics=best_metrics,
        seconds=time.time() - start_time,
        n_parameters=sum(
            parameter.numel()
            for parameter in model.parameters()
        ),
    )