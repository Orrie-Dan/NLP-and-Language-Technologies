"""Person 3: temporal CNN classifier for MFCC frame sequences."""

import torch
from torch import nn


class TemporalCNNClassifier(nn.Module):
    def __init__(
        self,
        n_features,
        n_classes,
        channels=64,
        kernel_size=5,
        dropout=0.2,
    ):
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be positive and odd.")

        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(
            n_features, channels, kernel_size, padding=padding
        )
        self.conv2 = nn.Conv1d(
            channels, channels, kernel_size, padding=padding
        )
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(channels, n_classes)

    def forward(self, x, lengths):
        # Input shape: (batch, time, features)
        lengths = lengths.to(x.device)
        steps = torch.arange(x.shape[1], device=x.device)
        mask = (steps[None, :] < lengths[:, None]).unsqueeze(1)

        x = x.transpose(1, 2).masked_fill(~mask, 0.0)
        
        x = self.dropout(self.activation(self.conv1(x)))
        x = x.masked_fill(~mask, 0.0)

        x = self.dropout(self.activation(self.conv2(x)))
        x = x.masked_fill(~mask, 0.0)

        summary = x.sum(dim=2) / lengths[:, None]
        return self.head(self.dropout(summary))