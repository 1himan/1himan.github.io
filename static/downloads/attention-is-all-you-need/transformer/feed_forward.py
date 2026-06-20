"""
Section 3.3 — Position-wise Feed-Forward Networks

Applied identically at every token position (like two 1x1 convolutions):

    FFN(x) = max(0, x W1 + b1) W2 + b2

Paper defaults: d_model=512, d_ff=2048.
"""

import torch.nn as nn


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(nn.functional.relu(self.linear1(x))))
