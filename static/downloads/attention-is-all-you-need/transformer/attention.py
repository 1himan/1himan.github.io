"""
Section 3.2 — Attention

The paper defines attention as: given a query and a set of key-value pairs,
produce an output that is a weighted sum of the values, where each weight
comes from a compatibility score between the query and the corresponding key.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
    dropout: nn.Dropout | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Equation (1): Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

    Shapes:
        query: (batch, heads, seq_q, d_k)
        key:   (batch, heads, seq_k, d_k)
        value: (batch, heads, seq_k, d_v)
        mask:  broadcastable to (batch, 1, seq_q, seq_k); True = keep, False = block
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    weights = F.softmax(scores, dim=-1)

    if dropout is not None:
        weights = dropout(weights)

    output = torch.matmul(weights, value)
    return output, weights


class MultiHeadAttention(nn.Module):
    """
    Section 3.2.2 — Multi-Head Attention

    Instead of one attention with full d_model dimensions, project Q, K, V into
    h smaller subspaces (heads), run attention in parallel, concatenate, and
    project back to d_model.

    head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_k: int,
        d_v: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_k
        self.d_v = d_v

        self.w_q = nn.Linear(d_model, num_heads * d_k, bias=False)
        self.w_k = nn.Linear(d_model, num_heads * d_k, bias=False)
        self.w_v = nn.Linear(d_model, num_heads * d_v, bias=False)
        self.w_o = nn.Linear(num_heads * d_v, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = query.size(0)

        q = self._split_heads(self.w_q(query), self.d_k)
        k = self._split_heads(self.w_k(key), self.d_k)
        v = self._split_heads(self.w_v(value), self.d_v)

        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(1)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)

        attn_output, attn_weights = scaled_dot_product_attention(
            q, k, v, mask=mask, dropout=self.dropout
        )

        concat = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.num_heads * self.d_v
        )
        return self.w_o(concat), attn_weights

    def _split_heads(self, x: torch.Tensor, head_dim: int) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, head_dim)
        return x.transpose(1, 2)
