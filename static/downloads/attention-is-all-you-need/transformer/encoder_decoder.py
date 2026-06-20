"""
Section 3.1 — Encoder and Decoder Stacks

Each encoder layer:
  1. Multi-head self-attention (all positions attend to all positions)
  2. Position-wise feed-forward network
  Residual + LayerNorm after each sub-layer: LayerNorm(x + Sublayer(x))

Each decoder layer adds a third sub-layer:
  - Encoder-decoder (cross) attention: queries from decoder, keys/values from encoder
  - Self-attention uses a causal mask so position i cannot see future positions
"""

import torch
import torch.nn as nn

from transformer.attention import MultiHeadAttention
from transformer.feed_forward import PositionwiseFeedForward


class EncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_k: int,
        d_v: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, d_k, d_v, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        src_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attn_out, _ = self.self_attn(x, x, x, mask=src_mask)
        x = self.norm1(x + self.dropout(attn_out))

        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x


class DecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_k: int,
        d_v: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, d_k, d_v, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, d_k, d_v, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self_attn_out, _ = self.self_attn(x, x, x, mask=tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_out))

        cross_attn_out, _ = self.cross_attn(x, memory, memory, mask=memory_mask)
        x = self.norm2(x + self.dropout(cross_attn_out))

        ff_out = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_out))
        return x


class Encoder(nn.Module):
    def __init__(self, layer: EncoderLayer, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([layer for _ in range(num_layers)])

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor | None = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, src_mask=src_mask)
        return x


class Decoder(nn.Module):
    def __init__(self, layer: DecoderLayer, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([layer for _ in range(num_layers)])

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
        return x
