"""
Full Transformer model — Sections 3.1, 3.4, 3.5

Encoder-decoder for sequence transduction:
  source tokens -> encoder -> memory
  shifted target tokens -> decoder (+ memory) -> logits -> softmax
"""

import math

import torch
import torch.nn as nn

from transformer.config import TransformerConfig
from transformer.encoder_decoder import Decoder, DecoderLayer, Encoder, EncoderLayer
from transformer.masks import combine_masks, make_causal_mask, make_pad_mask
from transformer.positional_encoding import PositionalEncoding


class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        d = config.d_model

        self.src_embed = nn.Embedding(config.vocab_size, d, padding_idx=config.pad_idx)
        self.tgt_embed = nn.Embedding(config.vocab_size, d, padding_idx=config.pad_idx)
        self.positional_encoding = PositionalEncoding(
            d, max_len=config.max_seq_len, dropout=config.dropout
        )

        encoder_layer = EncoderLayer(
            d, config.num_heads, config.d_k, config.d_v, config.d_ff, config.dropout
        )
        decoder_layer = DecoderLayer(
            d, config.num_heads, config.d_k, config.d_v, config.d_ff, config.dropout
        )
        self.encoder = Encoder(encoder_layer, config.num_layers)
        self.decoder = Decoder(decoder_layer, config.num_layers)

        self.output_projection = nn.Linear(d, config.vocab_size, bias=False)
        self.output_projection.weight = self.tgt_embed.weight

        self._init_parameters()

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def embed_scale(self, x: torch.Tensor) -> torch.Tensor:
        """Section 3.4: multiply embeddings by sqrt(d_model)."""
        return x * math.sqrt(self.config.d_model)

    def encode(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.embed_scale(self.src_embed(src))
        x = self.positional_encoding(x)
        return self.encoder(x, src_mask=src_mask)

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.embed_scale(self.tgt_embed(tgt))
        x = self.positional_encoding(x)
        return self.decoder(x, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        memory = self.encode(src, src_mask=src_mask)
        decoder_out = self.decode(
            tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask
        )
        return self.output_projection(decoder_out)

    def build_masks(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pad_idx = self.config.pad_idx
        device = src.device

        src_pad = make_pad_mask(src, pad_idx)
        memory_pad = make_pad_mask(src, pad_idx)

        tgt_pad = make_pad_mask(tgt, pad_idx)
        causal = make_causal_mask(tgt.size(1), device)
        tgt_mask = combine_masks(tgt_pad, causal)

        return src_pad, tgt_mask, memory_pad

    @torch.no_grad()
    def greedy_decode(
        self,
        src: torch.Tensor,
        max_len: int,
        bos_idx: int,
        eos_idx: int,
    ) -> torch.Tensor:
        """Autoregressive decoding (Section 1): one token at a time."""
        self.eval()
        batch_size = src.size(0)
        device = src.device

        src_mask, _, memory_mask = self.build_masks(
            src, torch.full((batch_size, 1), bos_idx, device=device, dtype=torch.long)
        )
        memory = self.encode(src, src_mask=src_mask)

        ys = torch.full((batch_size, 1), bos_idx, dtype=torch.long, device=device)

        for _ in range(max_len - 1):
            tgt_pad = make_pad_mask(ys, self.config.pad_idx)
            causal = make_causal_mask(ys.size(1), device)
            tgt_mask = combine_masks(tgt_pad, causal)

            out = self.decode(ys, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
            logits = self.output_projection(out[:, -1, :])
            next_token = logits.argmax(dim=-1, keepdim=True)
            ys = torch.cat([ys, next_token], dim=1)

            if (next_token == eos_idx).all():
                break

        return ys
