"""Attention masks used during training and inference (Section 3.2.3)."""

import torch


def make_pad_mask(seq: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """
    Padding mask: True for real tokens, False for padding.

    Shape: (batch, 1, 1, seq_len) — broadcast over heads and query positions.
    """
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)


def make_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Causal (look-ahead) mask for decoder self-attention.

    Position i may only attend to positions <= i. Future positions are masked
    with -inf before softmax (Section 3.2.3).
    """
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
    return mask.unsqueeze(0).unsqueeze(0)


def combine_masks(*masks: torch.Tensor) -> torch.Tensor:
    """Logical AND of multiple boolean masks."""
    combined = masks[0]
    for mask in masks[1:]:
        combined = combined & mask
    return combined
