"""
Section 5 — Training utilities

  5.3 Optimizer: Adam with custom learning-rate schedule (Equation 3)
  5.4 Regularization: label smoothing (epsilon_ls = 0.1)
"""

import math

import torch
import torch.nn as nn


class NoamScheduler:
    """
    Equation (3):

        lrate = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))

    Linear warmup for warmup_steps, then inverse-square-root decay.
    Paper uses warmup_steps = 4000.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        d_model: int,
        warmup_steps: int = 4000,
    ):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self._step = 0

    def step(self):
        self._step += 1
        lr = self._learning_rate(self._step)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _learning_rate(self, step: int) -> float:
        step = max(step, 1)
        factor = self.d_model ** -0.5
        scale = min(step ** -0.5, step * (self.warmup_steps ** -1.5))
        return factor * scale


def build_adam_optimizer(
    model: nn.Module,
    d_model: int,
    betas: tuple[float, float] = (0.9, 0.98),
    eps: float = 1e-9,
    warmup_steps: int = 4000,
) -> tuple[torch.optim.Adam, NoamScheduler]:
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0, betas=betas, eps=eps)
    scheduler = NoamScheduler(optimizer, d_model=d_model, warmup_steps=warmup_steps)
    return optimizer, scheduler


class LabelSmoothingLoss(nn.Module):
    """
    Section 5.4 — Label smoothing with epsilon_ls = 0.1.

    Instead of a hard one-hot target, distribute (1 - eps) mass on the true
    token and eps / (vocab_size - 1) on all other non-padding tokens.
    """

    def __init__(
        self,
        vocab_size: int,
        padding_idx: int = 0,
        smoothing: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, vocab_size = logits.shape
        logits = logits.view(-1, vocab_size)
        target = target.view(-1)

        log_probs = nn.functional.log_softmax(logits, dim=-1)

        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (vocab_size - 2))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
            true_dist[:, self.padding_idx] = 0.0
            mask = target == self.padding_idx
            true_dist[mask] = 0.0

        loss = -(true_dist * log_probs).sum(dim=-1)
        loss = loss.masked_fill(mask, 0.0)
        return loss.sum() / (~mask).sum().clamp(min=1)
