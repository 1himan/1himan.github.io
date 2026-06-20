"""
Minimal training demo for the Transformer.

The full paper trains on WMT 2014 (~4.5M sentence pairs). This script uses a
tiny synthetic copy task so you can verify the implementation runs end-to-end
on CPU or GPU in minutes.
"""

import argparse
import random

import torch
from torch.utils.data import DataLoader, Dataset

from transformer.config import TransformerConfig
from transformer.model import Transformer
from transformer.training import LabelSmoothingLoss, build_adam_optimizer


SPECIAL = {"pad": 0, "bos": 1, "eos": 2}
DATA_START = 3


class CopyDataset(Dataset):
    """Maps random integer sequences: source -> identical target (translation toy task)."""

    def __init__(self, num_samples: int, min_len: int, max_len: int, vocab_size: int):
        self.samples = []
        token_vocab = list(range(DATA_START, vocab_size))
        for _ in range(num_samples):
            length = random.randint(min_len, max_len)
            tokens = random.choices(token_vocab, k=length)
            src = [SPECIAL["bos"], *tokens, SPECIAL["eos"]]
            tgt = [SPECIAL["bos"], *tokens, SPECIAL["eos"]]
            self.samples.append((src, tgt))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_batch(batch, pad_idx: int):
    src_batch, tgt_batch = zip(*batch)
    src_len = max(len(s) for s in src_batch)
    tgt_len = max(len(t) for t in tgt_batch)

    src = torch.full((len(batch), src_len), pad_idx, dtype=torch.long)
    tgt_in = torch.full((len(batch), tgt_len), pad_idx, dtype=torch.long)
    tgt_out = torch.full((len(batch), tgt_len), pad_idx, dtype=torch.long)

    for i, (s, t) in enumerate(batch):
        src[i, : len(s)] = torch.tensor(s)
        tgt_in[i, : len(t)] = torch.tensor(t)
        tgt_out[i, : len(t) - 1] = torch.tensor(t[1:])
        tgt_out[i, len(t) - 1 :] = pad_idx

    return src, tgt_in[:, :-1], tgt_out[:, :-1]


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Device: {device}")

    config = TransformerConfig(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        d_model=args.d_model,
        d_ff=args.d_ff,
        num_heads=args.num_heads,
        d_k=args.d_model // args.num_heads,
        d_v=args.d_model // args.num_heads,
        dropout=args.dropout,
        pad_idx=SPECIAL["pad"],
    )

    model = Transformer(config).to(device)
    criterion = LabelSmoothingLoss(
        vocab_size=config.vocab_size,
        padding_idx=config.pad_idx,
        smoothing=args.label_smoothing,
    )
    optimizer, scheduler = build_adam_optimizer(
        model,
        d_model=config.d_model,
        warmup_steps=args.warmup_steps,
    )

    dataset = CopyDataset(
        num_samples=args.num_samples,
        min_len=5,
        max_len=args.max_len,
        vocab_size=config.vocab_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_batch(b, SPECIAL["pad"]),
    )

    model.train()
    for step in range(1, args.steps + 1):
        src, tgt_in, tgt_out = next(iter(loader))
        src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)

        src_mask, tgt_mask, memory_mask = model.build_masks(src, tgt_in)
        logits = model(src, tgt_in, src_mask, tgt_mask, memory_mask)
        loss = criterion(logits, tgt_out)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % args.log_every == 0 or step == 1:
            lr = scheduler.optimizer.param_groups[0]["lr"]
            print(f"step {step:5d} | loss {loss.item():.4f} | lr {lr:.2e}")

    model.eval()
    src, _, _ = collate_batch([dataset[0]], SPECIAL["pad"])
    src = src.to(device)
    decoded = model.greedy_decode(
        src,
        max_len=args.max_len + 4,
        bos_idx=SPECIAL["bos"],
        eos_idx=SPECIAL["eos"],
    )
    print("\nSample decode:")
    print(f"  source:  {src[0].tolist()}")
    print(f"  decoded: {decoded[0].tolist()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Transformer on a copy task")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-samples", type=int, default=2000)
    parser.add_argument("--max-len", type=int, default=20)
    parser.add_argument("--vocab-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--d-ff", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--cpu", action="store_true")
    train(parser.parse_args())
