from dataclasses import dataclass


@dataclass
class TransformerConfig:
    """Hyperparameters from Section 3 and Table 3 (base model) of the paper."""

    vocab_size: int = 37000
    num_layers: int = 6
    d_model: int = 512
    d_ff: int = 2048
    num_heads: int = 8
    d_k: int = 64
    d_v: int = 64
    dropout: float = 0.1
    max_seq_len: int = 512
    pad_idx: int = 0

    @classmethod
    def base(cls, vocab_size: int = 37000, pad_idx: int = 0) -> "TransformerConfig":
        return cls(vocab_size=vocab_size, pad_idx=pad_idx)

    @classmethod
    def big(cls, vocab_size: int = 37000, pad_idx: int = 0) -> "TransformerConfig":
        return cls(
            vocab_size=vocab_size,
            pad_idx=pad_idx,
            d_model=1024,
            d_ff=4096,
            num_heads=16,
            d_k=64,
            d_v=64,
            dropout=0.3,
        )
