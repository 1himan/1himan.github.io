# Attention Is All You Need — From Scratch

A complete PyTorch reproduction of [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762) with line-by-line explanations of every section of the paper.

## Quick start

```bash
pip install -r requirements.txt
python demo.py --steps 500
```

The demo trains a small Transformer on a synthetic **copy task** (source sequence → same sequence). The full paper used WMT 2014 machine translation (~4.5M pairs); the architecture and training recipe here match the paper, scaled down for a laptop.

## Project layout

| File | Paper section |
|------|---------------|
| `transformer/attention.py` | §3.2 Scaled dot-product & multi-head attention |
| `transformer/feed_forward.py` | §3.3 Position-wise FFN |
| `transformer/positional_encoding.py` | §3.5 Positional encoding |
| `transformer/encoder_decoder.py` | §3.1 Encoder/decoder stacks |
| `transformer/masks.py` | §3.2.3 Causal & padding masks |
| `transformer/model.py` | §3.1, §3.4 Full model |
| `transformer/training.py` | §5.3–5.4 Optimizer & label smoothing |
| `transformer/config.py` | Table 3 base/big hyperparameters |
| `demo.py` | Runnable training loop |

---

# Part 1 — The Problem the Paper Solves

Before Transformers, the best sequence models (machine translation, summarization, etc.) looked like this:

```
Input sentence  →  [Encoder RNN]  →  hidden states  →  [Decoder RNN]  →  Output sentence
                         ↑                                    ↑
                   processes one                         generates one
                   token at a time                       token at a time
```

**Recurrent Neural Networks (RNNs)** and **LSTMs** read tokens left-to-right. At step `t`, the hidden state `h_t` depends on `h_{t-1}` and the current token. That dependency chain is what makes RNNs powerful — but it is also their fatal flaw for modern deep learning:

1. **No parallelism within a sequence.** You cannot compute step 50 until step 49 finishes. GPUs love parallel matrix math; RNNs force sequential loops.
2. **Long-range dependencies are hard.** Information from token 1 must flow through every intermediate state to reach token 100. Gradients vanish or explode over long paths.

**Convolutional** seq2seq models (ByteNet, ConvS2S) fixed parallelism but needed many stacked layers (or dilated convolutions) for distant tokens to "see" each other — path length grows with distance.

**Attention** (Bahdanau et al., 2014) let decoders peek directly at all encoder states, but was still bolted onto RNN backbones.

The Transformer's radical idea: **drop recurrence and convolutions entirely. Use only attention.**

---

# Part 2 — High-Level Architecture (Section 3)

The Transformer is an **encoder-decoder** — the same outer shell as classic seq2seq:

```
┌─────────────────────────────────────────────────────────────────┐
│                         TRANSFORMER                              │
│                                                                  │
│  "The cat sat"          ENCODER              memory (vectors)    │
│       ↓                    ↓                      ↓              │
│  [embed + position]  [6 identical layers]   z₁ z₂ z₃ z₄        │
│                                                                  │
│  "Le chat"             DECODER              "s'assit"            │
│       ↓                    ↓                      ↓              │
│  [embed + position]  [6 identical layers]   next-token logits    │
│                           ↑                                      │
│                    attends to encoder memory                     │
└─────────────────────────────────────────────────────────────────┘
```

**Encoder** maps input tokens `(x₁, …, xₙ)` to continuous representations `(z₁, …, zₙ)`.

**Decoder** generates output `(y₁, …, yₘ)` **one token at a time** (autoregressive). When predicting position `i`, it may only use outputs at positions `< i` — never future tokens.

Each encoder layer has **two** sub-layers:
1. Multi-head self-attention
2. Position-wise feed-forward network

Each decoder layer has **three** sub-layers:
1. Masked multi-head self-attention (causal)
2. Multi-head cross-attention (decoder queries, encoder keys/values)
3. Position-wise feed-forward network

Every sub-layer uses **residual connection + layer normalization**:

```
output = LayerNorm(x + Sublayer(x))
```

All sub-layers output dimension **d_model = 512** (base model).

---

# Part 3 — Scaled Dot-Product Attention (Section 3.2.1)

Attention takes a **query** Q, **keys** K, and **values** V. Think of it as a soft database lookup:

- Query: "What am I looking for?"
- Keys: "What does each entry index on?"
- Values: "What content does each entry hold?"

**Steps:**

1. Score compatibility: dot product of query with every key → `(seq_q, seq_k)` matrix
2. Scale by `√d_k` (explained below)
3. Softmax → attention weights (each row sums to 1)
4. Weighted sum of values

**Equation (1):**

```
Attention(Q, K, V) = softmax(QKᵀ / √d_k) · V
```

In code (`transformer/attention.py`):

```python
scores = Q @ K.transpose(-2, -1) / sqrt(d_k)
weights = softmax(scores)
output = weights @ V
```

### Why scale by √d_k?

If each component of q and k has mean 0 and variance 1, their dot product has variance **d_k**. Large d_k → large dot products → softmax saturates (one weight ≈ 1, rest ≈ 0) → tiny gradients.

Dividing by √d_k keeps variance ≈ 1 regardless of head dimension. This is why unscaled dot-product attention underperforms additive attention for large d_k (Table 3 row B in the paper).

---

# Part 4 — Multi-Head Attention (Section 3.2.2)

One attention head with full d_model=512 dimensions forces all "lookup patterns" into a single subspace. **Multi-head attention** runs h=8 smaller attentions in parallel:

```
For each head i:
  Q_i = Q · W_i^Q    (512 → 64)
  K_i = K · W_i^K    (512 → 64)
  V_i = V · W_i^V    (512 → 64)
  head_i = Attention(Q_i, K_i, V_i)

MultiHead = Concat(head_1, ..., head_8) · W^O   (512 → 512)
```

With h=8, d_k = d_v = 64. Total compute ≈ single-head at d_model=512, but the model can attend to **different representation subspaces** simultaneously — one head might track syntax, another coreference, etc. (Appendix visualizations in the paper).

**Three uses in the model (Section 3.2.3):**

| Type | Q from | K, V from | Mask |
|------|--------|-----------|------|
| Encoder self-attention | encoder | encoder | padding only |
| Decoder self-attention | decoder | decoder | padding + causal |
| Cross-attention | decoder | encoder | padding on encoder |

**Causal mask:** upper triangle set to −∞ before softmax. Position 5 cannot attend to positions 6, 7, …

```
     pos:  1  2  3  4
  1       ✓  ✗  ✗  ✗
  2       ✓  ✓  ✗  ✗
  3       ✓  ✓  ✓  ✗
  4       ✓  ✓  ✓  ✓
```

Training trick: target input is **shifted right** — decoder input `[BOS, y₁, y₂, …]` predicts `[y₁, y₂, …, EOS]`.

---

# Part 5 — Feed-Forward Network (Section 3.3)

After attention mixes information **across positions**, the FFN transforms each position **independently**:

```
FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
```

Same weights at every position (like two conv1×1 layers). Dimensions: 512 → 2048 → 512.

Intuition: attention = "who should I listen to?"; FFN = "what do I do with what I heard?"

---

# Part 6 — Embeddings & Output (Section 3.4)

Token IDs → learned embedding vectors of size d_model=512.

**Scaling:** embeddings multiplied by √d_model before adding positional encoding. Keeps embedding magnitude comparable to positional signals.

**Weight tying:** the same weight matrix is shared between:
- source embedding
- target embedding  
- final linear layer before softmax

Fewer parameters, better generalization (Press & Wolf, 2016).

Output: linear(d_model → vocab_size) → softmax → P(next token).

---

# Part 7 — Positional Encoding (Section 3.5)

Without recurrence or convolution, the model is **permutation-equivariant** — shuffling tokens would give the same result. Order must be injected explicitly.

The paper adds fixed sinusoidal vectors to embeddings:

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

- `pos` = token index (0, 1, 2, …)
- `i` = dimension pair index

Low dimensions oscillate quickly (fine position detail); high dimensions slowly (coarse position). Wavelengths form a geometric progression from 2π to 10000·2π.

**Why sinusoids?** For fixed offset k, PE(pos+k) is a linear function of PE(pos) — the model can learn relative positions via learned linear transforms in attention.

Learned positional embeddings work almost as well (Table 3 row E); sinusoids may extrapolate better to longer sequences than seen in training.

Implementation: `transformer/positional_encoding.py` precomputes a `(max_len, d_model)` buffer and adds the first `seq_len` rows to each batch.

---

# Part 8 — Why Self-Attention? (Section 4)

Table 1 compares layer types for sequence length n, dimension d:

| Layer | Per-layer complexity | Sequential ops | Max path length |
|-------|---------------------|----------------|-----------------|
| Self-attention | O(n²·d) | O(1) | O(1) |
| Recurrent | O(n·d²) | O(n) | O(n) |
| Convolutional | O(k·n·d²) | O(1) | O(log_k(n)) |

For typical MT (n < d with subword tokens), self-attention is **faster per layer** than RNNs and connects every pair of positions in **one hop** — ideal for long-range dependencies.

Trade-off: O(n²) memory/time in sequence length — why later work (Longformer, FlashAttention, etc.) focuses on efficient attention.

---

# Part 9 — Training (Section 5)

### 9.1 Data (Section 5.1)

- **WMT 2014 En-De:** ~4.5M pairs, BPE vocab ~37k (shared src/tgt)
- **WMT 2014 En-Fr:** ~36M pairs, 32k word-piece vocab
- Batches sized by **~25k source + ~25k target tokens** (not fixed sentence count)

### 9.2 Hardware (Section 5.2)

- 8× NVIDIA P100
- Base: 100k steps ≈ 12 hours (0.4 s/step)
- Big: 300k steps ≈ 3.5 days (1.0 s/step)

### 9.3 Optimizer (Section 5.3)

Adam with β₁=0.9, β₂=0.98, ε=10⁻⁹.

**Noam learning rate schedule (Equation 3):**

```
lr = d_model^(-0.5) · min(step^(-0.5), step · warmup_steps^(-1.5))
```

- Warmup: lr increases linearly for first 4000 steps
- Then decays as step^(-0.5)
- Implemented in `transformer/training.py` → `NoamScheduler`

### 9.4 Regularization (Section 5.4)

1. **Residual dropout** P_drop=0.1 on sub-layer outputs and embedding sums
2. **Label smoothing** ε_ls=0.1 — target distribution is `(1-ε)` on true class, ε/(V-1) spread over others. Hurts perplexity, helps BLEU.

### Inference (Section 6.1)

- Beam search, beam=4, length penalty α=0.6
- Max output length = input length + 50
- Checkpoint averaging (last 5 for base, last 20 for big)

---

# Part 10 — Results (Section 6)

**Table 2 — WMT 2014 newstest2014:**

| Model | EN-DE BLEU | EN-FR BLEU |
|-------|-----------|-----------|
| Previous best ensemble | ~26.3 | ~41.3 |
| **Transformer (base)** | **27.3** | **38.1** |
| **Transformer (big)** | **28.4** | **41.0** |

Base model beat all published models at a fraction of training FLOPs. Big model set new SOTA on En-De (+2 BLEU over ensembles) and matched SOTA on En-Fr at ~¼ the cost.

**Table 3 — Ablations (dev set newstest2013):**

Key findings:
- **(A)** 8 heads optimal; 1 head −0.9 BLEU
- **(B)** Smaller d_k hurts — dot product may be too "simple" a compatibility function
- **(C)** Bigger d_model/d_ff helps (big model: d=1024, d_ff=4096, h=16)
- **(D)** Dropout 0.1 critical; label smoothing helps
- **(E)** Learned vs sinusoidal position: nearly identical

---

# Part 11 — Walking Through the Code

### Forward pass during training

```python
# src: (batch, src_len)   tgt_in: (batch, tgt_len)  — shifted right
src_mask, tgt_mask, memory_mask = model.build_masks(src, tgt_in)

memory = model.encode(src, src_mask)           # encoder stack
decoder_out = model.decode(                     # decoder stack
    tgt_in, memory, tgt_mask, memory_mask
)
logits = model.output_projection(decoder_out)    # (batch, tgt_len, vocab)
loss = LabelSmoothingLoss(...)(logits, tgt_out)
```

### One encoder layer

```python
# Self-attention: every token looks at every other token (respecting pad mask)
attn_out = MultiHeadAttention(x, x, x, mask=src_mask)
x = LayerNorm(x + Dropout(attn_out))

# FFN: per-token MLP
ff_out = FFN(x)
x = LayerNorm(x + Dropout(ff_out))
```

### One decoder layer

```python
x = LayerNorm(x + Dropout(SelfAttn(x, x, x, causal_mask)))
x = LayerNorm(x + Dropout(CrossAttn(x, encoder_memory, encoder_memory)))
x = LayerNorm(x + Dropout(FFN(x)))
```

---

# Part 12 — Hyperparameter Reference

### Base model (Table 3)

| Parameter | Value |
|-----------|-------|
| N (layers) | 6 |
| d_model | 512 |
| d_ff | 2048 |
| h (heads) | 8 |
| d_k, d_v | 64 |
| P_drop | 0.1 |
| ε_ls | 0.1 |
| train steps | 100k |
| params | ~65M |

### Big model

| Parameter | Value |
|-----------|-------|
| d_model | 1024 |
| d_ff | 4096 |
| h | 16 |
| P_drop | 0.3 (0.1 for En-Fr) |
| train steps | 300k |
| params | ~213M |

Use `TransformerConfig.base()` or `TransformerConfig.big()` in code.

---

# Part 13 — What This Paper Unlocked

The Transformer was originally a machine translation model. Its legacy:

- **BERT** (encoder-only) — language understanding
- **GPT** (decoder-only) — generative pre-training
- **T5, BART** — unified text-to-text
- Modern LLMs (GPT-4, Claude, Llama, etc.) — same core block: multi-head self-attention + FFN + residuals

The title is literal: for sequence modeling, **attention mechanisms alone** — no RNN, no CNN in the core — are enough.

---

# References

- Vaswani, A., et al. (2017). *Attention Is All You Need.* NeurIPS.
- Original tensor2tensor: https://github.com/tensorflow/tensor2tensor
- Annotated paper: `Attention Is All You Need annotated.pdf`
