# Module 11: The Transformer Block

## Overview

In Module 10 we built the attention mechanism — a soft database lookup. Now we wrap
it in the full **transformer block**: the repeating unit that stacks to form GPT,
BERT, and every modern LLM.

A transformer block is a **middleware stack**: the request (token embeddings) passes
through attention, normalization, and a feed-forward network, with the original input
added back at each stage (residual connections). Stack N of these and you have an LLM.

---

## The Full Data Flow

```
Input x  (shape: seq_len × d_model)
    │
    ▼
┌─────────────┐
│  LayerNorm  │  ← normalize before feeding into attention
└──────┬──────┘
       │
    ▼
┌────────────────────┐
│  Multi-Head        │
│  Attention (MHA)   │  ← the soft database lookup from Module 10
└──────┬─────────────┘
       │
    ▼
   + x  ◄─── Residual connection #1 (add original input back)
    │
    ▼
┌─────────────┐
│  LayerNorm  │  ← normalize again before FFN
└──────┬──────┘
       │
    ▼
┌──────────────────────────┐
│  Feed-Forward Network    │
│  FFN(x) = Linear→GELU    │  ← two linear layers with activation
│           →Linear        │
└──────┬───────────────────┘
       │
    ▼
   + x  ◄─── Residual connection #2 (add back again)
    │
    ▼
Output  (same shape: seq_len × d_model)
```

Each block produces output with the **same shape** as its input. You can chain N of
them together and the shapes flow through unchanged.

---

## Component 1: Layer Normalization

LayerNorm normalizes each token's embedding independently:

```python
# For each token embedding vector x of length d_model:
mean = x.mean()
std  = x.std()
x_norm = (x - mean) / (std + epsilon)   # zero mean, unit variance
output = gamma * x_norm + beta          # learned scale and shift
```

> **Backend analogy:** Z-score normalization for database column values before
> passing them into a ranking function — ensures no single dimension dominates
> just because it has a larger scale.

**Why LayerNorm here specifically?**
- **BatchNorm** normalizes across the batch dimension — problematic for variable-length
  sequences and single-item inference.
- **LayerNorm** normalizes across the feature dimension of a single sample — works
  identically at batch size 1 or 1000.
- Position: applied **before** the sub-layer (Pre-Norm), not after. Pre-Norm trains
  more stably for deep networks.

---

## Component 2: Multi-Head Attention (MHA)

Already covered in Module 10. The block just slots it in:

```python
attn_output = MHA(LayerNorm(x))   # attention on normalized input
x = x + attn_output               # residual: add back original
```

The key new idea here is the **residual connection** — covered below.

---

## Component 3: Feed-Forward Network (FFN)

After attention, every token passes through a small 2-layer neural network
**independently** (no cross-token interaction here):

```python
FFN(x) = Linear_2( GELU( Linear_1(x) ) )
```

- `Linear_1`: projects from `d_model` → `d_ff`  (typically `d_ff = 4 × d_model`)
- `GELU`:     activation function (smooth version of ReLU)
- `Linear_2`: projects back `d_ff` → `d_model`

> **Backend analogy:** The FFN is a **per-record transform** in a data pipeline —
> unlike attention (which joins rows), the FFN processes each row independently.
> It's where the model stores **factual knowledge** learned during pre-training.
> Researchers have shown that factual recall (e.g., "Paris is the capital of France")
> lives primarily in the FFN weights, not the attention weights.

**Why 4× expansion?**

The widening to `4 × d_model` creates a larger "working memory" for each token
to process information. GPT-3 uses `d_model=12288`, `d_ff=49152`.

---

## Component 4: Residual Connections

After each sub-layer (MHA and FFN), the original input is added back:

```python
x = x + sublayer(LayerNorm(x))
```

> **Backend analogy:** Like a middleware that either transforms the request or
> passes it through unchanged. If the sub-layer learns to output zeros,
> the block becomes an identity — it simply passes the input forward.
> This is a safety net: even a poorly initialized block can't *break* the signal.

**Why residuals matter:**

1. **Gradient flow**: gradients can skip directly through the `+x` path without
   passing through the sub-layer, making deep networks trainable.
2. **Refinement, not replacement**: each block *refines* token representations
   rather than replacing them — the original information is always preserved.
3. **Identity initialization**: at the start of training, small random weights
   make sub-layers output ~0, so the whole network starts near identity.

Without residuals, training networks >10 layers deep was nearly impossible (the
vanishing gradient problem that plagued early deep learning).

---

## Component 5: Positional Encoding

Transformers have **no built-in notion of position**. Attention is a set operation —
it treats the sequence as an unordered bag of tokens unless you explicitly encode order.

Solution: add **positional encodings** to the token embeddings before the first block.

### Sinusoidal Positional Encoding (original "Attention is All You Need"):

```
PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
```

- `pos`: position in the sequence (0, 1, 2, ...)
- `i`: dimension index (0, 1, ..., d_model/2 - 1)
- Different frequencies for different dimensions — like a multi-scale clock

> **Backend analogy:** Adding a **sequence number** (or Lamport timestamp) to
> a message in a distributed queue. Without it, the consumer can't tell which
> message came first. The sinusoidal encoding is deterministic — you don't need
> to store it, just recompute it on the fly.

**Why sinusoidal?**
- It's deterministic (no learned parameters)
- It generalizes to sequence lengths not seen in training
- The model can learn to decode position from the signal

Modern LLMs use **learned positional embeddings** (just a lookup table) or
**rotary embeddings (RoPE)**, but sinusoidal is the canonical baseline.

---

## Full GPT-Style Architecture

Stack it all together:

```
Token IDs  [101, 2054, 3026, ...]
    │
    ▼
Token Embedding   (lookup table: vocab_size × d_model)
    │
    ▼
+ Positional Encoding   (same shape: seq_len × d_model)
    │
    ▼
┌─────────────────────────┐
│   Transformer Block 1   │
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│   Transformer Block 2   │
└─────────────────────────┘
    │
    ▼
    ...  (GPT-3 has 96 blocks)
    │
    ▼
┌─────────────────────────┐
│   Transformer Block N   │
└─────────────────────────┘
    │
    ▼
LayerNorm   (final normalization)
    │
    ▼
Linear (d_model → vocab_size)   (project to logit for each vocab token)
    │
    ▼
Softmax   (convert logits to probabilities)
    │
    ▼
Predicted next-token probabilities
```

> **Backend analogy:** The entire architecture is a deep **middleware pipeline**.
> Each transformer block is a middleware stage: normalize the request, pass through
> the attention routing layer (which references other requests in the context window),
> add original back, normalize, pass through the business logic layer (FFN), add back
> again. The final layers are just the response serializer (Linear + Softmax converts
> internal representation to a token probability distribution).

---

## Backend Analogy: Full Stack Summary

| Transformer Component | Backend Equivalent |
|-----------------------|--------------------|
| Token embedding       | Enum/ID → row lookup in a database |
| Positional encoding   | Lamport timestamp / sequence number appended to each record |
| LayerNorm             | Z-score normalization before scoring/ranking |
| Multi-Head Attention  | Parallel queries with different indexes, results merged |
| Residual connection   | Passthrough middleware (identity if sub-layer outputs zero) |
| FFN                   | Per-record transform / stored procedure (factual knowledge) |
| N stacked blocks      | N-stage middleware pipeline |
| Final linear + softmax | API serializer: internal state → probability distribution response |

---

## Key Numbers (for intuition)

| Model | d_model | d_ff | Heads | Layers | Params |
|-------|---------|------|-------|--------|--------|
| GPT-2 small | 768 | 3072 | 12 | 12 | 117M |
| GPT-2 large | 1280 | 5120 | 20 | 36 | 774M |
| GPT-3 | 12288 | 49152 | 96 | 96 | 175B |
| Llama 3 8B | 4096 | 14336 | 32 | 32 | 8B |

---

## Key Takeaways

1. **The transformer block is a middleware stack**: LayerNorm → MHA → Residual →
   LayerNorm → FFN → Residual. Memorize this order.
2. **Residual connections** are what make deep transformers trainable — gradients
   flow through the `+x` paths even if sub-layers are unstable.
3. **LayerNorm before** (Pre-Norm) is the modern convention; it stabilizes training.
4. **FFN = factual memory**: attention figures out *which* tokens to blend,
   FFN figures out *what to do* with each token given its context.
5. **Positional encoding** is how the model knows token order — without it,
   "cat sat on mat" and "mat on sat cat" would look identical.
6. **Stack N blocks**: the same block repeated N times, each refining the
   representations further. Deeper = more reasoning capacity.
