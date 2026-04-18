# Module 08: Normalization & Residual Connections

## Why This Matters

Modern deep networks — including every transformer — rely on two stabilising tricks:
**normalization layers** and **residual (skip) connections**. Without them, training
a 12-layer or 96-layer network from scratch is practically impossible: gradients
vanish or explode, and learning stalls.

This module explains what these tricks are, why they work, and exactly which
variant (LayerNorm) transformers use and why.

---

## 1. Why Normalize? The Unstable-Scale Problem

As data flows through many layers, the activations can drift to wildly different
scales:

- Layer 1 outputs values in the range `[-1, 1]`
- Layer 5 outputs values in the range `[-500, 500]`
- Layer 10 outputs values in `[-0.00001, 0.00001]`

This matters because:
- Weights in later layers are tuned for the scale they receive. A sudden scale
  change is like a unit conversion bug in your data pipeline.
- Sigmoid/Tanh activations saturate at extreme values → gradients become ≈ 0
  → the network stops learning (vanishing gradient problem).
- The optimizer (e.g., Adam) expects gradients of similar magnitude across
  parameters; large scale differences cause it to misallocate its step sizes.

> **Backend analogy:** Imagine an API that returns response times — but sometimes
> in milliseconds and sometimes in hours, depending on the endpoint. Any downstream
> code that computes averages, thresholds, or comparisons will produce nonsense.
> Normalization is the preprocessing step that converts everything to a consistent
> unit before further processing.

---

## 2. Batch Normalization (BatchNorm)

Introduced by Ioffe & Szegedy (2015). For each feature dimension, normalise
**across all samples in the current batch**:

```
μ_j = mean of feature j across the batch
σ_j = std  of feature j across the batch

x̂_ij = (x_ij - μ_j) / (σ_j + ε)    ← normalised
y_ij  = γ_j * x̂_ij + β_j            ← rescale + shift with learned params
```

- `γ` (gamma) and `β` (beta) are **learned** — so the network can undo the
  normalisation if it turns out to be harmful for a particular layer.
- `ε` (epsilon) is a small constant to avoid division by zero.
- During **training**: use the current batch's mean/std.
- During **inference**: use **running statistics** (exponential moving averages
  accumulated during training), because there may be no batch at inference time.

> **Backend analogy:** BatchNorm is like z-score normalisation applied per column
> in a database — you subtract the column mean and divide by the column std.
> The "batch" is the rows you currently have in memory. The running stats are
> the pre-computed column statistics you'd save to a config file and load at
> query time.

**Limitation for LLMs:** BatchNorm depends on the batch. For language models:
- Sequences have variable lengths — the "batch" isn't a fixed grid of values.
- During autoregressive inference, you generate **one token at a time** — there
  is no batch to compute statistics over.

---

## 3. Layer Normalization (LayerNorm)

Introduced by Ba et al. (2016), and used in every transformer architecture.
For a single sample, normalise **across all features** (the feature dimension):

```
μ  = mean of all features for this sample
σ  = std  of all features for this sample

x̂_j = (x_j - μ) / (σ + ε)          ← normalised
y_j  = γ_j * x̂_j + β_j              ← rescale + shift with learned params
```

The key difference from BatchNorm: **each sample is normalised independently**.
There is no dependence on other samples in the batch.

> **Backend analogy:** LayerNorm is per-request normalisation — you normalise
> based only on the values in the current request, not on statistics from other
> requests. This makes it completely stateless and safe to use at any batch size,
> including batch size 1 (single-token generation).

### Why LayerNorm beats BatchNorm for LLMs

| Property | BatchNorm | LayerNorm |
|---|---|---|
| Normalises across | Batch (samples) | Features (within one sample) |
| Depends on batch size | Yes — breaks at batch=1 | No — works with batch=1 |
| Works with variable-length sequences | No | Yes |
| Needs running statistics at inference | Yes | No |
| Used in transformers | No | Yes |

> **Backend analogy:** BatchNorm is like a stateful session cache that requires
> seeing multiple requests to build its statistics. LayerNorm is a pure function —
> it only needs the current request's data. Pure functions are easier to reason
> about, test, and deploy. That's why transformers use LayerNorm.

---

## 4. Residual Connections (Skip Connections)

Introduced by He et al. (2015) for ResNets. The idea is simple:

```
output = F(x) + x
```

Instead of learning to transform `x` from scratch, the layer only needs to learn
the **residual** (the delta from the identity mapping). If the ideal output is
close to the input, the layer just needs to learn a small correction.

> **Backend analogy:** Think of a circuit breaker / fallback pattern. If a
> middleware layer `F(x)` fails or produces a near-zero output (e.g., because
> gradients vanished), the original signal `x` is added back directly — the
> "circuit" falls back to passing the request through unchanged. The network
> never completely loses the original information, no matter how deep it is.

### Why Residual Connections Help Training

**Problem:** In a 50-layer network without skip connections, the gradient must
multiply through 50 Jacobians to reach the first layer. If each factor is < 1,
the gradient shrinks exponentially → vanishing gradient → layer 1 barely learns.

**Solution:** The skip connection adds a +1 path to every Jacobian term:

```
d(output)/d(input) = d(F(x))/d(x) + 1
```

Even if `d(F(x))/d(x)` is near zero, the gradient is at least 1. Gradients
flow easily all the way back to layer 1.

### In Transformers

Every transformer sub-layer (attention + feed-forward) uses a residual connection:

```
x = x + Attention(LayerNorm(x))
x = x + FeedForward(LayerNorm(x))
```

This is the **Pre-LN** (pre-layer-norm) variant used in GPT-2, GPT-3, and most
modern LLMs. The original transformer used Post-LN, but Pre-LN trains more
stably.

---

## 5. Pre-LN vs Post-LN

### Post-LN (original "Attention is All You Need" transformer):
```
x = LayerNorm(x + Attention(x))
x = LayerNorm(x + FeedForward(x))
```

The LayerNorm sits *after* the residual addition. The gradient path through the
residual branch passes through a LayerNorm on the way back — this can destabilise
gradient flow at the start of training.

### Pre-LN (modern LLMs):
```
x = x + Attention(LayerNorm(x))
x = x + FeedForward(LayerNorm(x))
```

The LayerNorm sits *before* the sub-layer. The residual path `x + ...` leaves
one clean identity path with no LayerNorm in the way — gradients flow back
unobstructed through that path. Training is more stable, especially at the start.

> **Backend analogy:** Post-LN is like normalising *after* merging the fallback
> path — you might distort the fallback signal. Pre-LN is like normalising only
> the branch being computed, leaving the direct (identity) path clean.

---

## 6. Putting It All Together

A single transformer block uses both techniques:

```python
def transformer_block(x):
    # --- Attention sub-layer ---
    x = x + attention(layer_norm(x))    # Pre-LN + residual

    # --- Feed-forward sub-layer ---
    x = x + feed_forward(layer_norm(x)) # Pre-LN + residual

    return x
```

- **LayerNorm** keeps activations at a stable scale before each sub-layer.
- **Residual connection** ensures gradients can always flow to early layers.
- Together, they make it possible to train 96-layer GPT-3-scale networks.

---

## 7. How This Connects to LLMs

When you read about GPT, BERT, or LLaMA architectures, every transformer layer
is essentially:
1. LayerNorm the input
2. Run multi-head attention (residual connection wraps it)
3. LayerNorm the result
4. Run a 2-layer feed-forward network (residual connection wraps it)

The normalization and residual structure is not an optional detail — it is what
makes these models trainable at all.
