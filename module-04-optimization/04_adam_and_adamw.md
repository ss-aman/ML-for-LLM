# 04 — Adam and AdamW

## Adam: combining momentum and RMSprop

**Adam** (Adaptive Moment Estimation, Kingma & Ba, 2014) combines:
- **Momentum** — track the direction of gradients (first moment)
- **RMSprop** — adapt step size per parameter (second moment)

Plus one crucial addition: **bias correction** for cold-start at step 0.

---

## The full Adam algorithm

```python
# Hyperparameters (defaults that rarely need changing)
lr    = 0.001   # learning rate (often 1e-3 to 3e-4 for LLMs)
β1    = 0.9     # first moment decay (momentum)
β2    = 0.999   # second moment decay (adaptive rate)
ε     = 1e-8    # numerical stability (prevent division by zero)

# State — initialized once, updated every step
m = 0           # first moment  (momentum estimate)
v = 0           # second moment (gradient magnitude estimate)
t = 0           # step counter

# Each update step:
t  += 1
g   = gradient_of_loss(w)              # current gradient

m   = β1 * m + (1 - β1) * g           # update first moment (momentum)
v   = β2 * v + (1 - β2) * g**2        # update second moment (magnitude)

m_hat = m / (1 - β1**t)               # bias-corrected first moment
v_hat = v / (1 - β2**t)               # bias-corrected second moment

w -= lr * m_hat / (√v_hat + ε)        # weight update
```

Step by step:
1. `m` = exponential moving average of gradients (momentum, direction)
2. `v` = exponential moving average of squared gradients (magnitude)
3. `m_hat`, `v_hat` = bias-corrected versions
4. Update: move in the direction of `m_hat`, scaled by `1/√v_hat`

---

## Bias correction: why it matters

At step 1, `m` and `v` are initialized to 0. After one gradient update:

```
m_1 = β1 · 0 + (1 - β1) · g_1 = 0.1 · g_1     (with β1 = 0.9)
```

`m_1` is only 10% of the actual gradient — heavily biased toward zero because we started from 0.

Without correction: early steps are artificially tiny. The model barely moves in the first few iterations.

With correction:

```
m_hat_1 = m_1 / (1 - β1^1) = 0.1·g_1 / 0.1 = g_1   ← correct!
```

At step 1, the correction factor `1/(1-0.9^1) = 10x` exactly counteracts the cold-start bias.

By step ~1000: `β1^1000 ≈ 0`, so `1 - β1^1000 ≈ 1` — correction disappears automatically.

**Backend analogy:** Your monitoring system computes p99 latency with an exponential moving average. On startup, the EMA is 0. A correction factor of `1/(1 - decay^t)` gives you a valid estimate even before you have enough samples. After enough time, the correction factor approaches 1 and you have a real estimate.

---

## What Adam does per parameter

For each weight `w_i`, Adam maintains its own `m_i` and `v_i`. So if you have 7 billion parameters, Adam stores 14 billion additional numbers (doubles memory vs just storing weights).

The effective learning rate for parameter `w_i` at step t:

```
effective_lr_i = lr · m_hat_i / (√v_hat_i + ε)
               ≈ lr · (gradient direction) / (gradient magnitude)
```

This means:
- **Frequently updated params with large gradients**: `v_hat` is large → small effective step → prevents overstepping
- **Rarely updated params with tiny gradients**: `v_hat` is small → large effective step → accelerates learning
- **The direction** comes from `m_hat` (momentum-smoothed gradient)

---

## Why Adam is the default

1. **Works with default hyperparameters** — `β1=0.9, β2=0.999, ε=1e-8` work for the vast majority of problems. SGD requires careful per-problem LR tuning.

2. **Fast convergence** — the adaptive rates mean it efficiently handles differently-scaled parameters (common in large models)

3. **Handles sparse gradients** — important for embedding tables where most rows aren't updated each batch. Adam gives those rows a larger effective lr when they are updated.

4. **Stable early training** — bias correction + adaptive rates mean the first few steps are well-behaved

5. **Works across model types** — same optimizer for convnets, RNNs, transformers, RL

---

## The weight decay problem in Adam

**Weight decay** is a regularization technique that shrinks weights toward zero each step:

```
# Weight decay added to SGD (correct):
w = w - lr · g - lr · λ · w         # λ is weight decay coefficient
  = w · (1 - lr · λ) - lr · g       # weights are decayed each step
```

Intuition: weights that aren't pulled by gradients gradually shrink to zero. This prevents the model from relying on any single weight too heavily (regularization).

### The Adam weight decay bug

Naively adding weight decay to Adam:

```python
# Naive Adam + weight decay (WRONG):
g_modified = g + λ · w              # add decay to gradient
m = β1 * m + (1 - β1) * g_modified # update with modified gradient
v = β2 * v + (1 - β2) * g_modified**2
w -= lr * m_hat / (√v_hat + ε)      # scaled update
```

The problem: the weight decay term `λ·w` goes through the adaptive scaling `1/√v_hat`. So parameters with large `v_hat` (frequently updated) get less regularization than parameters with small `v_hat` (rarely updated). This is inconsistent and reduces regularization effectiveness.

**Ilya Loshchilov and Frank Hutter (2017) showed this is a significant bug** — the regularization doesn't work as intended in Adam.

---

## AdamW: the fix

**AdamW** decouples weight decay from the gradient update:

```python
# AdamW (correct):
# Step 1: apply gradient update (adaptive, as in Adam)
m = β1 * m + (1 - β1) * g          # gradient only, no weight decay
v = β2 * v + (1 - β2) * g**2
w -= lr * m_hat / (√v_hat + ε)      # adaptive gradient step

# Step 2: apply weight decay SEPARATELY (not through adaptive scaling)
w -= lr · λ · w                     # direct decay, not scaled by v_hat
```

Weight decay is now applied directly to weights at each step, with a fixed strength `λ`, regardless of the gradient history.

**Result:** All parameters get the same relative weight decay. The regularization works as intended.

---

## AdamW in code (complete implementation)

```python
import numpy as np

class AdamW:
    def __init__(self, lr=3e-4, beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.1):
        self.lr           = lr
        self.beta1        = beta1
        self.beta2        = beta2
        self.eps          = eps
        self.weight_decay = weight_decay
        self.m            = None
        self.v            = None
        self.t            = 0

    def step(self, params, grads):
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)

        self.t += 1
        b1, b2 = self.beta1, self.beta2

        # Update moments (gradient only — no weight decay here)
        self.m = b1 * self.m + (1 - b1) * grads
        self.v = b2 * self.v + (1 - b2) * grads ** 2

        # Bias correction
        m_hat = self.m / (1 - b1 ** self.t)
        v_hat = self.v / (1 - b2 ** self.t)

        # Gradient step (adaptive)
        params -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

        # Weight decay step (decoupled — NOT through adaptive scaling)
        params -= self.lr * self.weight_decay * params

        return params
```

---

## Adam vs AdamW: when does it matter?

| Aspect | Adam | AdamW |
|--------|------|-------|
| Weight decay | Applied through adaptive scaling (bug) | Decoupled (correct) |
| Regularization | Inconsistent across parameters | Consistent |
| Use in LLMs | Early models (GPT, BERT) | All modern LLMs |
| Effect | Can overfit more easily | Better generalization |

**GPT-3 paper:** Used Adam with β1=0.9, β2=0.95 (slightly different from default)  
**LLaMA/LLaMA 2:** Used AdamW with weight_decay=0.1  
**All modern LLMs:** AdamW is the default

The difference matters most for long training runs on large models. For small experiments, the difference is minimal.

---

## Adam hyperparameters for LLMs

| Hyperparameter | Default | LLM-specific value | Why |
|----------------|---------|-------------------|-----|
| `β1` (momentum) | 0.9 | 0.9 | Standard |
| `β2` (adaptive) | 0.999 | **0.95** | Faster adaptation to gradient changes |
| `ε` (stability) | 1e-8 | **1e-5 to 1e-8** | Larger ε for stability with bf16 precision |
| `weight_decay` | 0 | **0.1** | Regularization for generalization |

**Why β2 = 0.95 for LLMs instead of 0.999?**

With β2 = 0.999, the effective window for the second moment is `1/(1-0.999) = 1000` steps. This is very slow to react to changes.

With β2 = 0.95, the window is `1/(1-0.95) = 20` steps — much faster to react.

In long LLM training runs, the gradient distribution can shift significantly (different data, different training phase). β2 = 0.95 allows faster adaptation. This was empirically found to be better in GPT-3 training.

---

## Memory cost of Adam: a real engineering constraint

For a 7B parameter model:

| Item | Parameters | Memory (fp32) |
|------|-----------|---------------|
| Model weights | 7B | 28 GB |
| Adam first moment `m` | 7B | 28 GB |
| Adam second moment `v` | 7B | 28 GB |
| Gradients | 7B | 28 GB |
| **Total** | — | **112 GB** |

The optimizer state (m and v) doubles the memory requirement compared to just storing weights.

This is why:
1. **Mixed precision training** (bf16 weights + fp32 optimizer state) is standard — saves memory
2. **ZeRO optimizer** (DeepSpeed) shards optimizer state across GPUs — enables training models that don't fit on one GPU
3. **8-bit Adam** (bitsandbytes library) uses 8-bit quantization for optimizer states — 4x memory reduction

For an LLM practitioner: the optimizer is a major memory bottleneck, not just a training algorithm choice.

---

## Summary

| Optimizer | Formula | What it adds |
|-----------|---------|-------------|
| SGD | `w -= lr · g` | Baseline |
| Momentum | `v = β·v + g;  w -= lr·v` | Smoothing |
| Adam | `m, v updates + bias correction; w -= lr·m̂/√v̂` | Adaptive + smooth |
| AdamW | Adam + decoupled `w -= lr·λ·w` | Correct regularization |

**The LLM optimizer:** AdamW with β1=0.9, β2=0.95, ε=1e-8, weight_decay=0.1

Next: **learning rate schedules** — how to change lr over the course of training.
