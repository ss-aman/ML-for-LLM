# 05 — Softmax & Sampling: From Numbers to Words

## The pipeline

Every token the LLM generates goes through this exact pipeline:

```
Input tokens
    ↓
Transformer layers (many matrix multiplications)
    ↓
Logits: one real number per vocabulary token   shape: (vocab_size,)
    ↓
Softmax → probabilities summing to 1
    ↓
Sampling strategy → pick one token
    ↓
Output token (repeat from next position)
```

This file covers the last two steps: softmax and sampling.

---

## Softmax: converting logits to probabilities

**Logits** are the raw output of the model — arbitrary real numbers, any magnitude, can be negative:

```
logits = [2.1, -0.5, 4.3, 0.8, -1.2, ...]   # (vocab_size,) numbers
```

**Softmax** converts these to a valid probability distribution:

```
softmax(z)_i = exp(z_i) / Σ_j exp(z_j)
```

Properties of the output:
- All values `> 0` (exp is always positive)
- All values sum to exactly `1`
- Higher logit → higher probability, but nonlinearly

```python
import numpy as np

logits = np.array([2.1, -0.5, 4.3, 0.8])
probs  = np.exp(logits) / np.sum(np.exp(logits))
# ≈ [0.108, 0.010, 0.808, 0.073]
# index 2 had the highest logit (4.3) → 80.8% probability
```

---

## Why exponentiation?

The exp function has two key effects:

1. **Makes everything positive** — probabilities can't be negative
2. **Amplifies differences** — a logit advantage of 1 gives a probability ratio of `e ≈ 2.7x`

```
logit advantage of 2 → probability ratio of e² ≈ 7.4x
logit advantage of 5 → probability ratio of e⁵ ≈ 148x
```

This amplification encourages the model to be decisive. Without it (linear normalization), small logit differences would lead to wishy-washy probabilities.

**Backend analogy:** Softmax is like converting raw server performance scores into traffic allocation weights — with exponential weighting so the highest-scored server gets disproportionately more traffic. A score difference of 2 means 7.4× more traffic, not just 2× more.

---

## Numerical stability: the log-sum-exp trick

For large logits, `exp(z_i)` overflows (becomes `inf`):

```python
np.exp(1000)  # → inf (overflow)
```

**Fix:** Subtract the maximum before exponentiating. This doesn't change the result mathematically:

```
softmax(z)_i = exp(z_i) / Σ exp(z_j)
             = exp(z_i - max(z)) / Σ exp(z_j - max(z))    ← identical by algebra
```

After subtracting max, the largest value becomes 0, and `exp(0) = 1` — no overflow.

```python
def softmax(z):
    z_shifted = z - np.max(z)          # subtract max (numerical stability)
    exp_z     = np.exp(z_shifted)
    return exp_z / np.sum(exp_z)
```

PyTorch's `F.softmax` and `F.cross_entropy` do this automatically.

---

## Temperature: controlling the distribution sharpness

**Temperature** `T` is a scalar that scales the logits before softmax:

```
softmax(z / T)
```

Effect on the output distribution:
- `T = 1.0` — standard softmax (default)
- `T < 1.0` — sharper distribution (more confident, less diverse)
- `T > 1.0` — flatter distribution (less confident, more diverse/creative)

```python
logits = np.array([3.0, 1.0, 0.5])

T = 0.5:  probs ≈ [0.951, 0.046, 0.003]   # very peaked at index 0
T = 1.0:  probs ≈ [0.783, 0.106, 0.066]   # standard
T = 2.0:  probs ≈ [0.560, 0.251, 0.189]   # flatter, more spread
T → ∞:    probs → [0.333, 0.333, 0.333]   # uniform — completely random
T → 0:    probs → [1.000, 0.000, 0.000]   # argmax — greedy
```

**Backend analogy:** Temperature is like a routing policy aggressiveness knob. Low temperature = always route to the best server (exploitation). High temperature = distribute randomly (exploration). This is literally the exploration-exploitation tradeoff from reinforcement learning.

In inference APIs (like OpenAI), `temperature=0` means greedy (deterministic), `temperature=1` means standard sampling, `temperature=2` means creative/wild.

---

## Sampling strategies

Once you have the probability distribution, how do you pick a token? Several strategies, each with different trade-offs:

### 1. Greedy decoding (argmax)

Always pick the highest-probability token:

```python
next_token = np.argmax(probs)
```

- **Pro:** Deterministic, reproducible
- **Con:** Can get stuck in repetitive loops; misses diverse possibilities
- **When:** Tasks with one correct answer (math, code completion)

### 2. Random sampling

Sample from the full distribution:

```python
next_token = np.random.choice(len(probs), p=probs)
```

- **Pro:** Diverse outputs, can generate novel text
- **Con:** Can sample very unlikely (bad) tokens
- **When:** Creative writing at high temperature

### 3. Top-k sampling

Restrict sampling to the `k` most likely tokens, renormalize, then sample:

```python
def top_k_sample(probs, k=50):
    top_k_indices = np.argsort(probs)[-k:]      # indices of k highest probs
    top_k_probs   = probs[top_k_indices]
    top_k_probs   = top_k_probs / top_k_probs.sum()   # renormalize
    chosen        = np.random.choice(top_k_indices, p=top_k_probs)
    return chosen
```

- **Pro:** Prevents sampling from the long tail of unlikely tokens
- **Con:** Fixed k doesn't adapt to distribution shape (sometimes tail is meaningful)
- **When:** General text generation, GPT-2 used k=50 by default

### 4. Nucleus (top-p) sampling

Instead of top-k tokens, take the smallest set of tokens whose cumulative probability exceeds `p`:

```python
def nucleus_sample(probs, p=0.9):
    sorted_indices = np.argsort(probs)[::-1]          # descending order
    sorted_probs   = probs[sorted_indices]
    cumulative     = np.cumsum(sorted_probs)
    cutoff         = np.searchsorted(cumulative, p)    # first index where cumsum >= p
    nucleus        = sorted_indices[:cutoff + 1]
    nucleus_probs  = probs[nucleus] / probs[nucleus].sum()
    return np.random.choice(nucleus, p=nucleus_probs)
```

- **Pro:** Adapts to the distribution — when the model is confident (peaked), nucleus is small; when uncertain (flat), nucleus is large
- **Con:** Slightly more complex
- **When:** The default for modern LLMs (used by GPT-4, Claude, Llama)

---

## Top-p vs top-k: why nucleus wins

Consider two scenarios:

**Scenario A:** Model is very confident (`probs ≈ [0.98, 0.01, 0.01, ...]`)
- Top-k (k=50): samples from 50 tokens, including many with ~0% probability
- Top-p (p=0.9): nucleus = just 1 token (0.98 > 0.9), samples almost deterministically

**Scenario B:** Model is uncertain (`probs ≈ [0.1, 0.09, 0.08, ...]`, spread across many tokens)
- Top-k (k=50): cuts off at position 50 — might miss meaningful options at positions 51-100
- Top-p (p=0.9): might include 100+ tokens, preserving all meaningful alternatives

Nucleus adapts automatically. Top-k uses a fixed window regardless of distribution shape.

---

## Combined: temperature + top-p (the standard API setup)

Most production LLM APIs use both together:

```python
def generate_token(logits, temperature=1.0, top_p=0.9):
    # Step 1: apply temperature
    scaled_logits = logits / temperature

    # Step 2: softmax
    probs = softmax(scaled_logits)

    # Step 3: nucleus sampling
    return nucleus_sample(probs, p=top_p)
```

OpenAI API parameters: `temperature`, `top_p` — both directly control this pipeline.

---

## Beam search: deterministic but multi-path

Used in translation and structured generation (not for open-ended chat):

Keep the top-`B` (beam width) most probable **sequences** at each step, not just the most probable token.

```
Step 1: Generate B candidates from position 1
Step 2: For each candidate, generate B new tokens → B² sequences
        Keep only top-B sequences by total log-probability
Step 3: Repeat until EOS

Final: return the single highest-probability complete sequence
```

- **Pro:** Higher-quality structured outputs (code, translation)
- **Con:** Expensive (B × more compute), less diverse
- **When:** Machine translation, summarization — not open-ended chat

---

## The log-probability and why models work in log space

Multiplying many probabilities underflows to zero:

```python
# Probability of a 100-token sequence
0.3 ** 100  → essentially 0.0   (underflow)
```

Solution: work in log space. Log converts multiplication to addition:

```
log P(sequence) = Σ log P(token_t | context_t)
```

This is why:
1. Cross-entropy loss is in log space (`-log q[y]`)
2. Beam search computes log-probabilities and adds them
3. Perplexity uses `exp(average_log_loss)`

---

## Summary

```
Logits (raw model output)
    ↓  ÷ temperature
Scaled logits
    ↓  softmax
Probabilities [0,1], sum=1
    ↓  sampling strategy
Next token

Strategies:
  Greedy   → argmax, deterministic
  Random   → sample from full distribution
  Top-k    → restrict to k most likely, resample
  Top-p    → restrict to smallest set covering p% probability, resample
```

Temperature `T`:
- `T < 1` = more confident, less creative
- `T = 1` = standard
- `T > 1` = more diverse, more creative

Next: **the full picture** — every concept from this module mapped to real LLM code.
