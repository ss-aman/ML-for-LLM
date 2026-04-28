# 02 — Entropy: Measuring Uncertainty

## The core question

Given a probability distribution, how "uncertain" or "surprising" is it?

- A coin that always lands heads: zero uncertainty — you already know the outcome
- A fair coin: maximum uncertainty for two outcomes — you have no idea which way it'll go
- A coin that lands heads 90% of the time: somewhere in between

**Entropy** is the mathematical answer to "how uncertain is this distribution?"

---

## Shannon entropy

**Entropy** `H(p)` of a distribution `p` is:

```
H(p) = -Σ p(x) · log₂(p(x))
```

Units: **bits** (using log base 2). Natural log gives **nats** — both are valid, just different scales.

The negative sign makes it positive: since `0 ≤ p(x) ≤ 1`, `log(p(x)) ≤ 0`, so `-log(p(x)) ≥ 0`.

**Intuition:** `-log₂(p(x))` is the "surprise" of seeing outcome x. Rare events (low `p`) are very surprising (high `-log p`). Common events are unsurprising. Entropy averages the surprise across all outcomes.

---

## Building intuition step by step

### Example 1: A coin always landing heads

```
p = [1.0, 0.0]   # heads always, tails never

H(p) = -(1.0 · log₂(1.0) + 0.0 · log₂(0.0))
     = -(1.0 · 0 + 0)       # log₂(1) = 0; 0·log(0) = 0 by convention
     = 0.0 bits
```

Zero entropy = zero uncertainty. You already know what will happen.

### Example 2: A fair coin

```
p = [0.5, 0.5]

H(p) = -(0.5 · log₂(0.5) + 0.5 · log₂(0.5))
     = -(0.5 · (-1) + 0.5 · (-1))
     = -(-0.5 - 0.5)
     = 1.0 bit
```

1 bit = you need exactly 1 bit of information to determine the outcome. Makes sense: 1 binary question perfectly resolves a fair coin.

### Example 3: A biased coin (90% heads)

```
p = [0.9, 0.1]

H(p) = -(0.9 · log₂(0.9) + 0.1 · log₂(0.1))
     = -(0.9 · (-0.152) + 0.1 · (-3.322))
     = -(−0.137 − 0.332)
     = 0.469 bits
```

Less than 1 bit — you're already pretty sure it'll be heads, so less information is needed to resolve the outcome.

### Example 4: A fair six-sided die

```
p = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]

H(p) = -(6 · (1/6) · log₂(1/6))
     = -log₂(1/6)
     = log₂(6)
     ≈ 2.585 bits
```

You need about 2.6 binary questions on average to identify the die roll.

---

## Key properties of entropy

### Maximum entropy = uniform distribution

For a distribution over `n` outcomes, entropy is maximized when all outcomes are equally likely:

```
H_max = log₂(n)
```

The uniform distribution is "maximally uncertain" — you know nothing about which outcome will occur.

### Minimum entropy = certain outcome

```
H_min = 0    (when one outcome has probability 1)
```

No uncertainty at all.

### Entropy only depends on probabilities, not labels

`H([0.5, 0.5])` is the same whether the outcomes are {heads/tails}, {cat/dog}, or {Paris/London}.

---

## Backend analogy: traffic entropy

Think of your server traffic as a distribution over servers:

```python
# Round-robin (uniform): maximum entropy
uniform_traffic = [0.25, 0.25, 0.25, 0.25]   # H ≈ 2.0 bits

# All traffic to server A: zero entropy
hotspot_traffic = [1.00, 0.00, 0.00, 0.00]   # H = 0.0 bits

# Slightly skewed load balancing
normal_traffic  = [0.40, 0.30, 0.20, 0.10]   # H ≈ 1.85 bits
```

High traffic entropy = well-distributed load. Low entropy = hotspot, poor distribution.

A log aggregation system compressing events also uses this: high-entropy log messages (fully random) can't be compressed; low-entropy messages (repetitive patterns) compress very well. This is the Shannon source coding theorem.

---

## The 0·log(0) convention

When `p(x) = 0`, the term `p(x) · log(p(x))` would be `0 · (-∞) = undefined`. We define:

```
0 · log(0) = 0
```

This is justified by the limit: `lim_{p→0} p·log(p) = 0`. Intuitively, an impossible event contributes nothing to uncertainty.

In code, always clip probabilities away from exactly 0:

```python
# Safe entropy calculation
import numpy as np

def entropy(p):
    p = np.array(p, dtype=float)
    p = p[p > 0]          # exclude zeros (they contribute 0 anyway)
    return -np.sum(p * np.log2(p))
```

---

## Entropy in the context of language

The **entropy of the English language** (next character given context) is roughly:

- Per character: ~1.0–1.5 bits (estimated by Shannon in 1951)
- Per word: ~10–12 bits for a large vocabulary

This is the **information-theoretic lower bound** on language model loss. No model can achieve lower cross-entropy than the true entropy of the language. Modern LLMs are approaching this bound.

**Perplexity** — the most common LM evaluation metric — is directly related:

```
Perplexity = 2^(cross-entropy loss in bits)
           = exp(cross-entropy loss in nats)
```

A perplexity of 10 means the model is "as confused as if uniformly choosing among 10 options" at each step. Modern LLMs achieve perplexity of ~3–15 on standard benchmarks, depending on the task.

---

## Joint entropy and conditional entropy

**Joint entropy** of two random variables X, Y:

```
H(X, Y) = -Σ_x Σ_y P(x,y) · log P(x,y)
```

**Conditional entropy** — uncertainty in Y after knowing X:

```
H(Y|X) = H(X,Y) - H(X)
```

**Mutual information** — how much knowing X reduces uncertainty about Y:

```
I(X;Y) = H(Y) - H(Y|X)
```

In attention mechanisms, the attention distribution has high mutual information with the query — it's selecting the most relevant keys, reducing uncertainty about what context matters.

---

## Relationship to compression

Shannon's source coding theorem: the minimum average number of bits needed to encode outcomes of a distribution `p` is exactly `H(p)`.

This means:
- A stream of fair coin flips needs 1 bit/flip minimum — optimal compression achieves exactly this
- A stream of "always heads" needs 0 bits — you don't even need to send it
- Log files with repetitive patterns (low entropy) compress better than random data (high entropy)

This is why gzip/zstd work: they exploit low entropy (redundancy) in data.

---

## Summary

| Distribution | Entropy | Meaning |
|-------------|---------|---------|
| Certain (one p=1) | 0 bits | No uncertainty |
| Fair coin | 1 bit | Need 1 question to resolve |
| Biased coin (90/10) | 0.47 bits | Already fairly sure |
| Uniform over n | log₂(n) bits | Maximum uncertainty |

Key formula: `H(p) = -Σ p(x) · log₂(p(x))`

Next: **cross-entropy** — what happens when you use the wrong distribution to describe the data.
