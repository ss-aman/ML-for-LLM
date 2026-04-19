# 03 — Cross-Entropy Loss: The Loss Function of Every LLM

## The setup

You have two probability distributions:
- `p` — the **true** distribution (what actually happens)
- `q` — your **predicted/modeled** distribution (what you think will happen)

**Cross-entropy** measures: how many bits do you need on average to encode outcomes from `p`, if you designed your encoding using `q`?

```
H(p, q) = -Σ p(x) · log q(x)
```

Compare to regular entropy:
```
H(p)    = -Σ p(x) · log p(x)    ← using the TRUE distribution to encode
H(p, q) = -Σ p(x) · log q(x)    ← using the WRONG distribution q to encode
```

If `q = p` (your model is perfect): `H(p, q) = H(p)` — no wasted bits.  
If `q ≠ p` (your model is wrong): `H(p, q) > H(p)` — you waste bits.

The wasted bits = KL divergence (next file). Minimizing cross-entropy = minimizing that waste.

---

## Why cross-entropy is perfect for LLM training

When you train on labeled data, the "true" distribution `p` is a **one-hot vector**: all probability on the correct token, zero everywhere else.

```
Vocabulary: ["the", "a", "Paris", "cat", "dog"]

Context: "The capital of France is"
True next token: "Paris" (index 2)

p = [0.0, 0.0, 1.0, 0.0, 0.0]   ← one-hot: all probability on "Paris"
```

The model produces a predicted distribution:

```
Model output logits: [-1.2,  0.5,  3.1, -0.3,  0.8]
After softmax:  q = [0.02, 0.10, 0.70, 0.05, 0.13]   ← model thinks "Paris" 70% likely
```

Cross-entropy with one-hot `p`:

```
H(p, q) = -Σ p(x) · log q(x)
         = -(0·log(0.02) + 0·log(0.10) + 1·log(0.70) + 0·log(0.05) + 0·log(0.13))
         = -1 · log(0.70)
         = -log(0.70)
         = 0.357 nats  (or 0.515 bits)
```

**Key insight:** With one-hot `p`, cross-entropy collapses to:

```
H(p, q) = -log(q[correct_token])
         = -log(probability_assigned_to_the_right_answer)
```

The loss is simply the negative log probability of the correct token. Maximizing the probability of the right answer = minimizing cross-entropy.

---

## The loss tells you how surprised the model was

```
-log(probability) when probability = 1.0  → loss = 0.0   (certain, correct)
-log(probability) when probability = 0.5  → loss = 0.69  (unsure)
-log(probability) when probability = 0.1  → loss = 2.30  (wrong guess)
-log(probability) when probability = 0.01 → loss = 4.60  (very wrong)
```

High loss = model was very surprised by the correct answer = model is wrong.

**Backend analogy:** Imagine logging the "expected cost" of routing a request to the right server. If your load balancer assigns 90% probability to the server that actually handles the request, cost is low (`-log(0.9) = 0.1`). If it assigns only 1% probability to the right server, cost is high (`-log(0.01) = 4.6`). Cross-entropy is the average routing cost over all requests.

---

## Numerical example: comparing three models

Context: "The sky is"  
Correct next token: "blue" (index 1)

```python
import numpy as np

# Three different model predictions
model_A = np.array([0.05, 0.90, 0.03, 0.02])   # very confident, correct
model_B = np.array([0.50, 0.10, 0.25, 0.15])   # confident, wrong
model_C = np.array([0.25, 0.25, 0.25, 0.25])   # totally uncertain

correct_idx = 1

for name, q in [("Model A", model_A), ("Model B", model_B), ("Model C", model_C)]:
    loss = -np.log(q[correct_idx])
    print(f"{name}: prob_correct={q[correct_idx]:.2f}, loss={loss:.3f}")

# Model A: prob_correct=0.90, loss=0.105  ← great
# Model B: prob_correct=0.10, loss=2.303  ← terrible (confident but wrong)
# Model C: prob_correct=0.25, loss=1.386  ← mediocre (uniform)
```

Model B is the worst despite being "confident" — confidence in the wrong direction is maximally penalized.

---

## Average cross-entropy over a sequence

LLMs are trained on sequences of tokens. The loss for a sequence is the average cross-entropy over all positions:

```
L = (1/T) · Σ_{t=1}^{T} -log P(token_t | token_1, ..., token_{t-1})
```

This is called **negative log-likelihood (NLL)** or **cross-entropy loss**.

In practice (PyTorch):

```python
import torch
import torch.nn.functional as F

# logits: model output before softmax, shape (T, vocab_size)
# labels: correct token indices, shape (T,)
loss = F.cross_entropy(logits, labels)

# Internally does:
# 1. softmax(logits) → probabilities
# 2. -log(prob[i, label[i]]) for each position i
# 3. average over all positions
```

One line does everything because `F.cross_entropy` is:
- numerically stable (uses log-sum-exp trick)
- fused (softmax + log + NLL in one pass = more efficient)

---

## Cross-entropy ≥ entropy: the information-theoretic bound

Since `KL(p || q) ≥ 0`:

```
H(p, q) = H(p) + KL(p || q) ≥ H(p)
```

Cross-entropy can **never go below** the true entropy of the data. This is the fundamental bound: even a perfect model can't get loss below the irreducible uncertainty in the data.

For language, the true entropy of text is roughly:
- ~1.3 bits/character for English (Shannon's estimate)
- ~10 bits/token for typical vocabularies

This means there's a floor below which no model can go — some sequences are genuinely ambiguous.

---

## Perplexity: cross-entropy in disguise

**Perplexity (PPL)** is the standard LLM benchmark metric:

```
PPL = exp(cross_entropy_loss)    # using natural log
PPL = 2^(cross_entropy_in_bits)  # using log base 2
```

Interpretation: PPL = N means the model is "as confused as uniformly guessing among N options."

```
PPL = 1.0    → perfect model (impossible for real language)
PPL = 10     → good model (as confused as uniform over 10 words)
PPL = 1000   → bad model
PPL = 50000  → random model (vocab size = ~50k, guessing uniformly)
```

Historical progress:
- GPT-1 (2018): PPL ≈ 18 on PTB dataset
- GPT-2 (2019): PPL ≈ 35 on WebText (harder dataset)
- GPT-3 (2020): PPL ≈ 20 on Penn Treebank
- Modern LLMs: PPL ≈ 3–8 on many benchmarks

Lower is better. The gap between current models and the theoretical minimum is small — we're approaching the information-theoretic limit.

---

## Gradient of cross-entropy + softmax (why they work together)

The gradient of cross-entropy loss with respect to the logits is beautifully simple:

```
L = -log(softmax(z)[y])   # y = correct class

∂L/∂z_i = softmax(z)_i - 1[i == y]
         = q_i - p_i
```

In words: the gradient is just `(predicted_probability - true_probability)`. For the wrong classes, it's `q_i - 0 = q_i` (push probability down). For the correct class, it's `q_y - 1` (push probability up toward 1).

This is one reason cross-entropy + softmax is the standard for classification: the gradients are clean and numerically well-behaved.

**In PyTorch:** `F.cross_entropy` computes this gradient automatically via autograd. The backward pass through `log(softmax(z)[y])` gives exactly `q - p`.

---

## Backend analogy: SLA violation cost

Imagine you have an SLA that says "the correct server must handle each request." Cross-entropy is the average cost of your routing decisions:

```
Each request: cost = -log(probability_you_assigned_to_the_correct_server)
Total cost   = average over all requests
```

- If you always route correctly with high probability → low cost
- If you route to wrong server confidently → high cost  
- Minimizing this cost = learning to route correctly = what LLM training does

---

## Summary

| Scenario | Cross-entropy formula | Reduces to |
|----------|----------------------|------------|
| General | `-Σ p(x) · log q(x)` | Full sum |
| One-hot labels | `-log q[correct_class]` | One term |
| Sequence | `(1/T) Σ -log P(token_t | context)` | Average per token |

The training objective of every modern LLM: **minimize average cross-entropy over all tokens in the training corpus.**

When you see "the model achieved a loss of 2.3", that means `-log(prob_of_correct_token) = 2.3` on average, or `prob = e^{-2.3} ≈ 0.10` — the model assigns about 10% probability to the correct next token on average.

Next: **KL divergence** — how to quantify the difference between any two distributions.
