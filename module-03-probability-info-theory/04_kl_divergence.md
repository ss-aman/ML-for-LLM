# 04 — KL Divergence: Comparing Two Distributions

## The question KL answers

You have two probability distributions:
- `p` — the **true** distribution
- `q` — your **model/approximation** of it

How different are they? KL divergence answers this.

```
KL(p || q) = Σ p(x) · log(p(x) / q(x))
```

Equivalently:

```
KL(p || q) = Σ p(x) · (log p(x) - log q(x))
           = -H(p) + H(p, q)
           = H(p, q) - H(p)
```

So: **KL divergence = cross-entropy - entropy** = the extra bits wasted by using `q` instead of `p`.

---

## Properties

### Always non-negative

```
KL(p || q) ≥ 0
```

Equality holds if and only if `p = q` everywhere. This is **Gibbs' inequality** — you can never do better than using the true distribution.

### Not symmetric

```
KL(p || q) ≠ KL(q || p)   in general
```

This is why it's called a **divergence**, not a distance. It doesn't satisfy the symmetry axiom of a metric. The direction matters:

- `KL(p || q)` = "how much does q fail to describe p?"
- `KL(q || p)` = "how much does p fail to describe q?"

### What happens at zero probabilities

If `q(x) = 0` but `p(x) > 0`: KL is **infinite** — you assigned zero probability to something that actually happens. The model is infinitely wrong.

If `p(x) = 0` but `q(x) > 0`: the term is zero (same `0·log(0) = 0` convention). Assigning probability to impossible events isn't penalized by `KL(p||q)`.

This asymmetry is important and determines which direction of KL to use.

---

## The two directions and their behavior

### Forward KL: `KL(p || q)` — "mass-covering"

If `p(x) > 0` somewhere but `q(x) = 0` there, KL is infinite. So `q` **must** cover everywhere `p` has mass. This forces `q` to be **broad** — it would rather assign probability to unlikely regions than miss any region where `p` has probability.

Use case: variational inference, ensuring `q` doesn't miss modes of `p`.

### Reverse KL: `KL(q || p)` — "mode-seeking"

Now `p(x) = 0` doesn't penalize `q` placing mass there. But if `q(x) > 0` where `p(x) = 0`, that's allowed. The penalty only comes where `q` places mass. So `q` tends to find **one mode** of `p` and concentrate there.

Use case: RLHF and RL fine-tuning of LLMs.

---

## Backend analogy

You have two traffic distributions: `p` = actual traffic, `q` = your model/prediction.

```
KL(p || q) = extra cost per request due to modeling traffic wrong
```

If actual traffic goes to a server you modeled as "never receives traffic" (`q=0`):
- You didn't allocate resources there
- KL = ∞ — a total failure

If you allocated resources to a server that receives no actual traffic (`p=0`):
- Wasteful, but KL(p||q) doesn't penalize this
- KL(q||p) would penalize it

This asymmetry means: **which direction you minimize depends on what failures you want to avoid.**

---

## Connection to cross-entropy loss

The relationship `H(p, q) = H(p) + KL(p || q)` means:

```
Cross-entropy loss = Entropy of data + KL divergence
```

When training an LLM:
- `H(p)` = true entropy of language — fixed, can't be changed
- `KL(p || q)` = how far model `q` is from the true distribution `p`

**Minimizing cross-entropy loss = minimizing KL divergence from the true data distribution.** This is the fundamental objective of LLM training.

Since `H(p)` is constant, gradient of cross-entropy w.r.t. model parameters = gradient of KL divergence. They're equivalent objectives.

---

## KL divergence in RLHF

RLHF (Reinforcement Learning from Human Feedback) uses a KL penalty explicitly:

```
Objective = E[reward] - β · KL(π_RL || π_SFT)
```

Where:
- `π_RL` = current fine-tuned policy (model being trained)
- `π_SFT` = supervised fine-tuned baseline (starting point)
- `β` = penalty coefficient (typically 0.01–0.1)

**Why the KL penalty?** Without it, the model would learn to exploit the reward model's blind spots, producing text that scores high on the reward model but is garbage to humans (reward hacking). The KL term forces the model to stay "close" to the SFT baseline — it's the mathematical expression of "don't drift too far from what we already know works."

```python
# RLHF training step (simplified)
logprobs_rl  = rl_model.log_probs(response)
logprobs_sft = sft_model.log_probs(response)

kl_penalty = (logprobs_sft - logprobs_rl).mean()   # approx KL(RL || SFT)

reward = reward_model.score(prompt, response)
loss   = -(reward - beta * kl_penalty)              # maximize reward - KL
```

GPT-4, Claude, Gemini — all use variants of this formulation in their fine-tuning stage.

---

## Computing KL divergence

```python
import numpy as np

def kl_divergence(p, q):
    """
    KL(p || q) — how much q differs from p (p is the reference).
    p, q: probability distributions (arrays summing to 1)
    """
    p = np.array(p, dtype=float)
    q = np.array(q, dtype=float)

    # Only sum where p > 0 (0*log(0) = 0 by convention)
    mask = p > 0
    return np.sum(p[mask] * np.log(p[mask] / q[mask]))


# Example: uniform vs peaked distribution
p_uniform = np.array([0.25, 0.25, 0.25, 0.25])
q_peaked  = np.array([0.70, 0.10, 0.10, 0.10])

print(kl_divergence(p_uniform, q_peaked))  # p=true, q=model: model is wrong
print(kl_divergence(q_peaked, p_uniform))  # reverse direction — different value
```

---

## Numerical example: three distributions

Let the true distribution be `p = [0.5, 0.3, 0.2]` (3 tokens).

```
q1 = [0.5, 0.3, 0.2]   → KL = 0.0       (identical)
q2 = [0.4, 0.4, 0.2]   → KL = 0.027     (slightly wrong)
q3 = [0.9, 0.05, 0.05] → KL = 0.510     (very different)
q4 = [0.0, 0.5, 0.5]   → KL = infinite  (misses mass at index 0)
```

KL penalizes missing mass exponentially harder than spreading mass incorrectly.

---

## Jensen-Shannon (JS) divergence: the symmetric version

If you need a symmetric measure:

```
M = (p + q) / 2                   # mixture distribution
JS(p, q) = 0.5 · KL(p || M) + 0.5 · KL(q || M)
```

JS divergence is always finite (even when KL is ∞) and symmetric. Its square root is a true metric. Used in GANs (Generative Adversarial Networks) as the original training objective.

---

## Summary

| Property | Value |
|----------|-------|
| Formula | `KL(p\|\|q) = Σ p(x) · log(p(x)/q(x))` |
| Range | `[0, ∞)` |
| Zero when | `p = q` |
| Symmetric? | No — `KL(p\|\|q) ≠ KL(q\|\|p)` |

**KL in LLM training:**
- Pretraining: implicitly minimized via cross-entropy loss
- RLHF fine-tuning: explicit KL penalty keeps model near SFT baseline
- Evaluation: perplexity = `exp(cross-entropy) = exp(H(p) + KL(p||q))`

Next: **softmax** — the function that converts the model's raw numbers into the probability distributions that cross-entropy and KL act on.
