# 01 — Probability Basics

## What is probability?

A **probability** is a number between 0 and 1 that represents how likely an event is:
- `0` = impossible
- `1` = certain
- `0.5` = equally likely to happen or not

A **probability distribution** assigns a probability to every possible outcome. The probabilities must sum to exactly 1.

**Backend analogy:** A probability distribution is your monitoring dashboard's histogram — it shows how likely each outcome is. You already think in probability every day: "this request has a 2% chance of timing out."

---

## Discrete distributions

When outcomes are countable (categories, integers), we use a **probability mass function (PMF)**:

```python
# Example: which server handles the next request?
server_probs = {
    "server_A": 0.5,
    "server_B": 0.3,
    "server_C": 0.2,
}
# Must sum to 1: 0.5 + 0.3 + 0.2 = 1.0 ✓

# In LLMs: which token comes next?
vocab_probs = {
    "the":   0.15,
    "a":     0.10,
    "Paris": 0.70,
    "dog":   0.05,
}
# sum = 1.0 ✓
```

The LLM output at each step is a discrete distribution over ~50,000 tokens.

---

## Continuous distributions

When outcomes can be any real number (time, temperature, height), we use a **probability density function (PDF)**. The probability of falling in a range is the area under the curve.

The most important continuous distribution: the **Gaussian (normal)**:

```
f(x) = (1 / σ√2π) * exp(-(x - μ)² / 2σ²)
```

Parameters:
- `μ` (mu) = mean — the center of the bell curve
- `σ` (sigma) = standard deviation — how wide the bell is
- `σ²` = variance

**Backend analogy:** API latency under normal load is approximately Gaussian. `μ = 50ms` (average), `σ = 10ms` (typical spread). Outliers (GC pauses, cold starts) create a heavy tail — that's why p99 matters more than average.

---

## Why the Gaussian appears everywhere in ML

1. **Weight initialization:** Model weights are initialized from `N(0, σ²)` where `σ` is chosen carefully (Xavier/He initialization from Module 02 code)

2. **Central Limit Theorem:** The average of many independent random variables converges to a Gaussian regardless of their original distribution. Since neural network outputs are sums of many weighted inputs, they tend toward Gaussian.

3. **Maximum entropy:** Given a fixed mean and variance, the Gaussian is the distribution with the most uncertainty — a natural "neutral" prior.

---

## Expectation (mean)

The **expected value** `E[X]` is the probability-weighted average:

```
E[X] = Σ x · P(X = x)        # discrete
E[X] = ∫ x · f(x) dx          # continuous
```

Think of it as: if you ran the experiment millions of times and averaged the results, what number would you get?

```python
# Dice roll: E[X] = 1*(1/6) + 2*(1/6) + ... + 6*(1/6) = 3.5
outcomes = [1, 2, 3, 4, 5, 6]
probs    = [1/6] * 6
expected = sum(x * p for x, p in zip(outcomes, probs))
# = 3.5
```

---

## Variance and standard deviation

**Variance** `Var[X]` = average squared distance from the mean:

```
Var[X] = E[(X - E[X])²] = E[X²] - (E[X])²
```

**Standard deviation** `σ = √Var[X]` — same units as X, easier to interpret.

```python
# Service A: avg latency 50ms, std 5ms  → predictable, tight SLAs
# Service B: avg latency 50ms, std 30ms → same average, terrible tail behavior
# Variance reveals what avg hides
```

---

## Conditional probability

`P(A | B)` = "probability of A, **given** B has happened"

```
P(A | B) = P(A and B) / P(B)
```

**Backend analogy:** "Given this request came from the mobile app (B), what is the probability it hits the image processing service (A)?" Your routing rules are conditional probabilities in code.

**In LLMs — this is everything:**

```
P(next_token | all_previous_tokens)
```

The entire job of an LLM is to estimate this conditional probability. Given "The capital of France is", what is the probability that the next token is "Paris"? GPT-4 assigns that very high probability — it has learned this conditional distribution from trillions of tokens.

---

## Bayes' theorem

```
P(A | B) = P(B | A) · P(A) / P(B)
```

Where:
- `P(A)` = **prior** — your belief before seeing evidence B
- `P(B | A)` = **likelihood** — how likely is B if A is true?
- `P(A | B)` = **posterior** — updated belief after seeing B

**Backend analogy:** Fraud detection.
- Prior: 1% of requests are fraudulent → `P(fraud) = 0.01`
- Likelihood: 80% of fraud comes from new IPs → `P(new_IP | fraud) = 0.8`
- You observe a new IP. What's `P(fraud | new_IP)`?

```
P(fraud | new_IP) = P(new_IP | fraud) · P(fraud) / P(new_IP)
                  = 0.8 · 0.01 / P(new_IP)
```

If 5% of all requests come from new IPs: `P(new_IP) = 0.05`

```
P(fraud | new_IP) = 0.8 · 0.01 / 0.05 = 0.16 = 16%
```

Observing a new IP increased fraud probability from 1% to 16%.

**In ML:** Bayes' theorem underpins naive Bayes classifiers and Bayesian deep learning. For LLMs specifically, the pretraining process can be viewed as Bayesian inference over the distribution of text.

---

## Key properties of probability to remember

| Property | Rule | Example |
|----------|------|---------|
| Non-negativity | `P(A) ≥ 0` | No negative probabilities |
| Normalization | `Σ P(x) = 1` | All outcomes sum to 100% |
| Complement | `P(not A) = 1 - P(A)` | 5% timeout → 95% success |
| Independence | `P(A and B) = P(A)·P(B)` | Two coin flips |
| Conditional | `P(A\|B) = P(A,B)/P(B)` | Routing given region |

---

## Connection to LLMs

Every concept in this file maps directly to LLM operation:

| Concept | LLM usage |
|---------|-----------|
| Discrete distribution | Output probability over vocabulary |
| Gaussian | Weight initialization; some noise processes |
| Expectation | Expected token; perplexity calculation |
| Conditional probability | `P(token | context)` — the core task |
| Bayes | RLHF reward modeling; uncertainty estimation |

The next file (entropy) shows how to measure the quality of these distributions.

---

## Summary

- A probability distribution assigns numbers (summing to 1) to outcomes
- Discrete: LLM vocabulary predictions. Continuous: Gaussian for weights.
- Expectation = probability-weighted average
- Variance = how spread out the distribution is
- Conditional probability `P(A|B)` is the core of LLMs: predict next token given context
- Bayes' theorem: how to update beliefs when you see new evidence
