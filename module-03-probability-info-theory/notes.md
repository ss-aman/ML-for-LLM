# Module 03: Probability and Information Theory for ML

> **Who this is for:** You've worked with logs, distributions, and monitoring dashboards. You've seen p99 latency, error rates, and histograms. This module formalizes that intuition into the mathematical tools LLMs are built on — especially *why* cross-entropy is the right loss function for language models.

---

## 1. Probability Distributions — Histograms of Outcomes

A **probability distribution** assigns a probability to every possible outcome. For a discrete distribution, probabilities sum to 1:

```python
# Load balancer: probability each server gets the next request
server_probs = {"server_A": 0.5, "server_B": 0.3, "server_C": 0.2}
# sum = 1.0 ✓
```

For a continuous distribution (like response time), we use a **probability density function (PDF)** — values don't represent probabilities directly, but area under the curve in a range gives the probability.

**Backend analogy:** A probability distribution is exactly like a histogram in your monitoring dashboard — it shows you how likely each outcome is. The normal distribution describes your API latency distribution; a uniform distribution describes a round-robin load balancer; a skewed distribution describes cache hit rates.

---

## 2. Expectation and Variance

The **expectation** (mean) `E[X]` is the probability-weighted average of all outcomes:

```
E[X] = Σ x * P(X = x)        (discrete)
E[X] = ∫ x * f(x) dx          (continuous)
```

**Variance** measures how spread out the distribution is:

```
Var[X] = E[(X - E[X])^2]
```

**Backend analogy:** Expectation = average latency. Variance = how much latency fluctuates. A service with `E[latency] = 50ms` but `Var[latency] = 10000ms²` (std dev = 100ms) has terrible tail latency — this is exactly what p99 monitoring catches.

---

## 3. The Gaussian (Normal) Distribution

The **Gaussian** or **normal** distribution is the bell curve:

```
f(x) = (1 / σ√2π) * exp(-(x-μ)²/2σ²)
```

Parameterized by mean `μ` and standard deviation `σ`.

**Why it appears everywhere:**
- Central Limit Theorem: the average of many independent random variables converges to a Gaussian, regardless of the original distribution
- It's the maximum entropy distribution given a fixed mean and variance (more on entropy below)
- Many ML weight initializations use Gaussian noise

**Backend analogy:** Your API latency under normal load follows roughly a Gaussian. Outliers (GC pauses, network blips) create a heavier tail than a true Gaussian — this is why p99 matters more than average latency.

---

## 4. Conditional Probability — Context-Dependent Probabilities

`P(A | B)` reads "probability of A *given* B" — the probability of event A, knowing that B has already occurred.

```
P(A | B) = P(A and B) / P(B)
```

**Backend analogy:** "Given this request came from the EU region (B), what is the probability it hits the GDPR-compliant database (A)?" Traffic routing logic is conditional probability in code. Feature flags are conditional probabilities with discrete conditions.

**In language models:** The entire job of an LLM is to compute conditional probabilities:

```
P("Paris" | "The capital of France is ___")
```

The model outputs a probability distribution over all possible next tokens, conditioned on the input context.

---

## 5. Bayes' Theorem — Updating Beliefs

```
P(A | B) = P(B | A) * P(A) / P(B)
```

Where:
- `P(A)` is the **prior** — your belief before seeing evidence
- `P(B | A)` is the **likelihood** — how probable is the evidence given the hypothesis
- `P(A | B)` is the **posterior** — your updated belief after seeing evidence

**Backend analogy:** Anomaly detection. Prior: "99% of requests are legitimate." Likelihood: "if the request is fraudulent, there's an 80% chance it comes from a new IP." After observing a new IP, you use Bayes' theorem to update the fraud probability.

---

## 6. Entropy — Measuring Uncertainty

**Entropy** `H(p)` measures the uncertainty (or average information content) of a distribution:

```
H(p) = -Σ p(x) * log₂(p(x))
```

Units: **bits** (when using log base 2). Intuitively: how many binary questions do you need to ask, on average, to determine the outcome?

**Key properties:**
- A **certain** outcome (one probability = 1, rest = 0) has **zero entropy** — no uncertainty
- A **uniform** distribution has **maximum entropy** — maximum uncertainty
- Entropy increases as the distribution becomes more spread out / unpredictable

**Backend analogy:** Think of entropy as the unpredictability of your traffic patterns.
- `entropy = 0`: every request goes to server A (deterministic, fully predictable, zero entropy)
- `entropy = max`: requests are uniformly distributed across all servers (maximum unpredictability)
- A load balancer that always routes based on user ID has low entropy (predictable); a round-robin has higher entropy

**Example calculations:**

```
Fair coin: p = [0.5, 0.5]
H = -(0.5 * log₂(0.5) + 0.5 * log₂(0.5))
  = -(0.5 * (-1) + 0.5 * (-1))
  = 1.0 bit   ← you need exactly 1 bit to encode a fair coin flip

Biased coin: p = [0.9, 0.1]
H = -(0.9 * log₂(0.9) + 0.1 * log₂(0.1))
  ≈ 0.469 bits   ← less uncertain, takes less than 1 bit on average

Always heads: p = [1.0, 0.0]
H = 0.0 bits    ← perfectly predictable, zero information needed
```

---

## 7. KL Divergence — How Different Are Two Distributions?

**Kullback-Leibler (KL) divergence** measures how much distribution `q` differs from distribution `p`:

```
KL(p || q) = Σ p(x) * log(p(x) / q(x))
```

Properties:
- `KL(p || q) ≥ 0` always
- `KL(p || q) = 0` if and only if `p = q`
- **NOT symmetric**: `KL(p || q) ≠ KL(q || p)` in general

**Backend analogy:** Compare two load balancing policies. Policy `p` (actual traffic distribution) vs policy `q` (your target/model of traffic). KL divergence measures how surprised you'd be if you assumed the traffic followed `q` but it actually follows `p`. If your traffic model is wrong, you've wasted capacity or caused hotspots — KL divergence quantifies that cost.

**In ML:** KL divergence is used in variational autoencoders (VAEs), reinforcement learning (KL penalty in RLHF), and when analyzing how closely a model's predicted distribution matches the true distribution.

---

## 8. Cross-Entropy — The Loss Function of Language Models

**Cross-entropy** between distributions `p` (true) and `q` (predicted) is:

```
H(p, q) = -Σ p(x) * log(q(x))
```

**The relationship:** Cross-entropy = Entropy + KL divergence:

```
H(p, q) = H(p) + KL(p || q)
```

Since `H(p)` is fixed (it's the true distribution you can't change), **minimizing cross-entropy is equivalent to minimizing KL divergence** — making the model's predicted distribution as close as possible to the true distribution.

### Why Cross-Entropy is Right for Language Models

When training on labeled data:
- `p` is a **one-hot distribution** (the true label): `p = [0, 0, 1, 0, ...]` (all probability on the correct token)
- `q` is the model's predicted probability distribution: `q = [0.05, 0.1, 0.7, 0.15, ...]`

Cross-entropy simplifies beautifully for one-hot `p`:

```
H(p, q) = -Σ p(x) * log(q(x))
         = -1 * log(q(correct_class))   ← all other p(x) = 0 cancel out
         = -log(probability_assigned_to_correct_token)
```

**Intuition:** Cross-entropy loss = negative log probability of the correct answer. The model is penalized by how surprised it was by the correct token.

**Examples:**

```
Correct class: token 0 (index 0)

Model A says: q = [0.9, 0.05, 0.05]  (very confident, correct)
Loss = -log(0.9) = 0.105   ← very low loss

Model B says: q = [0.1, 0.5, 0.4]  (wrong, confident it's class 1)
Loss = -log(0.1) = 2.303   ← high loss

Model C says: q = [0.33, 0.33, 0.34]  (totally uncertain)
Loss = -log(0.33) = 1.109  ← medium loss
```

**Backend analogy:** Cross-entropy is like measuring how surprised your routing algorithm is when traffic behaves in the "expected" (true) way. If your algorithm expects 90% of traffic to be API calls but assigns only 10% probability to that outcome, the cross-entropy penalty is high — you modeled it poorly.

### Cross-Entropy ≥ Entropy

Since `KL(p || q) ≥ 0`:
```
H(p, q) = H(p) + KL(p || q) ≥ H(p)
```

The cross-entropy loss can never be lower than the true entropy of the data. This is the **information-theoretic lower bound** on how well any model can do. For language, the true entropy of English text is roughly 1-1.5 bits per character — a perfect language model could get no better than that.

**Perplexity:** Language models are often evaluated with **perplexity** = 2^(cross-entropy loss in bits). A perplexity of 10 means the model is "as confused as if it had to choose uniformly among 10 options" at each step.

---

## 9. Softmax — Converting Raw Scores to Probabilities

Neural networks output raw scores (called **logits**) — arbitrary real numbers. The **softmax** function converts these to a valid probability distribution (non-negative, sums to 1):

```
softmax(z)_i = exp(z_i) / Σ_j exp(z_j)
```

**Why exponentiation?** It amplifies differences — high scores get disproportionately more probability mass. Combined with cross-entropy loss, it encourages the model to be confident about the right answer.

**Numerical stability:** `exp(z)` overflows for large `z`. Standard fix: subtract the maximum before exponentiating.

```python
z_stable = z - max(z)   # shift so max is 0 → exp(0) = 1, no overflow
softmax = exp(z_stable) / sum(exp(z_stable))
```

**Backend analogy:** Softmax is like converting raw CPU usage numbers for each server into "what percentage of traffic should go to each server" in a load balancer — with exponential weighting so the most capable servers get the most traffic.

---

## 10. Summary Table

| Concept | Definition | Backend Analogy |
|---|---|---|
| Probability distribution | P for each outcome, sums to 1 | Traffic distribution histogram |
| Expectation E[X] | Probability-weighted average | Average latency |
| Variance | Average squared deviation from mean | Latency variance (p99 tells this story) |
| Gaussian | Bell curve distribution | Normal latency distribution |
| P(A\|B) | Conditional probability | "Given EU request, P(GDPR DB)" |
| Bayes' theorem | Update P given new evidence | Bayesian anomaly detection |
| Entropy H(p) | -Σ p log p | Traffic unpredictability measure |
| KL divergence | How different are two distributions | Comparing two load balancing policies |
| Cross-entropy H(p,q) | -Σ p log q | LLM's surprise at seeing the right answer |
| Softmax | Converts raw scores to probabilities | Weighted load balancer allocation |

---

## 11. How This Appears in LLMs

- **Tokenization:** The vocabulary is a discrete distribution over ~50,000 tokens
- **Next-token prediction:** The model outputs a softmax distribution over all tokens
- **Training loss:** Cross-entropy between model's distribution and one-hot true label
- **Temperature:** Scales the logits before softmax — high temperature = more uniform (creative), low temperature = more peaked (conservative)
- **Top-k / nucleus sampling:** Truncate the distribution to only sample from the top-k most probable tokens
- **RLHF KL penalty:** During fine-tuning, a KL divergence penalty keeps the model from drifting too far from the pretrained distribution

---

## Further Reading

- [3Blue1Brown: Bayes' Theorem](https://www.youtube.com/watch?v=HZGCoVF3YvM) — best visual introduction
- [Visual Information Theory](https://colah.github.io/posts/2015-09-Visual-Information/) by Chris Olah — entropy and KL divergence explained visually
- [The Shannon entropy Wikipedia article](https://en.wikipedia.org/wiki/Entropy_(information_theory)) — mathematical foundation
