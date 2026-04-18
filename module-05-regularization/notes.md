# Module 05: Regularization

> **Goal:** Understand why models that train perfectly often fail in production, and the toolkit of techniques that fix this. If you've ever seen a service that performs great on your test dataset but falls over with real traffic, you already understand the core problem.

---

## 1. The Fundamental Problem: Overfitting

A model **overfits** when it memorizes the training data rather than learning the underlying pattern.

### An Example

You have 10 data points and you fit a degree-15 polynomial. The polynomial threads through every single point — training error = 0. But ask it to predict a new point? It goes haywire.

> **Backend analogy:** Imagine you build a rule engine for rate limiting based on your last 10 incidents. You create 15 hyper-specific rules: "if requester is ServiceA AND endpoint is /checkout AND hour is 14:32 AND day is Tuesday…". Your rules perfectly describe those 10 incidents. But they're useless for the next incident, which doesn't match any rule exactly. You over-fit your rules to historical incidents instead of learning the general pattern.

### Visualising Overfitting

```
Training data:       ●  ●  ●  ●  ●  ●  ●  ●  ●  ●
Overfit model:       ~~~^~~~v~~~^~~~v~~~^~~~v~~~^~~~   (wiggles through every point)
Good model:         ________________/‾‾‾‾‾‾‾‾‾‾‾‾    (smooth, generalizable curve)
```

**Signs of overfitting:**
- Training loss is very low
- Validation/test loss is much higher
- The gap between training and validation loss is large and growing

---

## 2. Underfitting

The opposite problem: the model is too simple to capture the real pattern.

- **Symptoms:** High training loss AND high validation loss — the model just doesn't work at all
- **Cause:** Not enough parameters, too much regularization, or training stopped too early

> **Backend analogy:** You use a single global rate limit for all endpoints regardless of traffic pattern. It's too coarse — you're either blocking legitimate traffic or letting abuse through on specific endpoints.

**The goal:** find the sweet spot between over- and under-fitting.

---

## 3. The Bias-Variance Tradeoff

Every model error has two sources:

| Term | Meaning | Analogy |
|---|---|---|
| **Bias** | Systematic error — model too simple to represent the truth | A GPS that's always 50m north of your real position |
| **Variance** | Sensitivity to training data noise — small data change, big model change | A GPS that's in a different spot every time you look |

```
Total Error ≈ Bias² + Variance + Irreducible Noise
```

- **High bias** (underfitting): model misses the real pattern
- **High variance** (overfitting): model fits noise in the data
- **Regularization reduces variance at the cost of a small increase in bias** — this is almost always a good trade

---

## 4. L2 Regularization (Weight Decay)

**Idea:** Add a penalty to the loss that discourages large weights.

```
L_total = L_original + λ * sum(w²)
```

The gradient becomes:
```
dL/dw = dL_original/dw + 2λw
```

This pulls every weight toward zero every step — like a constant "shrink toward zero" force.

**Effect:**
- Weights stay small and well-distributed
- No individual weight can dominate — the model can't rely on any single feature too heavily
- Equivalent to placing a Gaussian prior on weights (Bayesian interpretation)

> **Backend analogy:** L2 is like a gentle keep-alive timeout on idle connections. Every connection (weight) decays toward closed (zero) unless it's actively useful. Connections that carry real traffic (important features) stay open. Idle connections (noise-fitting weights) time out. The `λ` is the idle timeout threshold — larger λ = more aggressive timeouts.

**The `λ` hyperparameter (regularization strength):**

| λ value | Effect |
|---|---|
| 0 | No regularization — pure overfitting risk |
| Very small (1e-4) | Light regularization — good default |
| Large (1.0+) | Strong regularization — may underfit |

---

## 5. L1 Regularization (Lasso)

**Idea:** Use absolute value penalty instead of squared:

```
L_total = L_original + λ * sum(|w|)
```

Gradient:
```
dL/dw = dL_original/dw + λ * sign(w)
```

The key difference: the L1 gradient is **constant** regardless of weight magnitude. This creates a strong push toward exactly zero for small weights — many weights end up exactly 0.

**Effect:** **Sparsity** — L1 eliminates unimportant features entirely, L2 only shrinks them.

> **Backend analogy:** L1 is like an aggressive circuit breaker that fully opens (disconnects) weak connections rather than just throttling them. If a route is not carrying meaningful traffic, it gets disabled entirely — not just reduced. This is useful when you want automatic feature selection (which connections to completely remove).

### L1 vs L2 Side-by-Side

| Property | L1 | L2 |
|---|---|---|
| Penalty | sum(&#124;w&#124;) | sum(w²) |
| Effect on small weights | Drives them to exactly 0 | Shrinks toward 0, rarely exactly 0 |
| Produces | **Sparse** weight vectors | **Dense** small weight vectors |
| Use case | Feature selection, sparse models | General regularization |
| Gradient near 0 | Discontinuous (sign function) | Continuous (linear) |

---

## 6. Dropout

**Idea:** During each training step, randomly "turn off" a fraction of neurons (set their outputs to zero). During inference, use all neurons but scale outputs down.

```python
# Training: randomly zero out neurons with probability p
mask = np.random.binomial(1, 1-p, shape)   # 1=keep, 0=drop
output = activation * mask / (1-p)          # scale to keep expected value same

# Inference: no dropout, use full network
output = activation
```

**Why does this work?**
- Each training step uses a different random sub-network
- No single neuron can rely on any specific other neuron being present
- Forces the network to learn **redundant representations** — multiple pathways to the same answer
- Equivalent to training an ensemble of many different sub-networks

> **Backend analogy:** Dropout is **chaos engineering** for your neural network. Just like Netflix's Chaos Monkey randomly kills service instances to ensure your architecture doesn't have single points of failure, dropout randomly kills neurons to ensure the network doesn't have single points of reliance. The result is a more robust system that degrades gracefully.

**Typical dropout rates:**
- 0.1–0.2: light dropout (common for earlier layers)
- 0.5: classic value for fully-connected layers
- 0.0: no dropout (inference time, or output layers)

**Where to apply:** Usually after fully-connected layers, sometimes after convolutional blocks. Not typically on the final output layer.

---

## 7. Early Stopping

**Idea:** Track performance on a held-out **validation set** during training. Stop when validation loss stops improving — even if training loss is still going down.

```
Epoch  | Train Loss | Val Loss  | Action
-------|------------|-----------|------------------
  10   |   0.50     |   0.52    | keep going
  20   |   0.30     |   0.33    | keep going
  30   |   0.15     |   0.30    | keep going — val still improving
  40   |   0.08     |   0.35    | hmm, val is getting worse
  50   |   0.04     |   0.42    | STOP — save checkpoint from epoch 30
```

The model at epoch 30 is the best one, even though training loss kept falling.

> **Backend analogy:** Early stopping is like your deployment rollout strategy. You deploy a new service version and watch real-user latency (validation loss), not just your benchmark suite results (training loss). If real-user latency starts ticking up even as your local benchmark improves, you roll back to the last good version. You don't blindly trust the benchmark — you trust the metric that matters.

**Implementation details:**
- **Patience:** how many epochs to wait after the last improvement before stopping (e.g., patience=10)
- **Best checkpoint:** save model weights whenever validation loss improves; restore at end
- **Validation set size:** typically 10–20% of data, held out from training

---

## 8. Putting It Together: A Practical Regularization Checklist

When your model overfits (val loss >> train loss):

1. **First: get more data** — the most effective fix, if possible
2. **L2 weight decay** — add `λ=1e-4` as a default starting point
3. **Dropout** — add 0.1–0.5 after large fully-connected layers
4. **Early stopping** — always track val loss, save best checkpoint
5. **Reduce model size** — fewer parameters = less capacity to overfit
6. **Data augmentation** — artificially expand training set (images: flip, crop; text: paraphrase)

> **Backend analogy:** Think of this as your reliability checklist. More data = bigger traffic volume to stress-test. L2 = timeout policies. Dropout = chaos testing. Early stopping = staged rollouts with automatic rollback. Reduce model size = simplify your service topology. Data augmentation = synthetic load testing.

---

## Key Formulas Summary

| Technique | What Changes | Effect |
|---|---|---|
| L2 (weight decay) | Loss += λ·∑w² | All weights shrink toward 0 |
| L1 | Loss += λ·∑&#124;w&#124; | Small weights → exactly 0 (sparse) |
| Dropout | During training: zero neurons with probability p | Redundant representations |
| Early stopping | Stop when val loss stops improving | Avoid over-training |

---

## What's Next

Module 06 covers **embeddings** — how to represent discrete tokens (words, subwords) as dense vectors that can be processed by neural networks. Regularization helps your model generalize; embeddings determine what your model even *sees* as input.
