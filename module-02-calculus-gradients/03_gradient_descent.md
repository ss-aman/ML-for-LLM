# 03 — Gradient Descent: The Training Algorithm

## The Core Idea

You have a loss `L(w)` that measures how wrong your model is.
You have the gradient `∇L(w)` that points toward increasing loss.
You want to find `w` that minimizes `L`.

**Gradient descent** is the loop:

```
initialize w randomly

repeat until good enough:
    grad = ∇L(w)         # compute gradient
    w = w - lr * grad    # step against the gradient
```

That's it. This one algorithm trains every neural network and every LLM.

---

## Backend Analogy: Auto-Tuning Server Config

You're auto-tuning three config parameters: thread_pool_size, cache_ttl,
connection_timeout. Your objective is to minimize p99 latency.

You can't derive the formula analytically, but you can measure.

Gradient descent would:
1. Measure current latency (compute loss)
2. Try nudging each parameter slightly, measure what changes (compute gradient)
3. Move all three parameters in the direction that reduces latency
4. Repeat

After many iterations you've converged to a good configuration — not necessarily
the perfect global minimum, but good enough to deploy.

---

## The Update Rule in Detail

```
w ← w - α · ∇L(w)
```

- `w` is the current parameters (a vector)
- `α` (alpha) is the **learning rate** — how big each step is
- `∇L(w)` is the gradient — which direction is "uphill"
- `w - α · ∇L(w)` takes one step "downhill"

Step-by-step on paper:

```
w₀ = [0, 0]          (random start)
L(w₀) = 34.0

∇L(w₀) = [-6, -10]   (gradient at this point)
w₁ = [0, 0] - 0.1 * [-6, -10]
w₁ = [0 + 0.6,  0 + 1.0]
w₁ = [0.6, 1.0]

L(w₁) = 22.8         (lower — we're improving!)
```

Each step moves parameters toward the minimum.

---

## The Learning Rate: Critical Hyperparameter

The learning rate `α` controls the step size. It's the most important
hyperparameter to tune.

### Too small

```
α = 0.001
```
- Takes many tiny steps
- Converges, but slowly
- You spend too much compute before reaching a good solution
- Fine if you have infinite time; usually not acceptable in practice

### Too large

```
α = 2.0
```
- Takes huge steps — may jump over the minimum entirely
- Can oscillate back and forth, never settling
- Can diverge (loss gets larger and larger → NaN)
- In LLM training: loss spikes or goes to NaN early on

### Just right

```
α = 0.1  (typical starting point for many problems)
```
- Converges steadily in a reasonable number of steps
- Finding this is more art than science; grid search or schedules are common

**Learning rate visualization:**

```
Loss
 │       ↙ lr too large (oscillates/diverges)
 │    ↙ lr just right (smooth convergence)
 │ ↙ lr too small (converges but slowly)
 └─────────────────── iterations
```

---

## Types of Gradient Descent

### Full-Batch Gradient Descent

Compute gradient using the ENTIRE training dataset.

```
grad = (1/N) * sum of gradients over all N examples
```

- Most accurate gradient estimate
- Prohibitively slow for large datasets (GPT-3's dataset: 300 billion tokens)
- Never used for large-scale ML

### Stochastic Gradient Descent (SGD)

Compute gradient using just ONE random example.

```
grad = gradient on example i   (random i each step)
```

- Very fast per step (one example at a time)
- Noisy gradient — direction jumps around
- Paradoxically, the noise sometimes helps escape local minima
- The term "SGD" in ML usually means mini-batch SGD below

### Mini-Batch Gradient Descent (what's actually used)

Compute gradient on a random **batch** of B examples.

```
grad = (1/B) * sum of gradients over B examples
```

- `B` is the **batch size**: typically 32, 128, 512, or 2048
- For LLM training: batch sizes of thousands to millions of tokens
- Good balance: stable enough gradient, fast enough to run
- Fits on GPU memory (each batch must fit)

**Why batching helps:**
1. GPU parallelism: process all B examples simultaneously (matrix operations)
2. More gradient steps per unit of compute than full-batch
3. The noise from small batches acts as implicit regularization

---

## Learning Rate Schedules

In practice, you don't use a constant learning rate for the entire training.

### Warmup

Start with a very small learning rate, ramp up linearly for the first few
thousand steps.

**Why:** early in training, gradients are large and noisy. A large initial
learning rate causes chaotic updates. Warmup lets the model stabilize.

```
GPT-3 warmup: linear increase from 0 to 6e-5 over 375M tokens
```

### Cosine Decay

After warmup, decay the learning rate following a cosine curve.

```python
lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(π * step / total_steps))
```

The loss landscape becomes more sensitive as training progresses.
Smaller learning rates make finer adjustments near the end.

```
lr
 │  /──\
 │ /    \────────────────
 │/
 └─────────────────── training steps
   warmup   cosine decay
```

This exact schedule (with variations) is used by GPT-2, GPT-3, LLaMA, and
most modern LLMs.

---

## Convergence Criteria

How do you know when to stop training?

1. **Loss plateaus**: validation loss stops decreasing for many steps
2. **Gradient norm approaches zero**: `||∇L|| ≈ 0` at a minimum
3. **Compute budget**: real LLMs train until they run out of allocated GPU hours
4. **Target metric**: train until your benchmark scores are good enough

In practice for large LLMs, the answer is (3) — you train for a fixed compute
budget and the model you have at the end is what you ship.

---

## Why Gradient Descent Works in High Dimensions

In 2D, saddle points are common (like a mountain pass: minimum in one direction,
maximum in another). You might get stuck.

In very high dimensions (millions of parameters), saddle points still exist
but gradient descent tends to **escape** them quickly. The reason: in a
1M-dimensional space, a true local minimum requires ALL 1M directions to be
"upward." This is exponentially unlikely. Most critical points are saddle
points, and gradient descent naturally avoids them.

This is one of the deep mysteries of deep learning: gradient descent on a
loss surface with billions of parameters reliably finds good solutions,
even though the problem is formally non-convex.

---

## Key Takeaway

> Gradient descent is the engine of neural network training. It repeatedly
> takes a small step against the gradient, decreasing the loss by a tiny
> amount each time. Applied millions of times over billions of examples,
> this simple rule produces a model that can write, reason, and code.

---

## What's Next

`04_chain_rule.md`: to compute the gradient for a deep network, you need
the chain rule. This is the mathematical foundation of backpropagation.
