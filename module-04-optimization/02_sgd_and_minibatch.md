# 02 — SGD and Mini-Batch Training

## The fundamental algorithm

Gradient descent updates weights by walking in the direction that reduces loss:

```
w ← w - lr · ∇L(w)
```

- `w` = all model weights (a very large vector)
- `lr` = learning rate (how big a step to take)
- `∇L(w)` = gradient of loss with respect to all weights

The update rule is the same regardless of the optimizer variant. What differs is:
1. Which data is used to compute `∇L(w)`  ← this file
2. How the gradient is used to update `w`  ← next two files

---

## Three variants: how much data per gradient step?

### Batch gradient descent (full batch)

Use the **entire training dataset** to compute one gradient:

```python
# Pseudocode
for step in range(n_steps):
    loss = compute_loss(model, all_training_data)   # forward pass over ALL data
    grad = compute_gradient(loss)                    # one gradient
    w    = w - lr * grad                             # one update
```

**Problem:** If you have 300 billion training tokens (GPT-3 scale), you can't process all of them before taking a single step. You'd wait months between updates.

**When used:** Very small datasets, convex problems (linear regression), when you need exact gradients.

### Stochastic gradient descent (true SGD)

Use **one example** per gradient step:

```python
for step in range(n_steps):
    x, y = random_sample(training_data)    # 1 example
    loss = compute_loss(model, x, y)
    grad = compute_gradient(loss)          # gradient on 1 example
    w    = w - lr * grad
```

**Problem:** One example is an extremely noisy estimate of the true gradient. Steps are highly variable — you might move in a completely wrong direction.

**Benefit:** Very fast updates. The noise can help escape saddle points.

### Mini-batch SGD (what everyone uses)

Use a **small random subset (mini-batch)** per step:

```python
for step in range(n_steps):
    batch = random_sample(training_data, size=batch_size)   # 32-4096 examples
    loss  = compute_loss(model, batch)
    grad  = compute_gradient(loss)    # gradient averaged over batch
    w     = w - lr * grad
```

**This is what all modern LLMs use.** It's the best of both worlds:
- Fast updates (don't need the full dataset)
- Less noisy than single-example (averaged over a batch)
- Compatible with GPU parallelism (batch processing is very efficient on GPUs)

---

## Why noise is actually helpful

Counterintuitively, the stochasticity in SGD isn't just a necessary evil — it actively helps training.

### Escaping saddle points

In a saddle point, the true gradient is exactly zero. With full-batch GD, you'd be stuck forever. Mini-batch noise means different batches give slightly different gradients, some of which point away from the saddle.

```
True gradient at saddle: [0, 0, 0, 0]
Batch gradient sample 1: [+0.1, -0.03, +0.02, -0.05]  → escapes!
Batch gradient sample 2: [-0.02, +0.08, -0.01, +0.03] → also escapes!
```

### Finding flatter (better) minima

Sharp minima have high curvature — the loss changes a lot for small weight changes. Flat minima are robust.

Large gradient noise (small batches) prevents settling in sharp minima because the noisy updates keep "bouncing out" of them. Only flat, wide minima are stable attractors under high noise.

Research shows: **smaller batch sizes → flatter minima → better generalization** (but slower training per step).

---

## Batch size in LLM training

LLM training uses **very large effective batch sizes** — often millions of tokens per step.

But there's a trick: **gradient accumulation**. You accumulate gradients over many small batches before doing one weight update. Mathematically equivalent to a large batch, but fits on the available GPU memory.

```python
# Gradient accumulation: effective_batch = batch_size * accumulation_steps
optimizer.zero_grad()
for i in range(accumulation_steps):
    batch  = get_next_batch()               # small batch that fits in memory
    loss   = model(batch) / accumulation_steps  # scale loss
    loss.backward()                         # accumulate gradients (don't update yet)

optimizer.step()   # one update after accumulating gradients
```

**Real LLM batch sizes:**
- GPT-3: 3.2M tokens per step (gradually increased during training)
- LLaMA 2: 4M tokens per step
- GPT-4 (estimated): 10M+ tokens per step

The batch size is carefully chosen — too small → noisy, poor hardware utilization. Too large → fewer update steps, can hurt generalization.

---

## The batch size – learning rate relationship

When you increase batch size, you should also increase learning rate proportionally:

```
Linear scaling rule:
  If batch_size doubles → lr doubles

Example:
  batch=256, lr=0.001  →  batch=512, lr=0.002
```

**Intuition:** A larger batch gives a more accurate gradient estimate. With a more accurate gradient, you can afford to take a bigger step. The noise level is lower, so a larger learning rate is stable.

**Formal:** For SGD, variance of mini-batch gradient scales as 1/batch_size. To maintain the same effective noise level as batch doubles, lr can double.

GPT-3 used this: batch size was progressively increased during training (from 32k tokens to 3.2M tokens), with learning rate adjusted accordingly.

---

## Epochs vs. steps

**Epoch:** One complete pass through the training data.  
**Step:** One gradient update (one mini-batch processed).

```
1 epoch = (dataset_size / batch_size) steps

Example:
  dataset: 10,000 examples
  batch:   32
  1 epoch = 10,000 / 32 ≈ 312 steps
```

**LLMs typically train for less than 1 epoch on their data.** GPT-3 trained on 300B tokens for roughly 1 epoch (used each training token approximately once). Why?

- Internet text is vast — you can always get more data
- Seeing each training example once avoids memorization
- Additional passes over the same data give diminishing returns but increase overfitting risk

---

## Learning rate: the most important hyperparameter

The learning rate `lr` controls step size. It's the single most impactful hyperparameter.

```
w ← w - lr · ∇L(w)
```

| lr | Effect |
|----|--------|
| Too large | Steps overshoot the minimum; loss oscillates or diverges |
| Too small | Converges correctly but takes forever; may get stuck |
| Just right | Smooth, efficient convergence |

**How to find it:**
1. Start with a small lr (e.g., 1e-4)
2. Do a short "lr warmup" sweep: increase lr linearly from 0 to a target
3. Watch where loss stops decreasing smoothly — that's too high
4. Use ~30% of that value as your base lr

**Typical starting values:**
- Adam/AdamW: `lr = 1e-4` to `3e-4`
- SGD: `lr = 0.01` to `0.1`

---

## Backend analogy: distributed metrics sampling

Full-batch GD is like reading every single log line from your entire cluster before making one config change. Accurate, but you'd never finish.

Mini-batch SGD is like sampling a random 1000 requests per second, computing your p99 from that sample, and adjusting your load balancer. The estimate is slightly noisy, but:
- You get feedback in milliseconds, not hours
- The noise in the sample can actually reveal problems that a clean aggregate would hide
- Multiple independent samples over time converge to the truth

The batch size is your sampling window:
- Bigger window (larger batch) = more accurate signal, but slower feedback loop
- Smaller window (smaller batch) = noisier signal, but faster adaptation

---

## Summary

| Variant | Data per step | Pros | Cons |
|---------|--------------|------|------|
| Full-batch GD | All data | Exact gradient | Impractical at scale |
| True SGD | 1 example | Fast updates | Very noisy |
| Mini-batch SGD | 32–4096 examples | Best of both | Standard choice |

**Key facts for LLMs:**
- Batch size: millions of tokens (via gradient accumulation)
- Less than 1 epoch over the training data
- Linear scaling rule: larger batch → larger lr
- Noise from SGD helps find flat (generalizable) minima

Next: **momentum and RMSprop** — the first techniques for making gradient steps smarter.
