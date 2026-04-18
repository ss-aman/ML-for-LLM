# Module 04: Optimization Algorithms

> **Goal:** Understand how neural networks "learn" — which is really just repeatedly adjusting parameters to reduce a loss function. If you've ever tuned a retry backoff or load balancer weight, you already have the right intuition.

---

## 1. The Loss Landscape

Before we can optimize, we need something to optimize *against*. That something is the **loss function** — a single number that tells us how wrong our model is.

- Loss = 0: perfect predictions
- Loss = large: terrible predictions

### The Terrain Metaphor

Think of all possible parameter settings as a landscape (a high-dimensional terrain map). Each point on the terrain is a specific configuration of weights. The **height** of the terrain at any point is the loss value there.

**Training = finding the lowest valley in this terrain.**

> **Backend analogy:** Imagine you have a million config knobs for your service. Each combination of knob settings produces a certain p99 latency (your "loss"). Training is an automated search through all those configs to find the combination with the lowest latency. The terrain is the "config space → latency" mapping.

Key properties of loss landscapes:
- **Global minimum:** The absolute lowest point — ideal, but often unreachable
- **Local minima:** Valleys that aren't the deepest — you can get stuck here
- **Saddle points:** Flat regions where gradients vanish — common in high dimensions
- **Convex vs. non-convex:** Simple problems (linear regression) have one bowl-shaped valley; neural networks have complex, non-convex landscapes

---

## 2. Gradient Descent

The gradient is the direction of **steepest ascent** in the loss landscape. To minimize loss, we walk in the **opposite direction** — steepest descent.

### The Algorithm

```
θ_new = θ_old - lr * ∇L(θ)
```

Where:
- `θ` = parameters (weights)
- `lr` = learning rate (step size)
- `∇L(θ)` = gradient of loss with respect to parameters

### One Step of Gradient Descent

1. Compute loss on the entire training dataset
2. Compute gradient of loss w.r.t. every parameter
3. Subtract gradient * learning rate from each parameter
4. Repeat until convergence

> **Backend analogy:** Imagine you're doing manual load balancing. You measure total p99 latency across your cluster (loss). You check which servers are contributing most to latency (gradient). You reduce traffic to those servers (subtract gradient). You repeat until latency stops improving.

---

## 3. Learning Rate: The Step Size Problem

The learning rate (`lr`) controls how big a step you take each iteration. This is the single most important hyperparameter in training.

| Learning Rate | Effect |
|---|---|
| Too large (e.g., 10.0) | Overshoots the minimum, loss bounces around or diverges |
| Too small (e.g., 0.000001) | Converges correctly but takes forever |
| Just right | Steady convergence to a good minimum |

> **Backend analogy:** This is exactly like tuning an exponential backoff. If your initial retry wait is too long (lr too small), you're wasting time. If it's too short (lr too large), you're hammering the service and causing cascading failures. The sweet spot depends on the specific system — and so does the right learning rate.

**Typical starting values:** 0.001 to 0.01 for Adam, 0.01 to 0.1 for SGD.

---

## 4. Stochastic Gradient Descent (SGD)

**Problem with vanilla gradient descent:** Computing the gradient on the *entire* dataset is expensive. If you have 1 million training examples, every single update requires a forward pass over all 1M examples.

**SGD solution:** Instead of using all data, randomly sample a **mini-batch** (e.g., 32 or 256 examples) and compute the gradient on just that mini-batch.

```
for each mini-batch B sampled from training data:
    θ = θ - lr * ∇L_B(θ)
```

The mini-batch gradient is a *noisy estimate* of the true gradient — but it's fast, and the noise actually helps escape local minima.

| Variant | Batch Size | Notes |
|---|---|---|
| Batch GD | All data | Exact gradient, slow |
| Stochastic GD | 1 example | Very noisy, fast per step |
| Mini-batch SGD | 32–512 | Best of both worlds — standard practice |

> **Backend analogy:** You can't read every log line to measure service health (too slow). Instead, you sample a random window of recent logs and compute metrics on that sample. The estimate is noisy but fast — and if you keep sampling, the noise averages out over time.

**Why noise helps:** A noisy gradient can "jiggle" you out of shallow local minima. Pure gradient descent might get stuck; SGD's randomness provides a natural escape mechanism.

---

## 5. Momentum

Plain SGD can oscillate — imagine a ball bouncing back and forth in a narrow valley instead of rolling straight to the bottom. **Momentum** fixes this by accumulating velocity in consistent directions.

### The Algorithm

```
v_t = β * v_{t-1} + (1 - β) * ∇L(θ)    # velocity update
θ   = θ - lr * v_t                        # parameter update
```

- `v` = velocity (exponential moving average of past gradients)
- `β` = momentum coefficient (typically 0.9)

The velocity builds up in directions where gradients consistently point the same way, and cancels out in directions where they oscillate.

> **Backend analogy:** TCP slow start. TCP doesn't immediately send at full speed — it builds up velocity as acknowledgments come in (confidence that the path is clear). Similarly, momentum builds velocity in directions that have consistently been "right." If the gradient keeps pointing the same way, the optimizer accelerates in that direction.

**Effect:** Faster convergence through consistent gradient directions, less oscillation in noisy directions.

---

## 6. Adaptive Learning Rates: AdaGrad and RMSProp

Different parameters may need different learning rates. A parameter updated rarely (sparse features) should get a larger effective LR; one updated constantly should get a smaller one.

### AdaGrad (Adaptive Gradient)

```
G_t = G_{t-1} + (∇L)^2            # accumulate squared gradients
θ   = θ - (lr / sqrt(G_t + ε)) * ∇L
```

- Parameters with large historical gradients get a smaller effective LR
- Parameters with small historical gradients get a larger effective LR

**Problem:** `G_t` only grows — learning rate shrinks to zero and training stalls.

### RMSProp (Root Mean Square Propagation)

Fixes AdaGrad's "shrinking LR" problem by using an **exponential moving average** of squared gradients instead of cumulative sum:

```
s_t = β * s_{t-1} + (1 - β) * (∇L)^2    # EMA of squared gradients
θ   = θ - (lr / sqrt(s_t + ε)) * ∇L
```

- `β` ≈ 0.999
- Older gradients are "forgotten" — the LR can recover if gradients change

> **Backend analogy:** Both are like per-route rate limiting that adapts to each endpoint's traffic patterns. AdaGrad is like a bucket that never empties — eventually all routes get throttled to zero. RMSProp is like a leaky bucket — old traffic patterns fade, so a quiet endpoint can burst again.

---

## 7. Adam: The Default "Just Works" Optimizer

**Adam (Adaptive Moment Estimation)** combines momentum (first moment) with RMSProp-style adaptive rates (second moment). It's the de facto standard optimizer for most ML work.

### The Full Algorithm

```python
# Hyperparameters (defaults that almost never need changing)
lr    = 0.001
beta1 = 0.9    # momentum decay
beta2 = 0.999  # RMSProp decay
eps   = 1e-8   # numerical stability

# State (per parameter)
m = 0  # first moment (momentum)
v = 0  # second moment (adaptive rate)
t = 0  # timestep

# Each update step:
t   += 1
m    = beta1 * m + (1 - beta1) * grad          # momentum
v    = beta2 * v + (1 - beta2) * grad**2       # adaptive rate
m_hat = m / (1 - beta1**t)                     # bias correction
v_hat = v / (1 - beta2**t)                     # bias correction
theta = theta - lr * m_hat / (sqrt(v_hat) + eps)
```

### Bias Correction: Why It Matters

At step 1, `m` and `v` are initialized to 0. Without correction, early updates are biased toward zero (the initialization, not the true mean). Dividing by `(1 - beta^t)` cancels this cold-start bias — by step ~1000, `1 - beta1^t ≈ 1` and the correction disappears.

> **Backend analogy:** This is exactly like a p99 latency metric that uses an exponential moving average. On service startup, you have 0 data points — your "p99" is incorrectly zero. A warm-up period (bias correction) adjusts early readings until you have enough samples for the average to be trustworthy.

### Why Adam "Just Works"

1. **Momentum** smooths out noisy gradients and accelerates through flat regions
2. **Adaptive rates** automatically tune per-parameter learning rates — parameters that need small updates get small updates
3. **Bias correction** means early training steps are reliable, not artificially zero
4. **Default hyperparameters** (lr=0.001, β1=0.9, β2=0.999) work for the vast majority of models

> **Backend analogy:** Adam is like using nginx with default settings. You *could* hand-tune every TCP keepalive, buffer size, and worker count — but the defaults are the result of years of operational wisdom and work well out of the box. You only tune them when you have a very specific bottleneck.

### When to Use What

| Optimizer | Use When |
|---|---|
| SGD + Momentum | You have time to tune LR carefully; often best final accuracy |
| Adam | Default choice; fast convergence; less tuning needed |
| RMSProp | RNNs and reinforcement learning (historically) |
| AdaGrad | Sparse data (NLP bag-of-words, recommendation systems) |

---

## 8. Learning Rate Schedules

The learning rate doesn't have to be constant. Common schedules:

- **Step decay:** Reduce LR by a factor every N epochs (e.g., multiply by 0.1 every 30 epochs)
- **Cosine annealing:** LR follows a cosine curve from `lr_max` down to near 0
- **Warmup:** Start with tiny LR, ramp up, then decay — prevents instability at the start of training

> **Backend analogy:** Step decay is like your SLA ratchet: you start with loose latency targets and tighten them as the system matures. Warmup is like gradually ramping traffic to a new deployment rather than immediately sending 100% to an untested service.

---

## Key Formulas Summary

| Algorithm | Update Rule |
|---|---|
| Gradient Descent | `θ = θ - lr * ∇L` |
| SGD | Same, but `∇L` computed on mini-batch |
| SGD + Momentum | `v = β*v + ∇L; θ = θ - lr*v` |
| RMSProp | `s = β*s + (1-β)*∇L²; θ = θ - lr*∇L/√(s+ε)` |
| Adam | `m = β1*m + (1-β1)*∇L; v = β2*v + (1-β2)*∇L²; θ = θ - lr*m̂/√(v̂+ε)` |

---

## What's Next

Module 05 covers **regularization** — techniques to prevent your model from memorizing the training data instead of learning generalizable patterns. Optimization gets you to the bottom of the loss landscape; regularization ensures that bottom is somewhere useful.
