# 03 — Momentum and RMSprop

## The problem SGD doesn't solve

Plain SGD works, but it has two distinct failure modes:

**Problem 1: Oscillation in narrow valleys**

Imagine a loss landscape shaped like an elongated bowl (common in neural networks):
```
          │
  wide ←──┼──→ width: low curvature, shallow gradient
          │
  narrow: high curvature, steep gradient  
          ↓
```

SGD takes large steps in the steep direction (overshooting) and tiny steps in the flat direction. The result: zigzagging instead of going straight to the minimum.

**Problem 2: Different parameters need different step sizes**

- Parameter A: gradient is consistently small → needs larger effective step
- Parameter B: gradient is consistently large → needs smaller effective step

Plain SGD applies the same `lr` to all parameters, regardless of their gradient history.

**Momentum** solves Problem 1. **RMSprop** solves Problem 2. **Adam** solves both.

---

## Momentum

**Core idea:** Don't just use the current gradient — accumulate a velocity from all past gradients. Walk in the direction you've been consistently going.

Physical intuition: a ball rolling downhill. It builds up speed in the direction it's been rolling. Small obstacles (noise) don't stop it. It naturally smooths out the path.

### The algorithm

```
v_t = β · v_{t-1} + (1 - β) · g_t      # velocity update
w_t = w_{t-1} - lr · v_t                # weight update
```

Where:
- `v` = velocity (exponential moving average of gradients)
- `β` = momentum coefficient, typically `0.9`
- `g_t` = gradient at step `t`
- `lr` = learning rate

With `β = 0.9`:
```
v_1 = 0.1 · g_1
v_2 = 0.9 · v_1 + 0.1 · g_2 = 0.1·g_2 + 0.09·g_1
v_3 = 0.1·g_3 + 0.09·g_2 + 0.081·g_1
...
```

Gradients decay exponentially. The most recent gradient has weight `0.1`, the one before has `0.09`, etc. Gradients from ~10 steps ago have negligible weight (effective window ≈ `1/(1-β) = 10` steps).

### Why it works on oscillating gradients

In a narrow valley, gradients alternate direction in the narrow dimension but point consistently in the downhill direction:

```
Gradient directions over 4 steps:
  g1 = [+1, +10]    (right, strongly up)
  g2 = [+1, -10]    (right, strongly down)
  g3 = [+1, +10]    (right, strongly up)
  g4 = [+1, -10]    (right, strongly down)

Velocity with β=0.9:
  Narrow dimension (+/-10): cancels out over time → small velocity
  Wide dimension  (+1):    accumulates        → large velocity

Result: fast progress in the consistent direction, dampened oscillation
```

### Momentum hyperparameter

`β = 0.9` is the standard default. Higher β → more history, smoother but slower to react. Lower β → less history, faster to react but less smoothing.

**Backend analogy:** TCP congestion control. TCP builds up its sending rate (velocity) when ACKs come in consistently (consistent gradient direction). When it detects congestion (gradient reversal), it quickly reduces speed. The slow-start and AIMD algorithms are momentum-like algorithms for network optimization.

---

## Nesterov Momentum (NAG)

A subtle improvement: instead of computing the gradient at the current position, compute it at where you'll be after the momentum step:

```
# Standard momentum:
v_t = β · v_{t-1} + lr · g(w_{t-1})          # gradient at current position
w_t = w_{t-1} - v_t

# Nesterov:
v_t = β · v_{t-1} + lr · g(w_{t-1} - β · v_{t-1})  # gradient at lookahead position
w_t = w_{t-1} - v_t
```

The lookahead gradient is "more informative" — it tells you the slope at where you're going, not where you are.

Nesterov converges faster on convex problems and is sometimes used in ResNets with SGD. Adam doesn't use Nesterov by default (though some variants do).

---

## RMSprop

**Core idea:** Adapt the learning rate per-parameter based on the recent magnitude of that parameter's gradients.

- Parameters with consistently large gradients → reduce their effective lr
- Parameters with consistently small gradients → increase their effective lr

### The algorithm

```
s_t = β · s_{t-1} + (1 - β) · g_t²             # EMA of squared gradients
w_t = w_{t-1} - (lr / √(s_t + ε)) · g_t        # scaled update
```

Where:
- `s` = exponential moving average of squared gradients
- `β` = decay rate, typically `0.999`
- `ε` = small constant for numerical stability, typically `1e-8`

### Why squared gradients?

The square of the gradient measures the **magnitude** of gradient in each dimension (regardless of sign). Dividing by `√s` normalizes the step size:

```
Large recent gradients (large s): effective_lr = lr / √(large) = small
Small recent gradients (small s): effective_lr = lr / √(small) = large
```

Each parameter effectively gets its own learning rate, calibrated to its gradient history.

### Why EMA (not cumulative sum)?

AdaGrad (the predecessor) accumulated gradients indefinitely:

```
# AdaGrad (the problem):
G_t = G_{t-1} + g_t²    # cumulative sum — never decreases
w   = w - (lr / √G_t) · g  # learning rate shrinks forever → stalls
```

With cumulative sum, `G_t` only grows. Eventually `lr/√G_t → 0` and learning stops completely. This is catastrophic for long training runs.

RMSprop's EMA fixes this: old squared gradients "decay away" with factor `β`. If gradients become smaller later in training, the effective learning rate can recover.

```
# RMSprop (the fix):
s_t = 0.999 · s_{t-1} + 0.001 · g_t²   # old gradients fade out
```

### Backend analogy: per-endpoint rate limiting

Your API gateway has rate limits per endpoint. High-traffic endpoints get lower throughput limits; low-traffic endpoints get higher limits — automatically, based on observed traffic.

RMSprop does exactly this for model parameters:
- "High-traffic" parameters (frequent large gradients) → smaller effective step
- "Low-traffic" parameters (rare small gradients) → larger effective step

The exponential moving average is the same mechanism as an exponential moving average metric in your monitoring system — recent traffic matters more than ancient history.

---

## The problem both solve in different ways

| Problem | Symptom | Solution |
|---------|---------|---------|
| Oscillation in narrow valleys | Zigzagging, slow progress | Momentum: smooth out directions |
| Scale mismatch across parameters | Some params overtrained, some undertrained | RMSprop: normalize by gradient magnitude |

**The question naturally arises:** Can we use both at once?

Yes — that's **Adam**. Adam = Momentum + RMSprop, with an important correction for early steps. That's the next file.

---

## Convergence comparison

On a simple 2D problem `f(w1, w2) = w1² + 100·w2²` (very elongated bowl):

```
Starting at (5, 5):

SGD (lr=0.01):
  Step 5:   w ≈ (4.5, 3.2) — making progress
  Step 50:  w ≈ (2.1, 0.8) — slow, oscillating in w2 direction
  Step 500: w ≈ (0.1, 0.02) — converging

Momentum (β=0.9, lr=0.01):
  Step 5:   w ≈ (4.0, 1.5) — faster progress
  Step 50:  w ≈ (0.5, 0.1) — much less oscillation
  Step 200: w ≈ (0.001, 0.0001) — converged!

RMSprop (β=0.999, lr=0.01):
  Adapts step size per dimension:
    w1 direction: small gradients → larger effective lr
    w2 direction: large gradients → smaller effective lr
  Step 50: w ≈ (0.02, 0.001) — nearly converged

Adam (next file): combines both advantages
```

---

## Summary

| Algorithm | Update rule | Solves |
|-----------|------------|--------|
| SGD | `w -= lr · g` | Baseline |
| Momentum | `v = β·v + g;  w -= lr·v` | Oscillation, flat regions |
| RMSprop | `s = β·s + (1-β)·g²;  w -= lr·g/√(s+ε)` | Scale mismatch |

Key values:
- Momentum `β = 0.9` (standard)
- RMSprop `β = 0.999` (standard)
- `ε = 1e-8` (numerical stability)

Next: **Adam and AdamW** — the optimizer that powers every major LLM.
