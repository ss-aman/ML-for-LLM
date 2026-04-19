# 06 — Gradient Clipping

## The problem: exploding gradients

During training, occasionally the gradient can become extremely large — sometimes thousands of times larger than typical.

This is called an **exploding gradient** and it can completely destabilize training:

```
Normal step:  w_new = w - 0.001 * 0.05   = w - 0.00005  (small, safe update)
Explosion:    w_new = w - 0.001 * 15000  = w - 15.0     (huge, destructive update)
```

After one bad step, the model weights are in a completely wrong region, the loss spikes to an extreme value, and subsequent gradients are even more extreme. Training crashes.

---

## Why explosions happen

### Deep networks and the chain rule

Backpropagation multiplies many gradients together (chain rule across layers). If any layer has a weight slightly larger than 1, repeated multiplication grows exponentially:

```
Gradient at layer L = g_L × W_L × W_{L-1} × ... × W_1

If each W_i has values ≈ 1.1:
  After 50 layers: 1.1^50 ≈ 117   (gradient is 117x amplified)
  After 100 layers: 1.1^100 ≈ 13781  (explosive)
```

This is most common in:
- Very deep networks (transformers with many layers)
- Recurrent networks (LSTM, RNN — effectively infinite depth)
- Early training when weights are poorly initialized

### Rare but outlier data

Some training examples have unusual patterns that produce very large activations, which produce very large gradients. In a corpus of trillions of tokens, even rare patterns occur frequently in absolute terms.

### Loss landscape cliffs

Some loss landscapes have "cliffs" — regions where the loss changes very steeply. Crossing a cliff produces enormous gradients.

---

## Gradient clipping: norm clipping

The standard solution: **clip the gradient by its global norm**.

### The algorithm

1. Compute the norm (magnitude) of the entire gradient vector
2. If the norm exceeds a threshold, scale the entire gradient down proportionally

```python
def clip_gradient_norm(gradients, max_norm):
    """
    gradients: list of gradient arrays (one per parameter)
    max_norm:  maximum allowed gradient norm
    """
    # Global norm: sqrt(sum of squares of all gradient values)
    total_norm = np.sqrt(sum(np.sum(g**2) for g in gradients))

    if total_norm > max_norm:
        clip_coef = max_norm / total_norm    # scale factor
        gradients = [g * clip_coef for g in gradients]

    return gradients
```

**Key property:** Direction is preserved, only magnitude is reduced.

```
Without clipping:  g = [0.0, 1500.0]   (huge gradient in one direction)
After clipping:    g = [0.0, 1.0]      (same direction, capped magnitude)
```

The update still goes in the right direction — we just prevent it from being too large.

---

## Norm clipping vs value clipping

### Norm clipping (standard)
Clip the **global norm** of all gradients together:

```
if ‖g‖ > max_norm:
    g ← g × (max_norm / ‖g‖)
```

All parameters are scaled by the same factor. The relative proportions of gradients across parameters are preserved. This is the correct approach.

### Value clipping (not recommended for standard training)

Clip each gradient value independently:
```
g_i ← clip(g_i, -threshold, +threshold)
```

Problems: distorts the relative magnitudes of different parameter gradients. Only used in specific settings (e.g., DQN reinforcement learning).

---

## Effect on training

Gradient clipping provides a safety mechanism:

```
Without clipping:
  - Normal steps: gradient norm ≈ 0.1-2.0 → safe updates
  - Occasional bad step: gradient norm = 500 → training crashes
  - Recovery: may take many steps or never recover

With clipping (max_norm=1.0):
  - Normal steps: norm < 1.0 → no clipping, unchanged
  - Occasional bad step: norm = 500, clipped to 1.0 → training continues
  - Gradient norm log: visible spike but training recovers
```

In practice, clipping fires rarely during stable training. You can monitor the gradient norm to detect problems:

```python
# Training loop with gradient norm monitoring
grad_norm = compute_gradient_norm(model.parameters())
print(f"Step {step}: grad_norm={grad_norm:.4f}, clipped={grad_norm > max_norm}")
clip_gradients(model.parameters(), max_norm=1.0)
optimizer.step()
```

If clipping fires constantly, something is wrong (lr too high, bad initialization, degenerate data).

---

## Standard LLM gradient clipping values

All major LLMs use gradient norm clipping with `max_norm = 1.0`:

```
GPT-3:      gradient clipping = 1.0
GPT-2:      gradient clipping = 1.0
LLaMA:      gradient clipping = 1.0
LLaMA 2:    gradient clipping = 1.0
PaLM:       gradient clipping = 1.0
Falcon:     gradient clipping = 1.0
Mistral:    gradient clipping = 1.0
```

`max_norm = 1.0` is essentially universal for transformer LLMs. Why?

- Transformers with proper weight initialization (Xavier/He) typically have gradient norms in the range 0.1–2.0 during stable training
- Setting max_norm=1.0 clips only the outlier explosions (norm > 1), not normal gradients
- Too low (e.g., 0.1): clips normal gradients, slows training
- Too high (e.g., 10.0): doesn't catch explosions early enough

---

## Gradient clipping and AdamW

With AdamW, the clipping happens **before** the optimizer step:

```python
# Complete training step
loss.backward()                             # compute gradients
clip_gradient_norm(model.params, max=1.0)   # clip BEFORE optimizer
optimizer.step()                             # AdamW update
optimizer.zero_grad()                        # clear for next step
```

In PyTorch:

```python
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
optimizer.zero_grad()
```

`clip_grad_norm_` does exactly what we described: computes global norm, clips if necessary, in-place modification.

---

## Detecting gradient problems in practice

Monitoring gradient norm during LLM training reveals:

```
Healthy training (typical patterns):
  Steps 1-100:    norm = 0.5-2.0  (high early, model learning fast)
  Steps 100-1000: norm = 0.3-1.5  (settling)
  Steps 1000+:    norm = 0.1-0.8  (stable)

Warning signs:
  Constant norm > 5.0:   learning rate too high, or initialization problem
  Sudden spike to 100+:  potential loss spike, check data pipeline
  Norm → 0:              vanishing gradients, model not learning

Loss spike + gradient spike:
  → Usually caused by: bad data batch (wrong encoding, NaN values)
  → Or: numerical precision issue (overflow in fp16)
  → Gradient clipping limits damage, but spike should be investigated
```

---

## Backend analogy: circuit breakers

A circuit breaker in your service mesh cuts off a failing service before it takes down the entire system. If a downstream service starts returning errors at 1000 RPS, the circuit breaker opens and returns a fallback response.

Gradient clipping is the circuit breaker for the training loop:
- Normal operation: gradient flows through unchanged
- Anomaly detected (norm > threshold): circuit clips the gradient before it damages the model weights
- Training continues from a safe state

Both protect a running system from occasional pathological events without stopping the system.

---

## Summary

| Aspect | Detail |
|--------|--------|
| Problem | Exploding gradients crash training |
| Solution | Clip gradient norm to max_norm |
| Formula | `if ‖g‖ > max_norm: g ← g · max_norm/‖g‖` |
| Standard value | `max_norm = 1.0` (all major LLMs) |
| Direction preserved? | Yes |
| When to clip | Between `loss.backward()` and `optimizer.step()` |

Next: the **full picture** — all optimizer components combined into a real LLM training loop.
