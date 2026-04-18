# 06 — How Gradients Actually Train an LLM

## The Complete Training Loop

Here is the full training loop for a language model. Every step uses
something from this module.

```python
for step in range(total_training_steps):

    # 1. Sample a batch of text
    tokens = sample_batch(dataset, batch_size)      # (B, seq_len)

    # 2. Forward pass: compute predictions
    logits = model.forward(tokens[:, :-1])          # (B, seq_len-1, vocab_size)
    targets = tokens[:, 1:]                          # shift by 1: predict next token

    # 3. Compute loss (cross-entropy)
    loss = cross_entropy(logits, targets)            # scalar

    # 4. Backward pass: compute gradients for all parameters
    loss.backward()

    # 5. Gradient clipping (prevent exploding gradients)
    clip_grad_norm_(model.parameters(), max_norm=1.0)

    # 6. Optimizer step: update all parameters
    optimizer.step()

    # 7. Zero out gradients for next step
    optimizer.zero_grad()
```

Each of the numbered steps connects directly to what you've learned.

---

## Step 3: The Loss Function for LLMs

Language models are trained with **cross-entropy loss**: predict the next
token from the previous context.

```
Loss = -log P(correct_next_token)
```

If the model assigns probability 0.8 to the correct token → loss ≈ 0.22 (low)
If the model assigns probability 0.01 to the correct token → loss ≈ 4.6 (high)

The model sees: `"the cat sat on the ___"`  
The correct next token is `"mat"`.  
The loss penalizes the model for assigning low probability to `"mat"`.

Cross-entropy is covered in detail in Module 03. For now: it's just
a differentiable function that produces a loss we can backpropagate through.

---

## Step 4: Backprop Through a Transformer

A transformer block has:
- Multi-head attention (matrix multiplications + softmax)
- Layer normalization
- Feed-forward network (two linear layers + GELU)
- Residual connections

Backprop flows through ALL of these. The computation graph is enormous —
GPT-3's graph for a single forward pass has billions of operations.

PyTorch builds and traverses this graph automatically. The math is the
same chain rule you just learned; the framework just does it at scale.

---

## Step 5: Gradient Clipping

Deep networks can produce very large gradients ("exploding gradients"),
especially early in training when parameter values are random.

**Gradient clipping** caps the total gradient norm:

```python
# Compute L2 norm of all gradients concatenated
total_norm = sqrt(sum(grad.norm()**2 for all grad))

# If too large, scale all gradients down
if total_norm > max_norm:
    for param in model.parameters():
        param.grad *= max_norm / total_norm
```

This prevents one bad batch from causing a catastrophically large parameter
update that destabilizes training.

**Max norm = 1.0** is the standard for transformer training (used in GPT-2,
GPT-3, LLaMA, etc.).

---

## Step 6: The Adam Optimizer

Pure gradient descent (`w -= lr * grad`) has problems:
- Some parameters need large steps, others need small steps
- Gradients are noisy — the direction oscillates

**Adam** (Adaptive Moment Estimation) solves both problems. It maintains
a running estimate of:
- `m`: the gradient direction (1st moment, like momentum)
- `v`: the gradient magnitude squared (2nd moment, for scaling)

```python
# Per-parameter update (conceptually):
m = β₁ * m + (1 - β₁) * grad          # exponential moving average of gradient
v = β₂ * v + (1 - β₂) * grad**2       # exponential moving average of gradient²

m_hat = m / (1 - β₁**t)               # bias correction (early steps)
v_hat = v / (1 - β₂**t)

w -= lr * m_hat / (sqrt(v_hat) + ε)   # adaptive update
```

Default values used by virtually every LLM:
```
β₁ = 0.9     (momentum coefficient)
β₂ = 0.95    (variance coefficient, sometimes 0.999)
ε  = 1e-8    (prevent division by zero)
lr = 1e-4 to 6e-4   (tuned per model)
```

**Why Adam works:**
- `m` provides momentum: accumulates direction, dampens oscillation
- `v` provides adaptive scaling: parameters with large gradient history get
  smaller effective lr; rarely updated parameters get larger effective lr
- Result: fast, stable convergence on diverse parameter types

Adam is covered in depth in Module 04.

---

## Learning Rate Schedule Used in LLM Training

This is the exact schedule used to train GPT-3, LLaMA, and similar models:

```
Phase 1 — Warmup (first ~1% of steps):
    lr increases linearly from ~0 to lr_max

Phase 2 — Cosine decay (remaining 99%):
    lr decreases following cosine curve
    from lr_max down to lr_min (typically lr_max / 10)

lr at step t:
    if t < warmup_steps:
        lr = lr_max * (t / warmup_steps)
    else:
        progress = (t - warmup_steps) / (total_steps - warmup_steps)
        lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(π * progress))
```

**GPT-3 specifics:**
- `lr_max = 6e-4`
- `warmup_steps = 375M tokens` (out of 300B total)
- `lr_min = 6e-5`

---

## What "Training" Means for an LLM at Scale

| Scale | Numbers |
|---|---|
| Training tokens | GPT-3: 300B; LLaMA-2: 2T; Chinchilla: 1.4T |
| Parameters | GPT-3: 175B; LLaMA-2-70B: 70B |
| Batch size | Typically 0.5M–4M tokens per step |
| Training steps | ~500K–3M gradient updates |
| GPU hours | GPT-3: ~3,640 V100 days |
| Each step | One forward pass + one backward pass + one Adam update |

Each "step" above:
1. Runs the 300B-parameter forward pass on a batch → chain of matrix multiplications
2. Computes cross-entropy loss → a scalar
3. Runs backpropagation → 300B gradients, one per parameter
4. Applies Adam → 300B parameter updates

Gradient descent, repeated 3 million times. Nothing conceptually new.
Just scale.

---

## Gradient Flow as a Design Principle

Once you understand backprop, you can read architectural choices as
deliberate decisions about gradient flow:

| Design Choice | Backprop Effect |
|---|---|
| Residual connections | Gradient highway bypasses layers; no vanishing |
| Layer normalization | Keeps activations in range; prevents exploding gradients |
| GELU instead of sigmoid | GELU saturates less; better gradient flow in deep nets |
| Weight initialization (e.g. GPT-2 scales by 1/√N) | Prevents gradient explosion at initialization |
| Gradient clipping | Prevents any single batch from causing catastrophic updates |

When you understand gradients, the transformer architecture stops being
arbitrary choices and starts being a carefully engineered gradient flow system.

---

## Key Takeaway

> Training an LLM is gradient descent at massive scale. Every architectural
> decision in a transformer (residuals, layernorm, attention scaling) is
> designed to keep gradients flowing cleanly through hundreds of layers
> and millions of update steps. The math — derivatives, chain rule, gradient
> descent — is the same regardless of scale.
