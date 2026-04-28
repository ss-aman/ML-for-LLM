# 05 — Learning Rate Schedules

## Why a fixed learning rate isn't optimal

Using the same learning rate throughout training is suboptimal:

**Early training:** 
- Model weights are random, loss is high
- Large steps help move quickly toward good regions
- BUT starting with too large an lr causes instability ("exploding gradients")

**Late training:**
- Model is near a good minimum
- Large steps would overshoot and bounce around
- Small steps allow fine-grained refinement

**Solution:** Start with a small lr, increase to maximum, then decrease over time.

---

## Linear warmup

**What:** Linearly increase learning rate from 0 (or a very small value) to the target maximum over the first N steps.

```
lr(t) = lr_max · (t / warmup_steps)    for t ≤ warmup_steps
```

**Why it's needed:**

At the very start of training:
1. Model weights are random (large values, arbitrary directions)
2. Gradients are correspondingly large and noisy
3. Adam's second moment `v` starts at 0 — bias correction amplifies early gradients
4. A large initial lr combined with large initial gradients = immediate instability

The warmup period lets:
- Adam's moment estimates warm up (m and v accumulate reliable history)
- The model settle into a reasonable region before large steps
- The optimizer "calibrate" before running at full speed

**Backend analogy:** Ramping traffic to a new deployment. You don't immediately route 100% of traffic to the new service — you start at 1%, watch for errors, ramp to 10%, ramp to 100%. Warmup is the optimizer equivalent: don't immediately apply the full learning rate to randomly initialized weights.

**Typical warmup duration:**
- GPT-3: 375M tokens (about 0.13% of total training)
- LLaMA 2: 2000 steps
- Smaller models: 100-1000 steps

---

## Cosine decay

**What:** After warmup, decrease the learning rate following a cosine curve from `lr_max` down to `lr_min`.

```
lr(t) = lr_min + 0.5 · (lr_max - lr_min) · (1 + cos(π · progress))

where:
  progress = (t - warmup_steps) / (total_steps - warmup_steps)
  progress ∈ [0, 1]
```

At `progress=0`: `lr = lr_max`  (start of decay)
At `progress=1`: `lr = lr_min`  (end of training)

**Why cosine?**

The cosine curve has a useful property: it decays slowly at first (when the model still has a lot to learn) and then faster near the end (when fine-grained refinement dominates).

```
Linear decay vs Cosine decay:
  Linear:    ████████░░░░░░░░   (constant decay rate)
  Cosine:    ████████████░░░░   (slow then fast)
                                 └─ stays near max longer, useful for learning
```

The flat early portion of cosine matches well with how training typically proceeds: loss drops quickly early on (large steps appropriate), then plateaus requiring finer adjustments.

**Why not just train until convergence?**

For LLMs trained for one epoch, you know in advance how many steps you'll take. Cosine decay to a known end point is efficient and reproducible.

---

## The warmup + cosine schedule: the LLM standard

```python
def lr_schedule(step, warmup_steps, total_steps, lr_max, lr_min):
    if step < warmup_steps:
        # Linear warmup
        return lr_max * (step / warmup_steps)
    else:
        # Cosine decay
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * progress))
```

This is the schedule used in GPT-3, LLaMA, Falcon, Mistral, and virtually all modern LLMs.

---

## Real hyperparameters from LLM papers

**GPT-3 (175B parameters, 300B tokens):**
```
lr_max:      6e-4
lr_min:      6e-5  (10% of max)
warmup:      375M tokens (about 117 steps at batch size 3.2M)
decay:       cosine to 300B tokens
```

**LLaMA (7B–65B parameters, 1T+ tokens):**
```
lr_max:      3e-4  (7B), 1.5e-4 (65B)
lr_min:      3e-5  (10% of max)
warmup:      2000 steps
decay:       cosine
```

**LLaMA 2 (7B–70B parameters, 2T tokens):**
```
lr_max:      3e-4  (7B), 1e-4 (70B)
lr_min:      3e-5
warmup:      2000 steps
decay:       cosine to 2T tokens
```

**Pattern:** Larger models use smaller learning rates. Why? More parameters = more capacity = smaller steps needed for stability. This is the inverse "linear scaling rule" for model size.

---

## Step decay

An older, simpler schedule: multiply lr by a factor every N steps.

```python
def step_decay(step, lr_init, drop_factor=0.1, drop_every=1000):
    return lr_init * (drop_factor ** (step // drop_every))
```

**Effect:** lr stays constant, then suddenly drops by 10x at step 1000, 2000, etc.

**Use case:** ResNets with SGD, old-school computer vision. Not typically used for LLMs (cosine decay is smoother and more predictable).

---

## Linear warmup + linear decay

Some recent work (e.g., WSD schedule from MiniCPM) uses warmup → stable → linear decay:

```
Phase 1 (warmup):   lr linearly increases to lr_max  (~1% of training)
Phase 2 (stable):   lr stays at lr_max               (~90% of training)
Phase 3 (decay):    lr linearly decreases to lr_min  (~10% of training)
```

Advantage: you can extend training by extending phase 2 without redesigning the schedule. With cosine, the schedule is tied to the total training budget.

---

## Cyclical learning rates

Instead of monotonically decaying, cycle the learning rate up and down:

```
lr oscillates between lr_min and lr_max
with period = cycle_length steps
```

**Intuition:** High lr phases explore the loss landscape. Low lr phases refine within found valleys. Cycling allows "annealing" into progressively better regions.

Used in some research settings, but not the standard for LLM pretraining.

---

## Restart schedules (SGDR)

**Cosine Annealing with Warm Restarts** — after each cosine decay to lr_min, reset lr to lr_max and repeat with a longer period:

```
Cycle 1: warmup → decay (T₁ steps)
Cycle 2: warmup → decay (T₁ × mult steps)
Cycle 3: warmup → decay (T₁ × mult² steps)
```

Used in some fine-tuning scenarios. Not standard for LLM pretraining.

---

## The relationship between schedule and batch size

When batch size is doubled (and lr is doubled per the linear scaling rule):
- The warmup period should also be scaled proportionally
- The total number of steps changes (fewer steps with larger batch)
- The schedule should remain the same fraction of training

**GPT-3 batch size scaling:** During training, GPT-3 gradually increased batch size from 32k to 3.2M tokens, scaling lr linearly. The warmup remained ~0.1% of total steps.

---

## Why the minimum lr isn't zero

Setting `lr_min = 0` means the model stops learning completely at the end. Setting `lr_min = 0.1 × lr_max` (the standard) means some learning continues.

Rationale: the last few steps of training can still make meaningful improvements. Setting min too low wastes compute; setting min to 0 leaves improvement on the table.

---

## Summary

| Schedule | Formula | Use case |
|----------|---------|----------|
| Constant | `lr = c` | Debugging only |
| Linear warmup | `lr = lr_max · t/warmup_steps` | First N steps of all LLMs |
| Cosine decay | `lr = lr_min + 0.5·(lr_max-lr_min)·(1+cos(π·t/T))` | Standard LLM decay |
| Step decay | Multiply by factor every N steps | Old CV models |
| Cyclical | Oscillate between min and max | Some research |

**The LLM standard:** Linear warmup (few thousand steps) + cosine decay to `lr_min = lr_max/10`.

Next: **gradient clipping** — protecting training from occasional catastrophically large gradients.
