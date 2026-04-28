# 07 — How Optimization Trains LLMs

## All components together

Every concept from this module combines into the LLM training loop. Here it is:

```python
# Complete LLM training setup

# 1. Model + optimizer
model     = GPTModel(d_model=4096, n_layers=32, n_heads=32, vocab=50257)
optimizer = AdamW(
    model.parameters(),
    lr           = 3e-4,    # will be overridden by schedule
    betas        = (0.9, 0.95),
    eps          = 1e-8,
    weight_decay = 0.1,
)

# 2. Learning rate schedule
scheduler = WarmupCosineSchedule(
    optimizer,
    warmup_steps = 2000,
    total_steps  = 1_000_000,
    lr_max       = 3e-4,
    lr_min       = 3e-5,
)

# 3. Training loop
for step, batch in enumerate(dataloader):
    tokens = batch["input_ids"]    # shape: (batch_size, seq_len)
    labels = batch["labels"]       # shifted tokens

    # --- Forward pass ---
    logits = model(tokens)         # shape: (batch_size, seq_len, vocab_size)
    loss   = cross_entropy(logits, labels)

    # --- Backward pass ---
    optimizer.zero_grad()
    loss.backward()                # compute gradients via chain rule

    # --- Gradient clipping ---
    grad_norm = clip_grad_norm_(model.parameters(), max_norm=1.0)

    # --- Optimizer step (AdamW) ---
    optimizer.step()

    # --- LR schedule step ---
    scheduler.step()

    # --- Logging ---
    if step % 100 == 0:
        print(f"step={step}, loss={loss:.4f}, "
              f"lr={scheduler.get_lr():.2e}, grad_norm={grad_norm:.4f}")
```

Six lines of "real" work in the loop: forward, zero_grad, backward, clip, step, schedule. Everything else is logging.

---

## The exact hyperparameters from major LLM papers

### GPT-3 (OpenAI, 2020)

```
Model: 175B parameters
Training: 300B tokens (Common Crawl, WebText, Books, Wikipedia)

Optimizer:     Adam  (β1=0.9, β2=0.95, ε=1e-8)
               Note: GPT-3 used Adam not AdamW (earlier paper)
lr_max:        6e-4  (smallest 125M model), decreasing to 1e-4 (175B model)
lr_min:        10% of lr_max
Warmup:        375M tokens
Schedule:      cosine decay over 260B tokens, then constant

Batch size:    3.2M tokens (gradually increased from 32k)
Grad clipping: 1.0
Weight decay:  0.1 (AdaGrad-style, not decoupled)
```

### LLaMA (Meta, 2023)

```
Model: 7B–65B parameters
Training: 1T–1.4T tokens

Optimizer:     AdamW (β1=0.9, β2=0.95, ε=1e-5)
lr_max:        3e-4 (7B), 1.5e-4 (65B)  ← smaller model = higher lr
lr_min:        10% of lr_max
Warmup:        2000 steps
Schedule:      cosine decay

Batch size:    4M tokens
Grad clipping: 1.0
Weight decay:  0.1
```

### LLaMA 2 (Meta, 2023)

```
Model: 7B–70B parameters
Training: 2T tokens

Optimizer:     AdamW (β1=0.9, β2=0.95, ε=1e-5)
lr_max:        3e-4 (7B), 1e-4 (70B)
lr_min:        3e-5
Warmup:        2000 steps
Schedule:      cosine decay

Batch size:    4M tokens
Grad clipping: 1.0
Weight decay:  0.1
```

### Mistral 7B (Mistral AI, 2023)

```
Optimizer:     AdamW (β1=0.9, β2=0.95)
lr_max:        3e-4
Warmup:        ~ linear
Schedule:      cosine
Grad clipping: 1.0
Weight decay:  0.1
```

**The pattern is clear:** Every modern LLM uses nearly identical optimization setup.

---

## Reading the training loss curve

When training an LLM, the loss curve tells you everything about optimizer health:

```
Healthy training curve:
  Steps 1-100:     loss drops rapidly (5.0 → 3.5)  ← warmup + fast initial learning
  Steps 100-10k:   loss drops steadily (3.5 → 2.8)  ← cosine decay working
  Steps 10k-100k:  loss slows down (2.8 → 2.3)      ← approaching information limit
  Steps 100k+:     loss flattens (2.3 → 2.1)         ← fine-grained improvement

Perplexity interpretation:
  loss = 3.5 → PPL = 33  (guessing among ~33 tokens per position)
  loss = 2.5 → PPL = 12  (guessing among ~12 tokens)
  loss = 2.0 → PPL = 7.4 (guessing among ~7 tokens)
```

```
Problems to watch for:

Loss spike (sudden ↑):
  → Bad data batch (corrupt/duplicate/encoding error)
  → Gradient explosion despite clipping
  → Sometimes resolves naturally; otherwise investigate data

Loss plateau:
  → Learning rate too low (warmup ended too early?)
  → Model saturated for current batch size
  → Gradient clipping too aggressive

Loss divergence (↑ continuously):
  → Learning rate too high
  → Gradient clipping value too high
  → Numerical instability (try lower lr, check bf16 overflow)
```

---

## Gradient accumulation for large effective batches

Most LLMs use effective batch sizes of millions of tokens. But a single GPU may only fit a batch of 8-32 sequences. Gradient accumulation bridges the gap:

```python
accumulation_steps = 128    # effective_batch = 32 * 128 = 4096 sequences

optimizer.zero_grad()
for micro_step in range(accumulation_steps):
    batch = get_next_micro_batch()
    loss  = model(batch) / accumulation_steps  # scale loss by 1/accumulation
    loss.backward()                             # gradients accumulate (add up)

# After all micro-steps: clip and update
clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
optimizer.zero_grad()
```

This is mathematically equivalent to computing one gradient over the full large batch — the gradient is the average over all micro-batches (scaling the loss by `1/accumulation_steps` achieves this).

---

## Mixed precision training

Real LLM training uses bf16 (16-bit bfloat) for compute, fp32 for optimizer state:

```python
# Training in mixed precision
with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    logits = model(tokens)           # forward pass in bf16 (fast)
    loss   = cross_entropy(logits, labels)

loss.backward()                      # backward in bf16

clip_grad_norm_(model.parameters(), max_norm=1.0)

optimizer.step()                     # optimizer state in fp32 (stable)
```

**Why?**
- bf16 forward + backward: 2x faster, 2x less memory
- fp32 optimizer state: prevents precision loss in small gradient updates
- The tradeoff: speed gain with minimal accuracy loss

This is why `ε = 1e-5` (not 1e-8) in LLaMA — bf16 has less precision, and a larger epsilon prevents numerical instability in the Adam denominator.

---

## Optimizer state sharding (ZeRO)

For the largest models (70B+), even with mixed precision, the optimizer state doesn't fit on one GPU:

```
7B model:
  Weights (bf16):    14 GB
  Adam m (fp32):     28 GB
  Adam v (fp32):     28 GB
  Total:             70 GB  ← exceeds a single A100 GPU (80 GB)
```

**ZeRO (Zero Redundancy Optimizer)** shards the optimizer state across GPUs:

```
ZeRO-1: Shard optimizer state across GPUs
ZeRO-2: Shard optimizer state + gradients
ZeRO-3: Shard optimizer state + gradients + model parameters

With 8 GPUs:
  ZeRO-1: optimizer state per GPU = 70 GB / 8 = 8.75 GB (fits!)
```

This is what DeepSpeed implements. All large open-source LLM training uses ZeRO or equivalent (FSDP in PyTorch).

---

## The complete per-step cost

For a 7B model, one training step involves:

```
1. Forward pass:
   - 7B multiplications + additions
   - Activations stored for backward: ~8-16 GB

2. Loss computation:
   - Cross-entropy over batch × seq_len × vocab_size

3. Backward pass:
   - Chain rule through all layers: ~2× forward cost
   - Gradient tensor: 14 GB (fp32)

4. Gradient clipping:
   - Compute global norm: one pass over all gradients
   - Scale if needed

5. AdamW update:
   - Update m, v, apply bias correction
   - Decoupled weight decay
   - One pass over all 7B parameters

Total per step: ~4-5× forward pass cost
```

Training a 7B model on 1T tokens with batch size 4M tokens:
- Steps: 1T / 4M = 250,000 steps
- Compute: ~6,000 GPU-hours on A100s
- Time: ~12 days on 20 A100s

---

## The full annotated picture

```
Training tokens in training corpus
    ↓
Tokenizer → token IDs
    ↓
Data loader → mini-batches (batch_size × seq_len)
    ↓
Forward pass:
  embedding lookup  (Module 01: matrix row lookup)
  transformer layers (Module 01: matrix mul, Module 02: activation)
  output logits     (vocab_size per position)
    ↓
Cross-entropy loss  (Module 03: -log P(correct_token))
    ↓
Backward pass       (Module 02: chain rule, backprop)
    ↓
Gradient clipping   (Module 04: ‖g‖ → max_norm)
    ↓
AdamW update        (Module 04: m, v, bias correct, weight decay)
    ↓
LR schedule step    (Module 04: warmup + cosine)
    ↓
Repeat for 250,000+ steps
    ↓
Trained LLM
```

---

## Summary: the full LLM optimizer config

```python
# What every modern LLM uses:

optimizer = AdamW(
    lr           = 3e-4,     # base lr (overridden by schedule)
    betas        = (0.9, 0.95),  # NOT the Adam defaults (0.9, 0.999)
    eps          = 1e-8,
    weight_decay = 0.1,
)

scheduler = WarmupCosineSchedule(
    warmup_steps = 2000,          # a few thousand steps
    total_steps  = total_steps,
    lr_max       = 3e-4,
    lr_min       = 3e-5,          # 10% of max
)

# In the loop:
loss.backward()
clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
scheduler.step()
```

Five decisions. All motivated by specific failure modes. All standard across the field.

---

## What's next

You now understand everything needed to train a language model from scratch:
- **Module 01**: the linear algebra of the forward pass (matrix mul, embeddings, attention)
- **Module 02**: how gradients are computed (chain rule, backpropagation)
- **Module 03**: what the loss means (cross-entropy, perplexity)
- **Module 04**: how weights are updated (AdamW + schedule + clipping)

**Module 05** (Regularization) covers how to prevent overfitting — keeping the model from memorizing training data instead of learning generalizable patterns. Dropout, layer norm, and early stopping.

**Module 06+** will cover the transformer architecture in detail — the specific model architecture that all modern LLMs use.
