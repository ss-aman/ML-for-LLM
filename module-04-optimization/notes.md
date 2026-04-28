# Module 04 — Optimization Algorithms

## Why this module matters for LLMs

Training a 70B-parameter LLM means adjusting 70 billion numbers simultaneously, thousands of times per second, on hundreds of GPUs, for months — without the training crashing or the model learning garbage.

That requires a very carefully engineered optimization strategy. Every major LLM paper specifies exactly which optimizer, which schedule, and which hyperparameters. After this module, you'll understand every choice.

---

## Reading Order

| File | Topic | Core idea |
|------|-------|-----------|
| `01_loss_landscape.md` | What we're navigating | The terrain gradient descent explores |
| `02_sgd_and_minibatch.md` | SGD and mini-batches | Why training on data subsets works and helps |
| `03_momentum_and_rmsprop.md` | Momentum + RMSprop | How to handle oscillation and adapt per-parameter |
| `04_adam_and_adamw.md` | Adam and AdamW | The optimizer every LLM actually uses |
| `05_lr_schedules.md` | Learning rate schedules | Warmup + cosine decay — real GPT-3/LLaMA numbers |
| `06_gradient_clipping.md` | Gradient clipping | Preventing training instability |
| `07_how_optimization_trains_llms.md` | Full picture | All components together with exact LLM hyperparameters |

---

## Code Files

| File | What it demonstrates |
|------|---------------------|
| `code_01_gradient_descent.py` | GD, SGD, mini-batch from scratch with visualizations |
| `code_02_optimizers.py` | Momentum, RMSprop, Adam, AdamW — all from scratch, comparison |
| `code_03_lr_schedules.py` | Warmup + cosine, step decay, cyclical LR |
| `code_04_llm_training_setup.py` | Complete LLM training loop: AdamW + schedule + clipping |

---

## Exercises

`exercises.py` — Implement from scratch, verify with the checker  
`solutions.py` — Reference solutions (try first!)

---

## The one-sentence summary

> **LLM training = AdamW optimizer + linear warmup + cosine decay schedule + gradient norm clipping, repeated for hundreds of billions of tokens — every component prevents a specific failure mode.**

---

## Module connections

```
Module 02 (Calculus/Gradients)
  └─ Gradient = direction of steepest increase
  └─ Gradient descent = walk opposite direction

Module 03 (Probability)
  └─ Cross-entropy = the loss we're minimizing

Module 04 (Optimization) ← YOU ARE HERE
  └─ How to walk: SGD → momentum → Adam → AdamW
  └─ How big to step: learning rate schedules
  └─ When to stop: gradient clipping, convergence

Module 06+ (Transformers, LLM Architecture)
  └─ The model being optimized
```

---

## Real LLM optimizer configs (what you'll understand by the end)

**GPT-3 (175B params):**
```
optimizer: Adam (β1=0.9, β2=0.95, ε=1e-8)
lr_max: 6e-4, warmup: 375M tokens, cosine decay to 6e-5
gradient clipping: 1.0, batch size: 3.2M tokens
```

**LLaMA 2 (70B params):**
```
optimizer: AdamW (β1=0.9, β2=0.95, ε=1e-5)
lr_max: 3e-4, warmup: 2000 steps, cosine decay to 3e-5
weight decay: 0.1, gradient clipping: 1.0
```
