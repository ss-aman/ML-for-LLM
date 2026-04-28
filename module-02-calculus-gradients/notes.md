# Module 02 — Calculus & Gradients

## What This Module Is About

Module 01 showed you WHAT an LLM computes (matrix multiplications).  
This module shows you HOW an LLM **learns** — how parameters get adjusted.

The answer is calculus. Specifically, three connected ideas:

1. **Derivatives** — measure the rate of change of a function
2. **Gradients** — the multi-variable version of derivatives  
3. **Gradient descent** — use the gradient to repeatedly improve parameters

And one mechanism that makes it work in deep networks:  
4. **Backpropagation** — apply the chain rule layer by layer

---

## Reading Order

### Theory (read in order)

| File | Topic |
|---|---|
| `01_derivatives.md` | What a derivative is; numerical approximation |
| `02_gradient.md` | Gradient = vector of all derivatives; what it tells you |
| `03_gradient_descent.md` | The training algorithm; learning rate |
| `04_chain_rule.md` | Chain rule; composing derivatives through layers |
| `05_backpropagation.md` | Backprop step by step through a neural network |
| `06_how_gradients_train_llms.md` | Full LLM training loop; Adam; gradient clipping |

### Code (run after reading the corresponding theory)

| File | Covers |
|---|---|
| `code_01_derivatives.py` | Numerical derivatives; partial derivatives; comparing numerical vs analytical |
| `code_02_gradient_descent.py` | GD loop; learning rate effects; momentum; visualization |
| `code_03_backprop_from_scratch.py` | Manual backprop through a 2-layer net; gradient check |
| `code_04_autograd_preview.py` | PyTorch autograd doing the same thing automatically |

### Practice

| File | Description |
|---|---|
| `exercises.py` | Blank exercises to implement |
| `solutions.py` | Reference solutions |

---

## The One-Sentence Summary

> Training an LLM = repeatedly compute how much each parameter contributed
> to the error (backprop), then nudge every parameter slightly in the
> direction that reduces that error (gradient descent).

---

## Setup

```bash
pip install numpy matplotlib torch
```

PyTorch is only needed for `code_04_autograd_preview.py`.
All other files use only numpy.
