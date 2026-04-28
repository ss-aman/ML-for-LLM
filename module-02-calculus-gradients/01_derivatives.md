# 01 — Derivatives: Measuring Rate of Change

## Why ML Needs Derivatives

In ML, you have a **loss function** — a number that says "how wrong is the
model right now?" Training means making that number smaller.

To make it smaller, you need to know: *if I change this parameter slightly,
does the loss go up or down?*

That question — "how does output change as input changes?" — is exactly what
a derivative answers.

---

## What Is a Derivative?

The derivative `f'(x)` (read: "f prime of x") measures the **rate of change**
of `f` at the point `x`.

Formally, it's the slope of the function — how much `f` rises per unit of `x`.

```
           rise     f(x + h) - f(x)
slope = ─────── = ──────────────────   as h → 0
           run            h
```

The "as h → 0" part means: take an infinitesimally small step, measure the
rise, divide by the step. That gives the instantaneous rate of change.

---

## Backend Analogy: Server Response Time

Think of `f(x)` as your API's p99 latency, where `x` is the number of
concurrent connections.

- `f'(x) = 5` at `x = 100` means: "adding one more connection right now
  increases p99 latency by ~5ms"
- `f'(x) = 0.1` at `x = 10` means: "at low load, adding a connection barely
  matters"
- `f'(x) = 50` at `x = 500` means: "you're near saturation — every extra
  connection is very expensive"

The derivative tells you the sensitivity of the output to a change in input —
at the current operating point.

---

## The Geometric Picture

The derivative at `x` is the **slope of the tangent line** at that point.

```
f(x)
  │         /
  │        / ← tangent line, slope = f'(x)
  │       /
  │   ___/___
  │──/──────── x
```

- Positive slope (`f'(x) > 0`): function is increasing at x
- Negative slope (`f'(x) < 0`): function is decreasing at x
- Zero slope (`f'(x) = 0`): at a minimum, maximum, or flat region

**For training:** you want the loss to decrease. If `f'(w) > 0`, increasing `w`
increases the loss — so you should *decrease* `w`. The gradient tells you
which way to go.

---

## Key Derivatives to Know

These come up constantly. Memorize them or keep this as a reference:

| `f(x)` | `f'(x)` | Notes |
|---|---|---|
| `c` (constant) | `0` | Constants don't change |
| `x` | `1` | |
| `x²` | `2x` | |
| `xⁿ` | `n · xⁿ⁻¹` | Power rule |
| `eˣ` | `eˣ` | The exponential is its own derivative |
| `ln(x)` | `1/x` | The natural log |
| `sin(x)` | `cos(x)` | |
| `cos(x)` | `−sin(x)` | |

### Derivative rules (for combining functions)

**Sum rule:** `(f + g)' = f' + g'`  
**Product rule:** `(f · g)' = f' · g + f · g'`  
**Chain rule:** `(f(g(x)))' = f'(g(x)) · g'(x)` ← covered in depth in `04_chain_rule.md`

---

## Numerical Derivatives: No Formula Needed

Here's the key insight: you don't need the analytical formula to compute a
derivative. You can always **estimate** it numerically:

```python
def derivative(f, x, h=1e-5):
    return (f(x + h) - f(x)) / h
```

This is the **finite difference** method. Nudge `x` by a tiny amount `h`,
measure how much `f` changes, divide by `h`.

```python
f = lambda x: x**2
derivative(f, 3)   # → ~6.0  (exact: 2*3 = 6)
derivative(f, -2)  # → ~-4.0 (exact: 2*(-2) = -4)
```

**Why this matters:**
- During ML development, you use numerical derivatives to **verify** your
  analytical gradients are correct ("gradient checking")
- Frameworks like PyTorch compute gradients analytically (via backprop),
  but you can use finite differences to double-check them
- For simple experiments where performance doesn't matter, numerical
  derivatives just work

**Choosing `h`:**
- Too large: inaccurate (not truly "at the limit")
- Too small: floating-point errors dominate
- `h = 1e-5` is a reliable default

### Centered difference (more accurate)

Instead of `(f(x+h) - f(x)) / h`, use both sides:

```
f'(x) ≈ (f(x+h) - f(x-h)) / (2h)
```

This is the **centered difference** — symmetric around `x`. It's more
accurate at the same `h` because it cancels out the leading error term.

```python
def derivative_centered(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)
```

---

## Common Derivatives in ML

### ReLU

```
ReLU(x) = max(0, x)

ReLU'(x) = 0  if x < 0
            1  if x > 0
            undefined at x = 0 (in practice use 0 or 1 here)
```

**Why it matters:** ReLU is the most common activation function. You need
its derivative for backpropagation through a layer that uses ReLU.

### Sigmoid

```
σ(x) = 1 / (1 + e^(-x))

σ'(x) = σ(x) * (1 - σ(x))
```

**Why it matters:** Used in output layers for binary classification. Also
used in LSTM gates (Module 09). The derivative has a nice form in terms of
the function itself.

### Softmax + Cross-Entropy (preview)

The loss function used to train LLMs. The derivative works out cleanly —
covered in Module 03.

---

## What "Derivative = 0" Means for Training

When `f'(w) = 0`, the function is flat at that point. This happens at:
- **Minima**: bowl-shaped, derivatives zero at the bottom → what we want
- **Maxima**: hill-shaped, derivatives zero at the top → bad, but rare
- **Saddle points**: zero in some directions, not others → common in high-D, mostly fine

In neural networks with billions of parameters, the loss landscape is
extremely high-dimensional. "Finding a minimum" in that space means finding
a point where the gradient (the multi-dimensional version of the derivative)
is approximately zero. That's training.

---

## Key Takeaway

> A derivative answers "if I change this input slightly, how much does the
> output change?" In ML, the input is a weight parameter and the output is
> the loss. The derivative tells you which direction to adjust the weight.

---

## What's Next

`02_gradient.md`: when you have millions of parameters, you compute the
derivative with respect to ALL of them simultaneously. That's the gradient.
