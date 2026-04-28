# 02 — The Gradient: All Derivatives at Once

## From One Variable to Many

A derivative handles one input variable: `f'(x)`.

Neural networks have millions (or billions) of parameters. Training requires
knowing how the loss changes with respect to **every single one** simultaneously.

The **gradient** is the solution: it stacks all the partial derivatives into
a single vector.

---

## Partial Derivatives: One Variable at a Time

When a function has multiple inputs, a **partial derivative** tells you how
it changes when you vary one input while holding all others fixed.

Notation: `∂f/∂x₁` = "partial derivative of f with respect to x₁"

**Example:**

```
f(w₁, w₂) = (w₁ - 3)² + (w₂ - 5)²
```

To find `∂f/∂w₁`: treat `w₂` as a constant, differentiate normally:
```
∂f/∂w₁ = 2(w₁ - 3)
```

To find `∂f/∂w₂`: treat `w₁` as a constant:
```
∂f/∂w₂ = 2(w₂ - 5)
```

---

## Backend Analogy: Profiling One Service at a Time

You're running a microservice system: auth, cache, database, serializer.
Total response time `T` depends on all four.

The partial derivative `∂T/∂cache_ttl` answers:
*"If I only change the cache TTL by a tiny amount (while leaving everything
else exactly the same), how does total response time change?"*

You're isolating the effect of one parameter. In a real system you'd measure
this by A/B testing. In ML, you compute it analytically or numerically.

---

## The Gradient: Stacking All Partial Derivatives

The **gradient** `∇f` (read "nabla f" or "del f") is the vector of all
partial derivatives:

```
∇f(w₁, w₂, ..., wₙ) = [ ∂f/∂w₁,  ∂f/∂w₂,  ...,  ∂f/∂wₙ ]
```

For our example `f(w₁, w₂) = (w₁-3)² + (w₂-5)²`:

```
∇f = [ 2(w₁-3),  2(w₂-5) ]
```

At the point `w = [0, 0]`:
```
∇f(0, 0) = [ 2(0-3),  2(0-5) ] = [ -6,  -10 ]
```

---

## What the Gradient Tells You

**The gradient points in the direction of steepest increase.**

More precisely:
- At any point, if you want to go "uphill" as fast as possible, move in the
  direction of `∇f`
- If you want to go "downhill" as fast as possible, move in direction `-∇f`
- The magnitude `||∇f||` tells you how steep the slope is

For training neural networks, you want to **minimize** loss → move in
the direction of `-∇f`. This is gradient descent.

```
gradient points uphill ↑         -gradient points downhill ↓

Loss surface:          f
        ↑ ∇f         /|
  ──────┼──────→    / |
        ↓ -∇f      /  |
                  ────── w
```

---

## The Gradient at a Minimum

At a minimum of `f`, the gradient is the **zero vector**:

```
∇f(w*) = [0, 0, ..., 0]
```

This makes sense: if you're at the bottom of a bowl, every direction is uphill.
No direction reduces the function further. This is the condition we're aiming
for during training.

At the minimum of `f(w₁, w₂) = (w₁-3)² + (w₂-5)²`:
- Point `w* = [3, 5]`
- `∇f(3, 5) = [ 2(3-3), 2(5-5) ] = [0, 0]` ✓

---

## Numerical Gradient (Works for Any Function)

Just like numerical derivatives, you can compute gradients numerically:

```python
def numerical_gradient(f, w, h=1e-5):
    grad = np.zeros_like(w)
    for i in range(len(w)):
        w_plus = w.copy(); w_plus[i] += h
        grad[i] = (f(w_plus) - f(w)) / h
    return grad
```

For each parameter `wᵢ`, perturb it by `h` and measure the change in loss.

This is **O(n)** in the number of parameters — for a 1-billion-parameter model
it would require 1 billion forward passes just to compute one gradient.
That's why we use backpropagation instead (one backward pass for all gradients).

But numerical gradients are still invaluable for **gradient checking**:
after implementing backprop, verify it gives the same result as the numerical
gradient on a small example.

---

## The Gradient in High Dimensions

A neural network like GPT-2 small has ~117 million parameters.

Its gradient is a vector of 117 million numbers: `[∂L/∂w₁, ∂L/∂w₂, ..., ∂L/∂w₁₁₇ₘ]`.

Each number says: "if I increase this parameter slightly, does the loss go up
or down, and by how much?"

Training adjusts all 117 million parameters simultaneously, each one moved
by a tiny amount in the direction that reduces loss.

You can't visualize a 117-million-dimensional space, but the math is identical
to the 2D case. The gradient is just a vector, and gradient descent is
just: `w = w - lr * ∇L(w)`.

---

## Gradient vs. Derivative: Summary

| | Derivative | Gradient |
|---|---|---|
| Applies to | Single-input functions | Multi-input functions |
| Result | A single number | A vector (one number per input) |
| Notation | `f'(x)` or `df/dx` | `∇f(w)` |
| Tells you | How f changes when x increases | Which direction to move in w-space to increase f |
| For training | Not enough | What you actually compute |

---

## Key Takeaway

> The gradient is the derivative generalized to many variables. It's a vector
> pointing in the direction that increases loss the most. Moving against it
> (−∇L) decreases loss. Training is just repeatedly computing this vector
> and taking a small step against it.

---

## What's Next

`03_gradient_descent.md`: the actual algorithm — how you use the gradient
to iteratively minimize the loss function and train a model.
