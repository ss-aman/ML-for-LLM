# 04 — The Chain Rule: Derivatives Through Nested Functions

## The Problem

A neural network is a deeply nested function:

```
Loss = cross_entropy(
          softmax(
              W₂ @ relu(
                  W₁ @ x + b₁
              ) + b₂
          )
       )
```

You need `∂Loss/∂W₁` — how much does the loss change if you tweak the first
weight matrix? That requires differentiating through five nested functions.

The **chain rule** is the mathematical tool for this.

---

## The Chain Rule (One Variable)

If `h(x) = f(g(x))` — a function nested inside another function — then:

```
h'(x) = f'(g(x)) · g'(x)
```

Read: "derivative of the outer function (evaluated at the inner result),
times the derivative of the inner function."

### Example: `h(x) = sin(x²)`

- Inner function: `g(x) = x²`,   so `g'(x) = 2x`
- Outer function: `f(u) = sin(u)`, so `f'(u) = cos(u)`

```
h'(x) = f'(g(x)) · g'(x)
       = cos(x²) · 2x
```

Verify numerically:
```python
h = lambda x: np.sin(x**2)
# at x = 1.5:
h_prime_numerical = (h(1.5 + 1e-5) - h(1.5)) / 1e-5   # → -0.795
h_prime_analytical = np.cos(1.5**2) * 2 * 1.5           # → -0.795  ✓
```

---

## Extending to More Nesting

For three nested functions `k(x) = f(g(h(x)))`:

```
k'(x) = f'(g(h(x))) · g'(h(x)) · h'(x)
```

Multiply the derivatives of each layer, evaluated at the input flowing
through that layer.

For n layers, you get n multiplied terms. This "chain" of multiplications
is why it's called the chain rule — and why it's the foundation of
backpropagation.

---

## Backend Analogy: Latency Through a Service Chain

You have three microservices in sequence: A → B → C.

- Total latency: `L = f_A(f_B(f_C(x)))`
- A request enters C first, then B, then A.

Question: "If I slow down service C by 10ms, how much does total latency
increase?"

```
∂L/∂C = ∂L/∂A · ∂A/∂B · ∂B/∂C
```

The chain rule says: multiply the sensitivity of each service to its
upstream dependency. If B is highly sensitive to C (`∂B/∂C = 3`) and A
is moderately sensitive to B (`∂A/∂B = 2`), then the total effect
`∂L/∂C = 2 · 3 = 6` — a 10ms slowdown in C causes 60ms more total latency.

This is exactly how backpropagation computes `∂Loss/∂W₁` for an early
layer W₁: multiply the sensitivities through every layer between W₁ and
the loss.

---

## Partial Derivatives and the Chain Rule

When the intermediate result is a vector (as in neural networks), the chain
rule uses partial derivatives and sums:

```
∂L/∂xⱼ = Σᵢ (∂L/∂yᵢ) · (∂yᵢ/∂xⱼ)
```

"How much does L change if I change xⱼ?" = sum over all downstream yᵢ
of "how much does L change via yᵢ" times "how much does yᵢ change if I
change xⱼ".

In matrix notation (for a linear layer `y = Wx`):

```
∂L/∂x = Wᵀ · ∂L/∂y
∂L/∂W = ∂L/∂y · xᵀ
```

These two formulas are the gradient of a linear layer. They're used in
every backward pass through a dense/linear layer in any network.

---

## Computing the Chain Rule Step by Step

Suppose you have this tiny network (no activation for simplicity):

```
z = W @ x        (linear layer)
L = sum(z)       (sum the outputs, just for this example)
```

Forward pass:
```
x = [1, 2, 3]
W = [[0.1, 0.2, 0.3],
     [0.4, 0.5, 0.6]]
z = W @ x = [1.4, 3.2]
L = sum(z) = 4.6
```

Now backward pass — we want `∂L/∂W`:

Step 1: `∂L/∂z`
```
L = z₁ + z₂
∂L/∂z₁ = 1,  ∂L/∂z₂ = 1
So: ∂L/∂z = [1, 1]   (a gradient flows back from L to z)
```

Step 2: `∂L/∂W` (using the chain rule through the linear layer)
```
z = W @ x,  so  ∂z/∂W[i,j] = x[j]  (only affects row i)

∂L/∂W = (∂L/∂z) · xᵀ
       = [1, 1]ᵀ @ [1, 2, 3]
       = [[1, 2, 3],
          [1, 2, 3]]
```

This says: to improve `L`, adjust `W[0]` by `[1, 2, 3]` (times learning
rate), and similarly for `W[1]`. Each weight's gradient is proportional to
the corresponding input value that was multiplied by it.

---

## The Key Insight: Local Gradients

Each layer only needs to handle its **local computation**:

1. In the **forward pass**: compute output from input, remember the input
2. In the **backward pass**: receive `∂L/∂output`, compute and return `∂L/∂input`

The layer doesn't need to know what's upstream or downstream. It just
passes the gradient through using its own local derivative.

This modularity is why neural networks with arbitrary depth can be trained.
Each layer is a small, self-contained differentiable function.

---

## A Critical Warning: Vanishing Gradients

The chain rule multiplies gradients layer by layer. If each multiplication
gives a number less than 1, the product shrinks exponentially:

```
After 5 layers: 0.5 × 0.5 × 0.5 × 0.5 × 0.5 = 0.03
After 10 layers: 0.5^10 = 0.001
After 20 layers: 0.5^20 ≈ 0.000001
```

By the time the gradient reaches early layers, it's effectively zero. Those
layers stop learning.

This was the main obstacle to training deep networks before ~2015.

**Solutions used in LLMs:**
- **Residual connections** (Module 11): `output = F(x) + x` — the gradient
  can "shortcut" through the identity path
- **Layer normalization** (Module 08): keeps activation scales in a healthy
  range
- **Careful weight initialization**: ensures gradients start in a good range
- **GELU/ReLU activations**: less prone to saturation than sigmoid/tanh

---

## Key Takeaway

> The chain rule tells you how to differentiate through nested functions by
> multiplying local derivatives. A neural network is just many nested
> functions. Backpropagation is the chain rule applied systematically,
> starting from the loss and working backward through every layer.

---

## What's Next

`05_backpropagation.md`: applying the chain rule layer by layer through a
real neural network, step by step.
