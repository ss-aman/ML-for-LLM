# Module 02: Calculus and Gradients for ML

> **Who this is for:** You understand vectors and matrices from Module 01. Now we need to understand *how to improve* a model — which means knowing how to measure "in which direction should I nudge these parameters to reduce error?" That's calculus.

---

## 1. Why Calculus? The Core Problem

In ML, we have a **loss function** — a number that measures how wrong our model is. Training means minimizing that loss. The question is: *which way do we adjust the parameters to make the loss go down?*

This is exactly the question calculus was invented to answer.

**Backend analogy:** Imagine you're tuning a server's thread pool size, connection timeout, and cache TTL to minimize p99 latency. You don't know the formula for latency — but you can measure it. Calculus gives you the tools to know: "if I increase the thread pool size by a tiny amount, does latency go up or down, and by how much?"

---

## 2. Derivatives — Rate of Change

The **derivative** `f'(x)` (or `df/dx`) answers: "if I change `x` by a tiny amount, how much does `f(x)` change?"

More precisely, it's the slope of the function at point `x` — the rise-over-run as the horizontal step shrinks to zero:

```
f'(x) = lim(h→0) [ f(x+h) - f(x) ] / h
```

**Backend analogy:** Think of monitoring API response time as a function of concurrent connections. The derivative at any point tells you: "right now, if I add one more connection, response time increases by X milliseconds." That's the rate of change — and it tells you whether you're in a safe operating zone or approaching a cliff.

### Key derivatives to memorize

| `f(x)` | `f'(x)` |
|---|---|
| `x^n` | `n * x^(n-1)` |
| `e^x` | `e^x` (special — its own derivative) |
| `ln(x)` | `1/x` |
| `sin(x)` | `cos(x)` |
| constant | `0` |

### Numerical approximation

You can always *estimate* a derivative numerically:

```python
def derivative(f, x, h=1e-5):
    return (f(x + h) - f(x)) / h
```

This is the "finite difference" method — nudge `x` by a tiny amount and measure the effect. Less precise than the analytical formula, but works for any function.

---

## 3. Partial Derivatives — One Variable at a Time

Most ML functions take many inputs (hundreds of millions of parameters). A **partial derivative** `∂f/∂x_i` measures how `f` changes when you vary *only* `x_i`, holding all other variables constant.

```
f(w1, w2) = (w1 - 3)^2 + (w2 - 5)^2

∂f/∂w1 = 2*(w1 - 3)    ← treat w2 as a constant
∂f/∂w2 = 2*(w2 - 5)    ← treat w1 as a constant
```

**Backend analogy:** This is like profiling a microservice system. You want to know: "if I scale *just* the auth service, how does overall latency change?" — while holding all other services constant. Partial derivatives let you isolate the effect of changing one parameter at a time in a system with many interdependencies.

---

## 4. The Gradient — A Vector of All Partial Derivatives

The **gradient** `∇f` (del f or "nabla f") of a multi-variable function is a vector that stacks all its partial derivatives:

```
∇f(w1, w2) = [ ∂f/∂w1,  ∂f/∂w2 ]
```

For `f(w1, w2) = (w1-3)^2 + (w2-5)^2`:
```
∇f = [ 2*(w1-3),  2*(w2-5) ]
```

**The gradient points in the direction of steepest increase.** Its negative (`-∇f`) points in the direction of steepest *decrease* — toward the minimum.

**Backend analogy:** The gradient is like a compass that always points toward higher ground (increasing loss). When you're tuning server config, the gradient of the latency function tells you: "the single direction in (thread_pool, cache_ttl, timeout) space that increases latency the fastest." To *minimize* latency, you move in the *opposite* direction.

---

## 5. Gradient Descent — Auto-Tuning Your Model

**Gradient descent** is the core optimization algorithm for training neural networks:

```
repeat until converged:
    gradient = ∇Loss(current_weights)
    weights = weights - learning_rate * gradient
```

Take a step in the direction of *steepest descent* (negative gradient). The **learning rate** `α` (alpha) controls how big each step is.

**Backend analogy:** This is like automatically tuning your server configuration to minimize error rate. Every few minutes, you:
1. Measure the current error rate (compute the loss)
2. Try slightly increasing each config parameter and see what happens (compute the gradient)
3. Move all parameters in the direction that reduces error rate

Repeat until you've found a good configuration. The learning rate is how aggressively you change parameters each iteration — too large and you overshoot, too small and it takes forever.

### Convergence

When the gradient is zero, you've found a **minimum** (or maximum, or saddle point). In practice for deep learning, we almost always find good-enough minima — perfect global minima aren't required.

### Learning Rate Sensitivity

| Learning rate | Effect |
|---|---|
| Too large | Oscillates or diverges — like overcorrecting a PID controller |
| Too small | Converges, but very slowly — wastes compute |
| Just right | Converges smoothly to a good minimum |

---

## 6. The Chain Rule — Derivatives Through Nested Functions

Most ML functions are deeply nested:

```
Loss = cross_entropy( softmax( W2 @ relu( W1 @ x + b1 ) + b2 ) )
```

To compute the gradient of `Loss` with respect to `W1`, you need the **chain rule**:

```
d/dx [ f(g(x)) ] = f'(g(x)) * g'(x)
```

Or in words: "derivative of the outer function (evaluated at the inner result) times the derivative of the inner function."

**Backend analogy:** Think of it as tracing through a call stack to find the root cause of a bug. If `service_A` calls `service_B` which calls `service_C`, and latency spikes in `C`, the chain rule tells you how that spike propagates back through `B` and `A`. The total latency sensitivity to `C`'s slowdown is the product of the sensitivities at each layer.

### Example: `h(x) = sin(x^2)`

```
Let g(x) = x^2      → g'(x) = 2x
Let f(u) = sin(u)   → f'(u) = cos(u)

h(x)  = f(g(x)) = sin(x^2)
h'(x) = f'(g(x)) * g'(x)
       = cos(x^2) * 2x
```

### Backpropagation = Chain Rule Applied Systematically

Neural network training uses **backpropagation**: systematically apply the chain rule from the loss backward through every layer to compute `∂Loss/∂W` for every weight matrix `W`. Then update each weight with gradient descent.

This is why deep learning frameworks like PyTorch and TensorFlow build "computation graphs" — they need to traverse the graph backward to apply the chain rule at each node.

---

## 7. Common Pitfalls

**Vanishing gradients:** Deep in a network, the chain rule multiplies many small numbers together. If each layer's gradient is < 1, the product shrinks exponentially. By the time you reach early layers, the gradient is effectively zero and those layers stop learning. This was a major problem before techniques like residual connections and layer normalization.

**Exploding gradients:** The opposite — multiplying large numbers produces enormous gradients, causing parameter updates to blow up. Solved with gradient clipping.

**Local minima vs. saddle points:** In high dimensions, most "stuck" points are saddle points (minimum in some directions, maximum in others), not true local minima. This is actually fine — gradient descent tends to escape saddle points.

---

## 8. Summary Table

| Concept | Definition | Backend Analogy |
|---|---|---|
| Derivative `f'(x)` | Rate of change of f at x | Response time sensitivity to load |
| Partial derivative `∂f/∂xᵢ` | Rate of change varying only xᵢ | Profiling one service in a microservice system |
| Gradient `∇f` | Vector of all partial derivatives | Multi-dimensional sensitivity report |
| Gradient descent | Repeatedly step in `-∇f` direction | Auto-tuning config to minimize error rate |
| Learning rate `α` | Step size for gradient descent | How aggressively to adjust config each iteration |
| Chain rule | `(f∘g)' = f'(g) * g'` | Tracing latency propagation through a call stack |
| Backpropagation | Chain rule applied to neural nets | Root cause analysis through a call stack |

---

## 9. How This Appears in LLMs

- **Training:** Compute loss (cross-entropy) on each batch, backpropagate to get gradients for all ~billions of parameters, update with Adam optimizer (an advanced gradient descent variant)
- **Adam optimizer:** Keeps a running estimate of the gradient and its square — like gradient descent with adaptive learning rates per parameter
- **Learning rate schedules:** Warm up slowly, then decay — like a PID controller with a ramp-up phase
- **Gradient checkpointing:** Trade compute for memory — recompute intermediate activations during backprop instead of storing all of them

---

## Further Reading

- [3Blue1Brown: What is a derivative?](https://www.youtube.com/watch?v=9vKqVkMQHKk) — best visual intuition
- [3Blue1Brown: Backpropagation](https://www.youtube.com/watch?v=Ilg3gGewQ5U) — gradient descent through a neural net
- Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd) — a complete autograd engine in ~150 lines
