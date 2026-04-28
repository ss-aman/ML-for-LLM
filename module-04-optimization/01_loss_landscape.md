# 01 — The Loss Landscape

## What we're actually doing

When you train a neural network, you're searching for a specific set of numbers (the weights) that make the model work well. A GPT-2-small model has **117 million** such numbers. GPT-3 has **175 billion**.

Optimization is the process of finding the right values for all of them.

---

## The loss function: what we're minimizing

A **loss function** takes the model's current weights and returns a single number: how wrong the model is right now.

```
L(weights) → scalar   (lower is better)
```

For LLMs, this is cross-entropy loss (Module 03):
```
L = -(1/T) Σ log P(correct_token_at_position_t)
```

High loss = model assigns low probability to the right answers.  
Low loss = model assigns high probability to the right answers.

**Backend analogy:** The loss function is like your p99 latency metric. It collapses the health of the entire system (all servers, all routes, all configs) into a single number. You're trying to minimize that number by adjusting the system configuration.

---

## The loss landscape: a very high-dimensional terrain

Imagine a map of all possible weight configurations. Each point on the map is a specific setting for all the weights. The **height** at each point is the loss value there.

```
2D example:
  w1 axis: weight 1 (runs left-right)
  w2 axis: weight 2 (runs front-back)
  height:  loss value at (w1, w2)

In reality: 175 billion axes for GPT-3
```

Training = finding the lowest point on this terrain.

```
High point (hill) = bad model, high loss
Low point (valley) = good model, low loss
Gradient at any point = which direction goes uphill fastest
```

---

## Types of critical points

### Global minimum
The absolute lowest point in the entire landscape. The theoretically best possible solution.

For neural networks: almost never found. The landscape is too complex and the global minimum may not even be well-defined (there are typically many equally good solutions due to symmetry).

### Local minimum
A valley that isn't the deepest one — a point where every direction goes up, but it's not the global minimum.

**For neural networks in practice:** Local minima are rarely the real problem. Research shows that for overparameterized networks (like LLMs), most local minima are about equally good. The loss is similar across different local minima.

### Saddle points
Points where the gradient is zero but it's not a minimum — some directions go up, some go down. In high-dimensional spaces, saddle points are far more common than local minima.

```
f(x, y) = x² - y²
Gradient at (0,0) = [0, 0]  ← looks like a minimum
But: x direction is a minimum, y direction is a maximum
```

**The real problem in deep learning:** Getting stuck at saddle points, not local minima. When gradients are near zero, learning stalls. Momentum and the noise in SGD help escape saddle points.

### Flat regions (plateaus)
Large areas where the gradient is nearly zero everywhere. The gradient provides no useful signal.

**In LLMs:** Early training often involves long plateaus before loss suddenly drops. The optimizer needs to find the right direction across a flat region.

---

## What makes neural network loss landscapes special

### They are non-convex

A **convex** function has exactly one minimum (like a bowl). Gradient descent on a convex function always finds the global minimum.

Neural network loss functions are **non-convex** — many valleys, ridges, saddle points.

```
Convex (linear regression):    ∪  ← one clear minimum, GD always works
Non-convex (neural network):   ∿  ← complex, multiple local optima
```

Yet gradient descent on neural networks works extremely well in practice. Why? Because:
1. Overparameterized networks have many paths to good solutions
2. Most local minima in deep networks have similar loss values
3. SGD noise helps explore the landscape

### They have many symmetries

A neural network with 2 hidden neurons has (at minimum) 2 equally good solutions: just swap neuron 1 and neuron 2. With 1000 hidden neurons: 1000! equally good configurations.

This means the loss landscape is symmetric in high dimensions — there are vast numbers of equivalent solutions.

### Sharp vs. flat minima

Research (Hochreiter & Schmidhuber 1997, Keskar et al. 2017) shows:
- **Sharp minimum:** High curvature — small weight changes cause large loss increases. Model won't generalize well.
- **Flat minimum:** Low curvature — small weight changes don't hurt much. Better generalization.

SGD with small batches tends to find flatter minima (better generalization) because the gradient noise prevents settling in sharp, narrow valleys. Large batch training tends to find sharper minima.

**Backend analogy:** A config setting that's very sensitive to small changes (sharp minimum) is fragile. A config that still works well even with slight perturbations (flat minimum) is robust and generalizes better.

---

## The loss landscape of an LLM

For a 7B-parameter model trained on web text:

```
Early training (step 1):
  - Loss ≈ 11 (random initialization: model guesses uniformly over 50k tokens)
  - log(50000) ≈ 10.8  ← that's what random looks like

After 1000 steps:
  - Loss ≈ 4-5 (model learned basic patterns)

After 100k steps:
  - Loss ≈ 2.5-3 (model learned language)

After 1M steps:
  - Loss ≈ 2.0-2.3 (model is quite good)

Theoretical minimum (true entropy of language):
  - ≈ 1.5-1.8 nats (can never go below this)
```

The landscape is navigable — steady progress is possible with the right optimizer.

---

## Why vanilla gradient descent isn't enough

Gradient descent is the core idea: walk downhill. But several problems arise in practice:

| Problem | Cause | Solution |
|---------|-------|---------|
| Slow convergence | Too-small steps | Momentum |
| Oscillation in narrow valleys | Gradient direction is wrong | Momentum |
| Parameters need different step sizes | Some dimensions flat, some steep | Adaptive rates (Adam) |
| Training instability | Occasional huge gradients | Gradient clipping |
| Slow start | Large learning rate at step 0 breaks things | LR warmup |
| Overfitting | Model memorizes training data | Weight decay |

Each of the next files covers one of these solutions.

---

## Visualizing the landscape

```
Loss contour of a simple 2D problem:

f(w1, w2) = w1² + 10·w2²   (elongated bowl)

Contours:
  . . . . . . . . . . . . .
  . . . ○ ○ ○ ○ ○ ○ . . . .
  . . ○ ○ ● ● ● ○ ○ . . . .
  . ○ ○ ● ● ★ ● ● ○ . . . .   ★ = minimum at (0,0)
  . . ○ ○ ● ● ● ○ ○ . . . .
  . . . ○ ○ ○ ○ ○ ○ . . . .
  . . . . . . . . . . . . .

Gradient descent on this:
  - Takes large steps in w2 direction (steep)
  - Takes small steps in w1 direction (flat)
  - Oscillates back and forth in w2 before converging
  - Adam fixes this: adapts step size per dimension
```

---

## Summary

| Concept | What it means |
|---------|---------------|
| Loss function | Single number measuring model error |
| Loss landscape | Map of loss over all possible weight values |
| Gradient | Direction of steepest increase at current point |
| Minimum | Point where gradient = 0 and loss is locally smallest |
| Saddle point | Gradient = 0 but not a minimum (common in deep networks) |
| Flat region | Gradient ≈ 0 everywhere — training stalls |
| Non-convex | Multiple valleys; gradient descent not guaranteed to find global min |
| Flat minimum | Low curvature — better generalization |

Next: **SGD and mini-batches** — how we actually take steps through this landscape.
