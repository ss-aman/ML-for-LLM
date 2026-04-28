# 05 — Backpropagation: Training a Network Step by Step

## What Is Backpropagation?

Backpropagation (backprop) is the algorithm that computes the gradient
of the loss with respect to **every parameter** in the network.

It's not a new idea — it's the chain rule, applied systematically starting
from the loss and working backward through each layer.

The key workflow is:

```
Forward pass:  input → layer 1 → layer 2 → ... → loss
               (compute outputs; STORE intermediate values)

Backward pass: loss → layer N → ... → layer 1
               (compute gradients using stored intermediate values)
```

Then: `w = w - lr * gradient`

---

## The Network We'll Use

A 2-layer neural network:

```
Layer 1: z₁ = W₁ @ x + b₁         (linear)
         a₁ = relu(z₁)              (activation)
Layer 2: z₂ = W₂ @ a₁ + b₂         (linear)
Output:  ŷ = z₂                     (prediction)
Loss:    L = MSE(ŷ, y) = mean((ŷ - y)²)
```

We want gradients: `∂L/∂W₁`, `∂L/∂b₁`, `∂L/∂W₂`, `∂L/∂b₂`.

---

## Forward Pass (with careful bookkeeping)

During the forward pass you compute the output AND store intermediate values
needed for backprop.

```python
# Input
x  = np.array([1.0, 2.0])    # 2D input
y  = np.array([1.0])          # target

# Layer 1
z1 = W1 @ x + b1             # linear: store z1
a1 = relu(z1)                 # activation: store a1

# Layer 2
z2 = W2 @ a1 + b2            # linear: store z2
y_hat = z2                    # prediction

# Loss
diff = y_hat - y
L = np.mean(diff ** 2)        # MSE loss
```

---

## Backward Pass: Chain Rule Layer by Layer

Now work backward. At each step, compute the gradient using the chain rule
and pass it to the next layer.

### Step 1: Gradient of loss

```
L = mean((ŷ - y)²)
∂L/∂ŷ = 2 * (ŷ - y) / N
```

Let `dL = ∂L/∂ŷ` — this is the "incoming gradient" that starts the backward pass.

### Step 2: Gradient through Layer 2 (linear: `z₂ = W₂ @ a₁ + b₂`)

For a linear layer `z = Wx + b`, the gradients are:

```
∂L/∂W₂ = dL · a₁ᵀ        (outer product)
∂L/∂b₂ = dL               (bias gradient = incoming gradient)
∂L/∂a₁ = W₂ᵀ · dL        (pass gradient backward to a₁)
```

In code:
```python
dL_dW2 = dL[:, None] @ a1[None, :]   # outer product: (out_dim, in_dim)
dL_db2 = dL
dL_da1 = W2.T @ dL                    # pass backward
```

### Step 3: Gradient through ReLU (`a₁ = relu(z₁)`)

```
relu(z) = max(0, z)
relu'(z) = 1 if z > 0 else 0
```

```python
dL_dz1 = dL_da1 * (z1 > 0).astype(float)   # zero where relu was inactive
```

This is the "local gradient" of ReLU: pass the incoming gradient through
where the neuron was active, block it where it was inactive.

### Step 4: Gradient through Layer 1 (linear: `z₁ = W₁ @ x + b₁`)

Same formula as Layer 2:

```python
dL_dW1 = dL_dz1[:, None] @ x[None, :]   # outer product
dL_db1 = dL_dz1
# (no need to compute dL_dx unless we're also learning x)
```

---

## Why This Works: The Computation Graph

Every operation in the forward pass creates a **node** in a computation graph.
Every node has:
- A **forward function**: compute output from inputs
- A **backward function**: given the incoming gradient, compute and return
  the gradient for each input

```
x ──→ [W₁@x+b₁] ──→ [relu] ──→ [W₂@a₁+b₂] ──→ [MSE] ──→ L
      ↑ store z1  ↑ store a1    ↑ store z2
      ↓ ∂L/∂W₁   ↓ ∂L/∂z1     ↓ ∂L/∂W₂   ↓ ∂L/∂ŷ
                backward pass (right to left)
```

This is exactly what PyTorch's `autograd` builds when you write forward-pass
code. Every operation (`@`, `+`, `relu`) registers a backward function.
Calling `.backward()` traverses the graph right-to-left and accumulates
gradients.

---

## The Gradient Check: How to Verify Backprop Is Correct

After implementing backprop manually, verify it against numerical gradients:

```python
# For each parameter w:
w_original = w.copy()
w[i] += h
loss_plus = forward(w)
w[i] -= 2*h
loss_minus = forward(w)
w[i] += h  # restore

numerical_grad = (loss_plus - loss_minus) / (2 * h)
analytical_grad = backprop(...)  # your implementation

assert abs(numerical_grad - analytical_grad) < 1e-5
```

If they match: your backprop is correct.
If they don't: there's a bug in your chain rule application.

This is called a **gradient check** and is standard practice when
implementing backprop from scratch.

---

## Vanishing Gradients in Deep Networks

In a network with 20 layers, the gradient of the loss w.r.t. the first
layer passes through 19 multiplications.

If each layer's gradient has magnitude 0.5:
```
Layer 20: gradient magnitude ≈ 1.0
Layer 10: 0.5^10 ≈ 0.001
Layer 1:  0.5^19 ≈ 0.000002
```

The first layer receives almost no gradient signal and can't learn.

**How LLMs solve this — residual connections:**

```python
# Without residual:
x = layer(x)    # gradient must flow entirely through layer

# With residual:
x = x + layer(x)    # gradient also flows through the identity path
```

Gradient through a residual block:
```
∂L/∂x = ∂L/∂(x + layer(x))
       = ∂L/∂output · (1 + ∂layer(x)/∂x)
```

The `1 +` term means even if `∂layer(x)/∂x` vanishes, the gradient still
flows back unattenuated through the identity path. This is why transformers
with 96 layers (GPT-3) can be trained — residuals make the gradient highway.

---

## What Frameworks Do For You

When you write PyTorch code:

```python
# Forward pass (PyTorch tracks this automatically)
z1 = x @ W1.T + b1
a1 = F.relu(z1)
z2 = a1 @ W2.T + b2
loss = F.mse_loss(z2, y)

# Backward pass (one line!)
loss.backward()

# Gradients are now in:
W1.grad   # ∂loss/∂W1
b1.grad   # ∂loss/∂b1
W2.grad   # ∂loss/∂W2
b2.grad   # ∂loss/∂b2
```

PyTorch builds the computation graph during the forward pass and traverses
it backward when you call `.backward()`. It does exactly what we computed
manually above — just for any arbitrary network, not just 2 layers.

---

## Key Takeaway

> Backpropagation computes the gradient for every parameter by applying
> the chain rule backwards through the computation graph. The forward pass
> computes the loss; the backward pass computes who is responsible for it.
> Then gradient descent updates every parameter to reduce its contribution
> to the loss.

---

## What's Next

`06_how_gradients_train_llms.md`: the complete LLM training loop — putting
together loss, backprop, and the Adam optimizer used in every real model.
