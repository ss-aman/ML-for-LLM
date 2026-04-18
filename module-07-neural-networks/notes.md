# Module 07: Neural Networks

## Why This Matters

Neural networks are the core building block of every LLM. Before you can understand transformers, attention, or fine-tuning, you need to understand what a neural network is, why it can learn, and how it updates itself from feedback. This module builds that foundation — no magic, just math and analogies you already know.

---

## 1. What Is a Neural Network?

A neural network is a **pipeline of linear transformations followed by nonlinearities**.

Each layer does:
1. A linear transform: multiply by a weight matrix, add a bias
2. A nonlinearity (activation function): squash or clip the result

Stack several such layers and you have a neural network.

> **Backend analogy:** Think of a neural network as a chain of middleware functions in a web server. Each middleware receives a request object, transforms it, and passes it along. The difference: in a neural network, every middleware has **learnable parameters** — the weights — that automatically tune themselves so the final output is as correct as possible.

```
Input → [Linear + ReLU] → [Linear + ReLU] → [Linear] → Output
          Layer 1             Layer 2          Layer 3
```

This is essentially the same as:

```python
def neural_net(x):
    x = relu(W1 @ x + b1)   # middleware 1
    x = relu(W2 @ x + b2)   # middleware 2
    x = W3 @ x + b3          # output layer (no activation)
    return x
```

A neural network is just a **function with millions of tuneable parameters**. Training is the process of automatically finding the parameter values that make the function accurate.

---

## 2. The Neuron — The Smallest Unit

A single **neuron** computes:

```
output = activation(w · x + b)
```

Where:
- `x` is the input vector (a row of data or the previous layer's output)
- `w` is the **weight vector** — a learned set of importances for each input
- `b` is the **bias** — a learned offset, like a constant in a linear equation
- `w · x` is the dot product (weighted sum of inputs)
- `activation(...)` is the nonlinearity applied to that sum

> **Backend analogy:** A neuron is like a single decision rule in a business logic function:
> ```python
> def approve_loan(income, debt, credit_score):
>     score = 0.4 * income + (-0.3) * debt + 0.6 * credit_score + (-100)
>     return sigmoid(score)  # → probability between 0 and 1
> ```
> The weights `[0.4, -0.3, 0.6]` and bias `-100` are what gets learned during training.

A **layer** is just many neurons operating in parallel on the same input, each with its own weight vector. If layer 1 has 64 neurons, it produces a 64-dimensional output vector.

---

## 3. Activation Functions — The Nonlinearities

Without activation functions, stacking linear layers just produces another linear layer (they collapse). Activations break the linearity so the network can learn complex patterns.

### ReLU (Rectified Linear Unit) — The Default

```
ReLU(x) = max(0, x)
```

- Negative values → 0 (the neuron "doesn't fire")
- Positive values → pass through unchanged
- Derivative: 0 for x < 0, 1 for x > 0 (simple!)
- **Why it works:** Fast to compute, doesn't saturate for positive inputs, sparse activations (many zeros)

> **Backend analogy:** ReLU is like a floor threshold in rate limiting — anything below zero gets clamped to zero, everything above passes through unchanged.

### Sigmoid — Squash to (0, 1)

```
sigmoid(x) = 1 / (1 + e^(-x))
```

- Output is always between 0 and 1 → perfect for probabilities
- Saturates (gradient ≈ 0) for very large or very small inputs → can cause vanishing gradients
- Used in binary classification output layers and LSTM gates

> **Backend analogy:** Sigmoid is like a soft on/off switch: at extreme inputs it's fully on (≈1) or fully off (≈0), but in the middle it's a smooth probability.

### Tanh — Squash to (-1, 1)

```
tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
```

- Output is always between -1 and +1 → zero-centered (unlike sigmoid)
- Used in RNN hidden states and LSTM cell gates

> **Backend analogy:** Tanh is like sigmoid but rescaled so "neutral" is 0 instead of 0.5. Useful when negative and positive signals carry different meanings.

### Summary Table

| Activation | Range | Use Case | Gradient Problem |
|---|---|---|---|
| ReLU | [0, ∞) | Hidden layers (default) | "Dying ReLU" for very negative inputs |
| Sigmoid | (0, 1) | Binary output, gates | Vanishing gradient for large \|x\| |
| Tanh | (-1, 1) | RNNs, LSTMs | Vanishing gradient (less than sigmoid) |

---

## 4. Layers — Depth Means Abstraction

A **layer** is a matrix multiplication + bias + activation applied to an entire input vector at once.

- **Input layer**: the raw data (e.g., pixel values, token embeddings)
- **Hidden layers**: intermediate transformations that build up increasingly abstract features
- **Output layer**: the final prediction (class probabilities, a continuous value, etc.)

> **Backend analogy:** Think of layers like stages in a data transformation pipeline:
> - Layer 1 (close to input) detects low-level patterns — like parsing raw HTTP bytes into a request struct
> - Layer 2 detects mid-level patterns — like extracting route, headers, body from the struct
> - Layer 3 (close to output) makes the final decision — like routing to the correct handler

**Depth = more abstraction.** A 10-layer network can learn features that a 2-layer network cannot — it builds up hierarchical representations of the input.

---

## 5. Forward Pass — Data Flows Input → Output

The **forward pass** is the process of running input data through every layer in order to produce a prediction.

```python
def forward(x, layers):
    for layer in layers:
        x = layer(x)  # each layer transforms x
    return x           # final output = prediction
```

> **Backend analogy:** The forward pass is exactly a web request flowing through a middleware chain. Input = HTTP request object. Each middleware transforms it. Output = response. Nothing is learned here — this is pure inference.

---

## 6. Loss Function — Measuring How Wrong You Are

Before the network can learn, we need to measure how wrong its predictions are. This is the **loss function**.

- For regression (predicting a number): **Mean Squared Error** = mean of (prediction - truth)²
- For classification (predicting a class): **Cross-Entropy Loss** = -log(probability assigned to the correct class)

> **Backend analogy:** Loss is like an SLA violation score. If your API was supposed to respond in 100ms but responded in 500ms, the violation is 400ms. The loss function quantifies how far you are from your target — and the goal of training is to drive it toward zero.

---

## 7. Backpropagation — Gradients Flow Output → Input

**Backpropagation** is the algorithm for computing how much each weight contributed to the error. It flows gradients from the output layer back through the network using the **chain rule** of calculus.

The chain rule says: if `y = f(g(x))`, then `dy/dx = (dy/dg) * (dg/dx)`. For a chain of layers:

```
Loss ← Layer 3 ← Layer 2 ← Layer 1 ← Input
  ↓ grad flows backward through chain rule ↓
dLoss/dW1 = dLoss/dout3 * dout3/dout2 * dout2/dout1 * dout1/dW1
```

> **Backend analogy:** Backprop is like **distributed tracing in reverse**. When a request fails (high loss), distributed tracing tells you which service in the chain caused the problem and by how much. Backprop does the same thing: given a wrong prediction, it traces back through each layer to figure out which weights are responsible and by how much. The gradient tells you the "blame" assigned to each parameter.

---

## 8. Weight Update — Learning from Mistakes

Once you have the gradient for each weight, you update it using **gradient descent**:

```
weight = weight - learning_rate * gradient
```

- The **gradient** points in the direction that increases the loss
- We subtract it to move in the direction that decreases the loss
- The **learning rate** (a small number like 0.01) controls the step size

> **Backend analogy:** Think of it like tuning a load balancer. If requests are consistently timing out (high loss), you adjust your timeout thresholds (weights) proportionally to how much each one contributes to the timeouts (gradient). Small adjustments each time prevent overcorrection.

After each weight update, you run the forward pass again, compute the new loss, and repeat. Over many iterations (epochs), the loss decreases and the network gets better at its task.

---

## 9. Putting It All Together

```
1. Forward pass:   x → Layer1 → Layer2 → prediction
2. Compute loss:   loss = cross_entropy(prediction, true_label)
3. Backprop:       compute dLoss/dW for every weight W
4. Update weights: W = W - lr * dLoss/dW
5. Repeat
```

This loop is **training**. Each full pass through the training dataset is an **epoch**.

---

## 10. Why Neural Networks Work

- **Universal approximation theorem:** A neural network with enough hidden units can approximate any continuous function. This is the theoretical basis for why they can learn almost anything.
- **Deep networks learn hierarchical features:** Early layers learn simple patterns; later layers combine them into complex ones.
- **Gradient descent finds good parameters:** With enough data and the right architecture, the optimization landscape has many good solutions.

> **Backend analogy:** The universal approximation theorem is like saying: given enough middleware, you can route any HTTP request to the correct handler. The network just needs to find the right chain of transformations.

---

## 11. How This Connects to LLMs

Every transformer layer contains two neural network components:
1. **Attention** (computes interactions between tokens — next modules)
2. **Feed-forward network** (a 2-layer neural network applied to each token independently)

The feed-forward layer in a transformer is literally: `output = W2 @ relu(W1 @ x + b1) + b2`. This is the exact 2-layer network you'll build in this module's code.

Understanding forward pass + backprop is the foundation for understanding how transformers are trained, how fine-tuning works, and why certain optimization choices (like layer norm placement) matter.
