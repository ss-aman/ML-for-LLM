"""
Module 07 — Exercises: Neural Networks from Scratch
====================================================
Work through these exercises to cement your understanding of:
  1. Activation functions and their derivatives
  2. A 1-layer network on linearly separable data
  3. (Challenge) A 3-layer network with manual backprop on non-linear data

Run with:  python exercise.py
"""

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Exercise 1: Implement activation functions + their derivatives
# ---------------------------------------------------------------------------

def exercise_1_activations():
    """
    Implement sigmoid, ReLU, and their derivatives.

    Why this matters:
      - Every backward pass multiplies the upstream gradient by the local
        derivative of the activation.
      - Getting these wrong silently breaks training (loss stops decreasing).

    Backend analogy: These are the local transfer functions at each "middleware"
    node. The derivative tells us how much a small change at the node's input
    affects its output — essential for the chain-rule blame-propagation.

    Tasks:
      a) Implement sigmoid(x) = 1 / (1 + exp(-x))
      b) Implement sigmoid_derivative(x) — in terms of sigmoid(x)
      c) Implement relu(x) = max(0, x)
      d) Implement relu_derivative(x) — element-wise, returns 0 or 1
      e) Print outputs for x in [-2, -1, 0, 1, 2] and verify visually
    """
    print("=" * 55)
    print("Exercise 1: Activation Functions & Derivatives")
    print("=" * 55)

    # --- YOUR CODE HERE ---

    def sigmoid(x: np.ndarray) -> np.ndarray:
        # TODO: implement  1 / (1 + exp(-x))
        raise NotImplementedError("Implement sigmoid")

    def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
        # TODO: implement  sigmoid(x) * (1 - sigmoid(x))
        raise NotImplementedError("Implement sigmoid_derivative")

    def relu(x: np.ndarray) -> np.ndarray:
        # TODO: implement  max(0, x) element-wise
        raise NotImplementedError("Implement relu")

    def relu_derivative(x: np.ndarray) -> np.ndarray:
        # TODO: implement  1 where x > 0, else 0
        raise NotImplementedError("Implement relu_derivative")

    # --- END YOUR CODE ---

    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    print(f"\nx values:            {x}")
    print(f"sigmoid(x):          {sigmoid(x).round(4)}")
    print(f"sigmoid_deriv(x):    {sigmoid_derivative(x).round(4)}")
    print(f"relu(x):             {relu(x).round(4)}")
    print(f"relu_deriv(x):       {relu_derivative(x).round(4)}")

    # Sanity checks
    expected_sigmoid = np.array([0.1192, 0.2689, 0.5, 0.7311, 0.8808])
    assert np.allclose(sigmoid(x), expected_sigmoid, atol=1e-3), \
        f"sigmoid values wrong: got {sigmoid(x).round(4)}"

    expected_relu = np.array([0.0, 0.0, 0.0, 1.0, 2.0])
    assert np.allclose(relu(x), expected_relu), \
        f"relu values wrong: got {relu(x)}"

    print("\nAll checks passed!\n")

    # Plot both activations and their derivatives
    x_plot = np.linspace(-4, 4, 200)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(x_plot, sigmoid(x_plot), label="sigmoid", color='steelblue')
    axes[0].plot(x_plot, sigmoid_derivative(x_plot), label="d/dx sigmoid",
                 color='steelblue', linestyle='--')
    axes[0].set_title("Sigmoid and its derivative")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x_plot, relu(x_plot), label="ReLU", color='tomato')
    axes[1].plot(x_plot, relu_derivative(x_plot), label="d/dx ReLU",
                 color='tomato', linestyle='--')
    axes[1].set_title("ReLU and its derivative")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("ex1-activations.png", dpi=100)
    plt.show()


# ---------------------------------------------------------------------------
# Exercise 2: 1-layer network on linearly separable data
# ---------------------------------------------------------------------------

def exercise_2_linear_classifier():
    """
    Build and train a 1-layer binary classifier on a simple rule:
        x > 0  →  class 1
        x ≤ 0  →  class 0

    This is linearly separable — a single weight and bias can draw the
    decision boundary at x = 0.

    Architecture:
        Input(1) → Linear(1→1) → Sigmoid → output probability

    Tasks:
      a) Implement the forward pass (linear + sigmoid)
      b) Implement BCE loss and its gradient w.r.t. the prediction
      c) Implement the backward pass (gradients for w and b)
      d) Run the training loop for 1000 epochs, print loss every 100 epochs
      e) After training, verify: predict(1.0) ≈ 1, predict(-1.0) ≈ 0

    Backend analogy: This is equivalent to logistic regression — a single
    decision rule with a learned threshold. Same as a rule engine that tunes
    its own thresholds based on observed outcomes.
    """
    print("=" * 55)
    print("Exercise 2: 1-Layer Linear Classifier")
    print("=" * 55)

    np.random.seed(42)
    # Generate data: 100 samples drawn from N(0,1), label = (x > 0)
    X = np.random.randn(1, 200)           # shape (1, 200)
    y = (X > 0).astype(float)             # shape (1, 200)

    # --- YOUR CODE HERE ---

    # Initialize weights
    w = np.random.randn(1, 1) * 0.01     # shape (1, 1)
    b = np.zeros((1, 1))                  # shape (1, 1)
    lr = 0.1
    losses = []

    def sigmoid(x):
        # TODO: implement
        raise NotImplementedError

    for epoch in range(1000):
        # TODO: Forward pass
        #   z = w @ X + b          → shape (1, 200)
        #   pred = sigmoid(z)      → shape (1, 200)

        # TODO: BCE loss and gradient
        #   loss = -mean(y*log(pred) + (1-y)*log(1-pred))
        #   d_pred = -(y/pred - (1-y)/(1-pred)) / batch_size

        # TODO: Backward pass
        #   d_z = pred * (1 - pred) * d_pred   ← sigmoid derivative already applied
        #   dw = (d_z @ X.T) / batch_size
        #   db = mean(d_z)

        # TODO: Weight update
        #   w -= lr * dw
        #   b -= lr * db

        raise NotImplementedError("Implement the training loop")

    # --- END YOUR CODE ---

    print(f"\nFinal loss: {losses[-1]:.4f}")
    test_pos = sigmoid(w @ np.array([[1.0]]) + b)
    test_neg = sigmoid(w @ np.array([[-1.0]]) + b)
    print(f"predict(+1.0) = {test_pos.item():.3f}  (should be close to 1.0)")
    print(f"predict(-1.0) = {test_neg.item():.3f}  (should be close to 0.0)")
    assert test_pos.item() > 0.8, "Network should confidently predict 1 for x=+1"
    assert test_neg.item() < 0.2, "Network should confidently predict 0 for x=-1"
    print("Checks passed!\n")


# ---------------------------------------------------------------------------
# Exercise 3 (Challenge): 3-layer network with manual backprop
# ---------------------------------------------------------------------------

def exercise_3_challenge_three_layer_network():
    """
    Build a 3-layer network from scratch and train it on a non-linear
    classification problem: classify whether a 2D point is inside a circle
    of radius 1 centred at the origin.

        inside circle (x^2 + y^2 < 1)  →  class 1
        outside circle                  →  class 0

    A single linear layer cannot solve this; you need at least 2 hidden layers
    to learn the circular decision boundary.

    Architecture:
        Input(2) → Linear(2→8) → ReLU → Linear(8→4) → ReLU → Linear(4→1) → Sigmoid

    Tasks:
      a) Generate 500 random 2D points, label them by inside/outside circle
      b) Initialise three weight matrices (W1, b1), (W2, b2), (W3, b3)
      c) Implement the full forward pass
      d) Implement BCE loss
      e) Implement manual backprop through all three layers
      f) Train for 5000 epochs, print loss every 500 epochs
      g) Achieve > 90% accuracy on the training set
      h) (Bonus) Plot the learned decision boundary

    Hint: The chain rule unwinds in strict reverse order:
        dLoss → d_sigmoid → d_W3/db3/d_a2 → d_relu2 → d_W2/db2/d_a1 → d_relu1 → d_W1/db1

    Backend analogy: Each layer is a middleware. The backward pass is like
    re-running the distributed trace in reverse, attributing error to each
    service proportionally to its contribution.
    """
    print("=" * 55)
    print("Exercise 3 (Challenge): 3-Layer Network on Circle Data")
    print("=" * 55)

    np.random.seed(0)
    n = 500
    # Generate points uniformly in [-1.5, 1.5]^2
    X = np.random.uniform(-1.5, 1.5, (2, n))
    y = ((X[0] ** 2 + X[1] ** 2) < 1.0).astype(float).reshape(1, n)
    print(f"Dataset: {n} points, {int(y.sum())} inside circle, {n - int(y.sum())} outside\n")

    # --- YOUR CODE HERE ---

    # Initialise weights using He initialisation: W ~ N(0, sqrt(2/fan_in))
    rng = np.random.default_rng(42)

    # Layer 1: (2 → 8)
    W1 = rng.standard_normal((8, 2)) * np.sqrt(2.0 / 2)
    b1 = np.zeros((8, 1))

    # Layer 2: (8 → 4)
    W2 = rng.standard_normal((4, 8)) * np.sqrt(2.0 / 8)
    b2 = np.zeros((4, 1))

    # Layer 3: (4 → 1)
    W3 = rng.standard_normal((1, 4)) * np.sqrt(2.0 / 4)
    b3 = np.zeros((1, 1))

    lr = 0.05
    losses = []

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    def relu(x):
        return np.maximum(0, x)

    for epoch in range(5000):
        # TODO: Forward pass
        # z1 = W1 @ X + b1         → shape (8, n)
        # a1 = relu(z1)
        # z2 = W2 @ a1 + b2        → shape (4, n)
        # a2 = relu(z2)
        # z3 = W3 @ a2 + b3        → shape (1, n)
        # pred = sigmoid(z3)

        # TODO: BCE loss
        # eps = 1e-9
        # loss = -mean(y * log(pred+eps) + (1-y) * log(1-pred+eps))

        # TODO: Backward pass (chain rule in reverse)
        # d_pred = -(y/(pred+eps) - (1-y)/(1-pred+eps)) / n
        # d_z3   = d_pred * pred * (1-pred)    ← sigmoid gradient
        # dW3    = d_z3 @ a2.T / n
        # db3    = mean(d_z3, axis=1, keepdims=True)
        # d_a2   = W3.T @ d_z3
        # d_z2   = d_a2 * (z2 > 0)            ← ReLU gradient
        # dW2    = d_z2 @ a1.T / n
        # db2    = mean(d_z2, axis=1, keepdims=True)
        # d_a1   = W2.T @ d_z2
        # d_z1   = d_a1 * (z1 > 0)            ← ReLU gradient
        # dW1    = d_z1 @ X.T / n
        # db1    = mean(d_z1, axis=1, keepdims=True)

        # TODO: Weight updates
        # W1 -= lr * dW1  etc.

        raise NotImplementedError("Implement the 3-layer training loop")

    # --- END YOUR CODE ---

    # Evaluate accuracy
    pred_final = sigmoid(W3 @ relu(W2 @ relu(W1 @ X + b1) + b2) + b3)
    accuracy = np.mean((pred_final > 0.5) == y)
    print(f"Final loss:    {losses[-1]:.4f}")
    print(f"Train accuracy: {accuracy * 100:.1f}%")
    assert accuracy > 0.90, f"Expected > 90% accuracy, got {accuracy*100:.1f}%"
    print("Challenge passed!\n")

    # Bonus: plot decision boundary
    xx, yy = np.meshgrid(np.linspace(-1.5, 1.5, 100), np.linspace(-1.5, 1.5, 100))
    grid = np.vstack([xx.ravel(), yy.ravel()])
    z = sigmoid(W3 @ relu(W2 @ relu(W1 @ grid + b1) + b2) + b3).reshape(100, 100)

    plt.figure(figsize=(6, 6))
    plt.contourf(xx, yy, z, levels=50, cmap='RdBu', alpha=0.7)
    plt.scatter(X[0, y[0] == 1], X[1, y[0] == 1], c='blue', s=10, label='inside')
    plt.scatter(X[0, y[0] == 0], X[1, y[0] == 0], c='red', s=10, label='outside')
    theta = np.linspace(0, 2 * np.pi, 200)
    plt.plot(np.cos(theta), np.sin(theta), 'k--', linewidth=2, label='true boundary')
    plt.legend()
    plt.title("3-Layer Network: Learned Decision Boundary")
    plt.tight_layout()
    plt.savefig("ex3-decision-boundary.png", dpi=100)
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    exercise_1_activations()
    exercise_2_linear_classifier()
    exercise_3_challenge_three_layer_network()


if __name__ == "__main__":
    main()
