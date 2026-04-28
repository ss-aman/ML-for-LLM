"""
Module 07 — Neural Networks: From Scratch Implementation
=========================================================
Builds a 2-layer neural network using only NumPy.
Demonstrates:
  - Why a single linear layer cannot solve XOR (non-linearly separable)
  - Why adding a hidden layer + ReLU solves XOR
  - Full forward pass + manual backpropagation

Backend analogy: This is the raw engine underneath every ML framework.
Think of it like writing an HTTP server from raw sockets before using FastAPI.
"""

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class LinearLayer:
    """
    A fully-connected linear layer: output = W @ x + b

    Backend analogy: Like a middleware function that multiplies every incoming
    feature by a learned importance weight and adds an offset. The weights W
    and biases b are the "config" that gets tuned during training.
    """

    def __init__(self, input_size: int, output_size: int, seed: int = 42):
        rng = np.random.default_rng(seed)
        # He initialization: scale weights by sqrt(2 / fan_in)
        # Keeps variance stable as we stack layers (avoids exploding/vanishing signals)
        self.W = rng.standard_normal((output_size, input_size)) * np.sqrt(2.0 / input_size)
        self.b = np.zeros((output_size, 1))

        # Cache inputs for backprop (we need them to compute gradients)
        self._input_cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        x shape: (input_size, batch_size)
        output shape: (output_size, batch_size)
        """
        self._input_cache = x
        return self.W @ x + self.b

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        """
        Receive gradient from the layer above (d_out = dLoss/d_output).
        Compute and store gradients for W and b.
        Return gradient w.r.t. the input so the layer below can continue backprop.

        Chain rule:
          dLoss/dW = dLoss/d_output  @  input^T
          dLoss/db = sum of dLoss/d_output across batch
          dLoss/d_input = W^T  @  dLoss/d_output
        """
        batch_size = self._input_cache.shape[1]

        self.dW = (d_out @ self._input_cache.T) / batch_size   # average across batch
        self.db = np.mean(d_out, axis=1, keepdims=True)
        d_input = self.W.T @ d_out
        return d_input

    def update(self, learning_rate: float):
        """Gradient descent step: move weights in the direction that reduces loss."""
        self.W -= learning_rate * self.dW
        self.b -= learning_rate * self.db


class ReLU:
    """
    ReLU activation: f(x) = max(0, x)

    Backend analogy: A floor threshold — like clamping a latency metric to 0.
    Negative activations get zeroed out (that neuron "didn't fire").
    Derivative is either 0 (input was negative) or 1 (input was positive).
    """

    def __init__(self):
        self._input_cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._input_cache = x
        return np.maximum(0, x)

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        # Gradient only flows where the input was positive
        return d_out * (self._input_cache > 0).astype(float)


class Sigmoid:
    """
    Sigmoid activation: f(x) = 1 / (1 + e^(-x))
    Output is in (0, 1) — useful as a probability for binary classification.
    """

    def __init__(self):
        self._output_cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        out = 1.0 / (1.0 + np.exp(-x))
        self._output_cache = out
        return out

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        # Derivative: sigmoid(x) * (1 - sigmoid(x))
        s = self._output_cache
        return d_out * s * (1.0 - s)


def binary_cross_entropy(predictions: np.ndarray, targets: np.ndarray):
    """
    Loss for binary classification.
    BCE = -mean( y * log(p) + (1-y) * log(1-p) )

    Backend analogy: Measures how "wrong" our probability estimates are.
    Predicting 0.99 when the answer is 0 is very costly; predicting 0.5 is
    moderately costly; predicting 0.99 when the answer is 1 is near-free.
    """
    eps = 1e-9  # avoid log(0)
    loss = -np.mean(
        targets * np.log(predictions + eps) + (1 - targets) * np.log(1 - predictions + eps)
    )
    # Gradient of BCE w.r.t. predictions
    d_pred = -(targets / (predictions + eps) - (1 - targets) / (1 - predictions + eps)) / targets.shape[1]
    return loss, d_pred


# ---------------------------------------------------------------------------
# Demonstration 1: XOR cannot be solved by a single linear layer
# ---------------------------------------------------------------------------

def demo_xor_fails_with_one_layer():
    """
    XOR is the classic non-linearly separable problem.
    A single linear layer can only draw a straight line (hyperplane) in feature
    space. XOR points cannot be separated by any straight line.

    Backend analogy: A single IF condition on raw inputs cannot classify XOR.
    You need at least one intermediate transformation first.
    """
    print("=" * 60)
    print("DEMO 1: XOR with a single linear layer (should FAIL)")
    print("=" * 60)

    # XOR truth table: 4 samples, 2 features each
    # Columns = samples
    X = np.array([[0, 0, 1, 1],
                  [0, 1, 0, 1]], dtype=float)   # shape (2, 4)
    y = np.array([[0, 1, 1, 0]], dtype=float)   # shape (1, 4)

    # Single-layer network: Linear(2→1) + Sigmoid
    layer = LinearLayer(input_size=2, output_size=1, seed=0)
    sigmoid = Sigmoid()

    losses = []
    lr = 0.5
    for epoch in range(2000):
        # Forward
        z = layer.forward(X)
        pred = sigmoid.forward(z)
        loss, d_pred = binary_cross_entropy(pred, y)
        losses.append(loss)

        # Backward
        d_z = sigmoid.backward(d_pred)
        layer.backward(d_z)
        layer.update(lr)

    print(f"Final loss after 2000 epochs: {losses[-1]:.4f}")
    print(f"Predictions: {sigmoid.forward(layer.forward(X)).flatten().round(3)}")
    print(f"True labels: {y.flatten()}")
    print(f"→ Loss stays high (~0.25+). A line can't separate XOR.\n")
    return losses


# ---------------------------------------------------------------------------
# Demonstration 2: XOR solved with a 2-layer network (hidden layer + ReLU)
# ---------------------------------------------------------------------------

def demo_xor_solved_with_two_layers():
    """
    Adding a hidden layer with ReLU gives the network the ability to learn
    a non-linear decision boundary.

    Architecture:
        Input(2) → Linear(2→4) → ReLU → Linear(4→1) → Sigmoid → output

    Backend analogy: The hidden layer is an intermediate transformation step —
    like first normalising/reshaping the request body before applying the
    business logic check. That extra step makes previously inseparable cases
    separable.
    """
    print("=" * 60)
    print("DEMO 2: XOR with a 2-layer network (should SUCCEED)")
    print("=" * 60)

    X = np.array([[0, 0, 1, 1],
                  [0, 1, 0, 1]], dtype=float)
    y = np.array([[0, 1, 1, 0]], dtype=float)

    # Network components (left to right in forward direction)
    layer1 = LinearLayer(input_size=2, output_size=4, seed=7)
    relu = ReLU()
    layer2 = LinearLayer(input_size=4, output_size=1, seed=13)
    sigmoid = Sigmoid()

    losses = []
    lr = 0.5

    for epoch in range(5000):
        # ---- Forward pass ----
        # Request flows through middleware chain: X → z1 → a1 → z2 → pred
        z1 = layer1.forward(X)      # Linear transform
        a1 = relu.forward(z1)       # Nonlinearity
        z2 = layer2.forward(a1)     # Second linear transform
        pred = sigmoid.forward(z2)  # Probability output

        loss, d_pred = binary_cross_entropy(pred, y)
        losses.append(loss)

        # ---- Backward pass (gradients flow in reverse order) ----
        # Like distributed tracing: start from the error, trace back to each layer
        d_z2 = sigmoid.backward(d_pred)    # gradient through sigmoid
        d_a1 = layer2.backward(d_z2)       # gradient through layer 2
        d_z1 = relu.backward(d_a1)         # gradient through ReLU
        layer1.backward(d_z1)              # gradient through layer 1

        # ---- Weight updates ----
        layer1.update(lr)
        layer2.update(lr)

        if epoch % 1000 == 0:
            print(f"  Epoch {epoch:4d} | Loss: {loss:.4f}")

    final_pred = sigmoid.forward(layer2.forward(relu.forward(layer1.forward(X))))
    print(f"\nFinal predictions: {final_pred.flatten().round(3)}")
    print(f"True labels:       {y.flatten()}")
    print(f"Rounded:           {(final_pred.flatten() > 0.5).astype(int)}")
    print(f"→ Network learns XOR perfectly!\n")
    return losses


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_losses(loss_1layer, loss_2layer):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(loss_1layer, color='crimson')
    axes[0].set_title("1-layer network on XOR\n(fails — loss stays high)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("BCE Loss")
    axes[0].set_ylim(0, 1)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(loss_2layer, color='steelblue')
    axes[1].set_title("2-layer network on XOR\n(succeeds — loss → 0)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("BCE Loss")
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("module-07-xor-loss.png", dpi=120)
    plt.show()
    print("Plot saved to module-07-xor-loss.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    loss_1 = demo_xor_fails_with_one_layer()
    loss_2 = demo_xor_solved_with_two_layers()

    print("=" * 60)
    print(f"1-layer final loss:  {loss_1[-1]:.4f}  (stuck, can't solve XOR)")
    print(f"2-layer final loss:  {loss_2[-1]:.6f}  (converged, solves XOR)")
    print("=" * 60)

    plot_losses(loss_1, loss_2)
