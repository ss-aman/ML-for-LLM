"""
Module 02 — Code 03: Backpropagation from Scratch
===================================================
Implement a 2-layer neural network and train it by computing gradients
manually (no autograd). Then verify against numerical gradients.

This shows exactly what happens inside the backward() call in PyTorch.

Network:
  Layer 1: z1 = W1 @ x + b1   (linear)
           a1 = relu(z1)        (activation)
  Layer 2: z2 = W2 @ a1 + b2   (linear)
  Loss:    L  = MSE(z2, y)      (mean squared error)

Run: python code_03_backprop_from_scratch.py
"""

import numpy as np


# =============================================================================
# ACTIVATION FUNCTIONS + DERIVATIVES
# =============================================================================

def relu(x):
    return np.maximum(0, x)

def relu_backward(grad_out, z):
    """
    Gradient through ReLU.
    Passes grad_out where relu was active (z > 0), blocks it where inactive.
    """
    return grad_out * (z > 0).astype(float)


# =============================================================================
# THE 2-LAYER NETWORK
# =============================================================================

class TwoLayerNet:
    """
    2-layer neural network with manual backpropagation.

    Architecture:
        x  → [Linear W1,b1] → z1 → [ReLU] → a1
           → [Linear W2,b2] → z2 → prediction ŷ
        L = MSE(ŷ, y)

    Every forward pass stores intermediate values needed for backprop.
    """

    def __init__(self, d_in, d_hidden, d_out, seed=None):
        if seed is not None:
            np.random.seed(seed)
        # Xavier initialization: scale by sqrt(1/fan_in) for stability
        self.W1 = np.random.randn(d_hidden, d_in)  * np.sqrt(1.0 / d_in)
        self.b1 = np.zeros(d_hidden)
        self.W2 = np.random.randn(d_out, d_hidden) * np.sqrt(1.0 / d_hidden)
        self.b2 = np.zeros(d_out)

        # Storage for intermediate values (set during forward pass)
        self.cache = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass: compute prediction from input x.
        IMPORTANT: store all intermediate values in self.cache for backprop.
        """
        # Layer 1: linear + relu
        z1 = self.W1 @ x + self.b1          # linear transformation
        a1 = relu(z1)                         # activation

        # Layer 2: linear (output)
        z2 = self.W2 @ a1 + self.b2          # linear transformation

        # Store everything needed for backward
        self.cache = {'x': x, 'z1': z1, 'a1': a1, 'z2': z2}

        return z2   # prediction ŷ

    def loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """MSE loss: mean((ŷ - y)²)"""
        return float(np.mean((y_pred - y_true)**2))

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> dict:
        """
        Backward pass: compute gradients for all parameters.

        Returns a dict with dW1, db1, dW2, db2.

        Step-by-step chain rule application:
          dL/dŷ  →  dL/dz2, dL/dW2, dL/db2
                 →  dL/da1
                 →  dL/dz1  (through relu)
                 →  dL/dW1, dL/db1
        """
        x  = self.cache['x']
        z1 = self.cache['z1']
        a1 = self.cache['a1']

        n = len(y_pred)

        # ── Step 1: gradient of MSE loss ──────────────────────────────────────
        # L = mean((ŷ - y)²) = (1/n) * sum((ŷᵢ - yᵢ)²)
        # dL/dŷ = (2/n) * (ŷ - y)
        dL_dypred = (2.0 / n) * (y_pred - y_true)    # shape: (d_out,)

        # ── Step 2: gradient through Layer 2 (z2 = W2 @ a1 + b2) ─────────────
        # Chain rule for linear layer:
        #   dL/dW2 = dL/dz2 ⊗ a1   (outer product — how did each W2[i,j] contribute?)
        #   dL/db2 = dL/dz2         (bias gradient = incoming gradient directly)
        #   dL/da1 = W2.T @ dL/dz2  (pass gradient backward to a1)

        dL_dz2 = dL_dypred                                  # same as dL/dŷ for this layer

        dW2 = np.outer(dL_dz2, a1)     # (d_out, d_hidden) — how each W2 element contributed
        db2 = dL_dz2                    # (d_out,)
        dL_da1 = self.W2.T @ dL_dz2    # (d_hidden,) — gradient flowing to a1

        # ── Step 3: gradient through ReLU (a1 = relu(z1)) ────────────────────
        # relu'(z) = 1 if z > 0 else 0
        # Chain rule: dL/dz1 = dL/da1 * relu'(z1)
        dL_dz1 = relu_backward(dL_da1, z1)   # (d_hidden,)

        # ── Step 4: gradient through Layer 1 (z1 = W1 @ x + b1) ──────────────
        dW1 = np.outer(dL_dz1, x)      # (d_hidden, d_in)
        db1 = dL_dz1                    # (d_hidden,)

        return {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}

    def update_params(self, grads: dict, lr: float):
        """Gradient descent update step."""
        self.W1 -= lr * grads['dW1']
        self.b1 -= lr * grads['db1']
        self.W2 -= lr * grads['dW2']
        self.b2 -= lr * grads['db2']


# =============================================================================
# GRADIENT CHECK: verify backprop against numerical gradients
# =============================================================================

def gradient_check(net: TwoLayerNet, x: np.ndarray, y: np.ndarray, h=1e-4):
    """
    Verify that our manual backprop matches numerical gradients.

    For each parameter, perturb it by ±h, measure the change in loss,
    and compare to the backprop gradient.
    """
    # Compute analytical gradients via backprop
    y_pred = net.forward(x)
    grads  = net.backward(y_pred, y)

    print("\nGradient Check (backprop vs numerical):")
    print(f"{'Param':>8}  {'Index':>8}  {'Backprop':>14}  {'Numerical':>14}  {'Rel Error':>12}  {'OK?':>5}")
    print("-" * 70)

    all_pass = True
    for name, param, grad in [
        ('W1', net.W1, grads['dW1']),
        ('b1', net.b1, grads['db1']),
        ('W2', net.W2, grads['dW2']),
        ('b2', net.b2, grads['db2']),
    ]:
        flat_param = param.flatten()
        flat_grad  = grad.flatten()

        # Check only first 4 elements per param (enough to verify)
        for i in range(min(4, len(flat_param))):
            orig = flat_param[i]

            flat_param[i] = orig + h
            param[:]       = flat_param.reshape(param.shape)
            loss_plus      = net.loss(net.forward(x), y)

            flat_param[i] = orig - h
            param[:]       = flat_param.reshape(param.shape)
            loss_minus     = net.loss(net.forward(x), y)

            flat_param[i] = orig   # restore
            param[:]       = flat_param.reshape(param.shape)

            numerical = (loss_plus - loss_minus) / (2 * h)
            analytical = flat_grad[i]

            denom    = max(abs(numerical), abs(analytical), 1e-8)
            rel_err  = abs(numerical - analytical) / denom
            ok       = rel_err < 1e-4
            if not ok:
                all_pass = False

            print(f"{name:>8}  {i:>8}  {analytical:>14.6f}  {numerical:>14.6f}  {rel_err:>12.2e}  {'✓' if ok else '✗'}")

    status = "ALL PASS ✓ — backprop is correct!" if all_pass else "SOME FAILURES ✗"
    print(f"\nGradient check result: {status}")
    return all_pass


# =============================================================================
# TRAINING DEMO: XOR Problem
# =============================================================================

def train_xor():
    """
    Train the 2-layer network to learn XOR.
    XOR cannot be solved by a single linear layer — needs at least 1 hidden layer.
    This demonstrates that backprop actually works.

    XOR truth table:
        [0, 0] → 0
        [0, 1] → 1
        [1, 0] → 1
        [1, 1] → 0
    """
    print("\n" + "=" * 55)
    print("Training on XOR (classic non-linear problem)")
    print("=" * 55)

    # XOR dataset
    X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
    Y = np.array([[0],[1],[1],[0]],          dtype=float)

    net = TwoLayerNet(d_in=2, d_hidden=8, d_out=1, seed=2)

    lr          = 0.3
    n_epochs    = 4000
    loss_hist   = []
    idx         = list(range(len(X)))

    for epoch in range(n_epochs):
        total_loss = 0.0

        # Shuffle examples each epoch — prevents getting stuck in bad update order
        np.random.shuffle(idx)

        for i in idx:
            x_i = X[i]
            y_i = Y[i]

            # Forward
            y_pred = net.forward(x_i)
            L      = net.loss(y_pred, y_i)
            total_loss += L

            # Backward
            grads = net.backward(y_pred, y_i)

            # Update
            net.update_params(grads, lr)

        avg_loss = total_loss / len(X)
        loss_hist.append(avg_loss)

        if epoch % 500 == 0:
            print(f"  Epoch {epoch:>5}: loss = {avg_loss:.6f}")

    print(f"  Epoch {n_epochs - 1:>5}: loss = {loss_hist[-1]:.6f}")

    # Test: check predictions
    print("\nFinal predictions:")
    print(f"  {'Input':>10}  {'Target':>8}  {'Predicted':>12}  {'Rounded':>8}")
    for x_i, y_i in zip(X, Y):
        pred   = float(net.forward(x_i).flat[0])
        target = float(y_i.flat[0])
        print(f"  {str(x_i.tolist()):>10}  {target:>8.1f}  {pred:>12.4f}  {round(pred):>8}")

    # Plot training curve
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7, 4))
    plt.semilogy(loss_hist, 'b-', linewidth=2)
    plt.xlabel('Epoch'); plt.ylabel('Loss (log scale)')
    plt.title('Backprop Training: XOR Problem')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('backprop_xor_training.png', dpi=100)
    plt.close()
    print("\nSaved: backprop_xor_training.png")

    correct = sum(
        1 for x_i, y_i in zip(X, Y)
        if round(float(net.forward(x_i).flat[0])) == int(y_i.flat[0])
    )
    print(f"\nAccuracy: {correct}/{len(X)} correct")


if __name__ == '__main__':
    print("=" * 55)
    print("Module 02 — Backpropagation from Scratch")
    print("=" * 55)

    # Use a simple test case for gradient check
    d_in, d_hidden, d_out = 3, 4, 2
    net = TwoLayerNet(d_in, d_hidden, d_out, seed=42)
    np.random.seed(7)
    x_test = np.random.randn(d_in)
    y_test = np.random.randn(d_out)

    gradient_check(net, x_test, y_test)

    # Train on XOR to show the full loop works
    train_xor()
