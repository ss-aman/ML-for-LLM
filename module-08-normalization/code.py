"""
Module 08 — Normalization: BatchNorm, LayerNorm, Residual Connections
======================================================================
Implements from scratch:
  - BatchNorm (training mode with running stats, inference mode)
  - LayerNorm
  - Residual block: output = LayerNorm(F(x) + x)

Demonstrates:
  - Why normalization helps: train a deep network with and without LayerNorm
    and compare loss curves

Backend analogy: Normalization layers are like request-scoped or batch-scoped
pre-processing pipelines that ensure every layer sees data at a consistent scale.
Without them, deep networks are like a microservices chain where each service
assumes a different unit system — chaos accumulates with each hop.
"""

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# BatchNorm from scratch
# ---------------------------------------------------------------------------

class BatchNorm:
    """
    Batch Normalisation layer.

    During training:
      - Normalise each feature using the current batch's mean and std.
      - Accumulate running mean/var (exponential moving average) for later use.

    During inference:
      - Use the stored running mean/var (no batch statistics available).

    Parameters:
      num_features: the size of the feature dimension being normalised
      momentum:     weight given to the current batch in the running average
      eps:          small constant for numerical stability

    Backend analogy: Training mode = computing per-column z-scores live from
    the incoming rows. Inference mode = using pre-computed column stats loaded
    from a saved config (because at inference time there may be only one row).
    """

    def __init__(self, num_features: int, momentum: float = 0.1, eps: float = 1e-5):
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
        self.training = True

        # Learnable scale (gamma) and shift (beta) — initialised to identity
        self.gamma = np.ones((num_features, 1))
        self.beta = np.zeros((num_features, 1))

        # Running statistics accumulated during training, used at inference
        self.running_mean = np.zeros((num_features, 1))
        self.running_var = np.ones((num_features, 1))

        # Cache for backprop
        self._cache = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        x shape: (num_features, batch_size)
        Returns normalised output of the same shape.
        """
        if self.training:
            # Compute batch statistics: mean and variance across the batch axis
            mean = x.mean(axis=1, keepdims=True)          # (num_features, 1)
            var = x.var(axis=1, keepdims=True)             # (num_features, 1)

            # Update running stats (exponential moving average)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var

            # Normalise
            x_hat = (x - mean) / np.sqrt(var + self.eps)
        else:
            # Inference: use stored running statistics
            x_hat = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)

        # Scale and shift with learned parameters
        out = self.gamma * x_hat + self.beta

        # Cache values needed for backward pass
        self._cache = {"x_hat": x_hat, "var": var if self.training else self.running_var,
                       "batch_size": x.shape[1]}
        return out

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        """
        Backprop through BatchNorm.
        Returns gradient w.r.t. the input x.
        Also stores gradients for gamma and beta.
        """
        x_hat = self._cache["x_hat"]
        var = self._cache["var"]
        N = self._cache["batch_size"]

        self.d_gamma = (d_out * x_hat).sum(axis=1, keepdims=True)
        self.d_beta = d_out.sum(axis=1, keepdims=True)

        # Gradient w.r.t. x_hat
        d_x_hat = d_out * self.gamma

        # Full gradient through normalisation (from the BatchNorm backward formula)
        inv_std = 1.0 / np.sqrt(var + self.eps)
        d_x = (1.0 / N) * inv_std * (
            N * d_x_hat
            - d_x_hat.sum(axis=1, keepdims=True)
            - x_hat * (d_x_hat * x_hat).sum(axis=1, keepdims=True)
        )
        return d_x

    def update(self, lr: float):
        self.gamma -= lr * self.d_gamma
        self.beta -= lr * self.d_beta

    def eval(self):
        self.training = False

    def train(self):
        self.training = True


# ---------------------------------------------------------------------------
# LayerNorm from scratch
# ---------------------------------------------------------------------------

class LayerNorm:
    """
    Layer Normalisation.

    Normalises across the feature dimension for each individual sample.
    Unlike BatchNorm, this does NOT depend on other samples — it is a
    pure function of the single input vector.

    Backend analogy: A pure, stateless transformation applied per-request.
    No shared state, no running averages, no batch dependency.
    Works identically during training and inference, at any batch size,
    including batch size 1 (which is exactly what LLMs do during generation).
    """

    def __init__(self, num_features: int, eps: float = 1e-5):
        self.eps = eps
        # Learnable scale and shift
        self.gamma = np.ones((num_features, 1))
        self.beta = np.zeros((num_features, 1))
        self._cache = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        x shape: (num_features, batch_size)
        Normalises each column (sample) independently across the feature axis.
        """
        # Compute mean and variance for each sample separately
        mean = x.mean(axis=0, keepdims=True)      # (1, batch_size)
        var = x.var(axis=0, keepdims=True)         # (1, batch_size)

        x_hat = (x - mean) / np.sqrt(var + self.eps)
        out = self.gamma * x_hat + self.beta

        self._cache = {"x_hat": x_hat, "var": var, "num_features": x.shape[0]}
        return out

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        """Backprop through LayerNorm — same structure as BatchNorm but axis is flipped."""
        x_hat = self._cache["x_hat"]
        var = self._cache["var"]
        D = self._cache["num_features"]  # normalise across features, not batch

        self.d_gamma = (d_out * x_hat).sum(axis=1, keepdims=True)
        self.d_beta = d_out.sum(axis=1, keepdims=True)

        d_x_hat = d_out * self.gamma
        inv_std = 1.0 / np.sqrt(var + self.eps)
        d_x = (1.0 / D) * inv_std * (
            D * d_x_hat
            - d_x_hat.sum(axis=0, keepdims=True)
            - x_hat * (d_x_hat * x_hat).sum(axis=0, keepdims=True)
        )
        return d_x

    def update(self, lr: float):
        self.gamma -= lr * self.d_gamma
        self.beta -= lr * self.d_beta


# ---------------------------------------------------------------------------
# Reusable linear layer (from module 07 — repeated here for self-containedness)
# ---------------------------------------------------------------------------

class LinearLayer:
    def __init__(self, in_features: int, out_features: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.W = rng.standard_normal((out_features, in_features)) * np.sqrt(2.0 / in_features)
        self.b = np.zeros((out_features, 1))
        self._cache = None

    def forward(self, x):
        self._cache = x
        return self.W @ x + self.b

    def backward(self, d_out):
        N = self._cache.shape[1]
        self.dW = (d_out @ self._cache.T) / N
        self.db = d_out.mean(axis=1, keepdims=True)
        return self.W.T @ d_out

    def update(self, lr):
        self.W -= lr * self.dW
        self.b -= lr * self.db


class ReLU:
    def __init__(self):
        self._cache = None

    def forward(self, x):
        self._cache = x
        return np.maximum(0, x)

    def backward(self, d_out):
        return d_out * (self._cache > 0)


# ---------------------------------------------------------------------------
# Residual block with LayerNorm
# ---------------------------------------------------------------------------

class ResidualBlock:
    """
    output = LayerNorm(F(x) + x)

    where F is a single linear + ReLU transformation.

    Backend analogy: A circuit breaker with fallback. If F(x) produces near-zero
    output (layer saturated or under-trained), the original signal x is still
    added in — the request always passes through. The LayerNorm then stabilises
    the combined signal before the next layer sees it.
    """

    def __init__(self, features: int, seed: int = 0):
        self.linear = LinearLayer(features, features, seed=seed)
        self.relu = ReLU()
        self.norm = LayerNorm(features)
        self._x_cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x_cache = x
        fx = self.relu.forward(self.linear.forward(x))  # F(x)
        return self.norm.forward(fx + x)                 # LayerNorm(F(x) + x)

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        # Through LayerNorm
        d_sum = self.norm.backward(d_out)
        # Through F(x) + x:  gradient splits — one path through F, one direct
        d_fx = self.relu.backward(self.linear.backward(d_sum))
        d_x_skip = d_sum   # direct / skip path
        return d_fx + d_x_skip

    def update(self, lr: float):
        self.linear.update(lr)
        self.norm.update(lr)


# ---------------------------------------------------------------------------
# Convergence experiment: deep network with vs. without LayerNorm
# ---------------------------------------------------------------------------

def make_deep_network(num_layers: int, features: int, use_layernorm: bool, seed: int = 0):
    """
    Build a list of (LinearLayer, optional LayerNorm, ReLU) tuples.
    Returns the component lists so we can call forward/backward manually.
    """
    layers, norms, relus = [], [], []
    for i in range(num_layers):
        layers.append(LinearLayer(features, features, seed=seed + i))
        norms.append(LayerNorm(features) if use_layernorm else None)
        relus.append(ReLU())
    # Output layer: features → 1
    output_layer = LinearLayer(features, 1, seed=seed + num_layers)
    return layers, norms, relus, output_layer


def deep_network_forward(x, layers, norms, relus, output_layer):
    """Run forward pass through the deep network, return prediction and all activations."""
    for layer, norm, relu in zip(layers, norms, relus):
        x = layer.forward(x)
        if norm is not None:
            x = norm.forward(x)
        x = relu.forward(x)
    return output_layer.forward(x)


def deep_network_backward(d_out, layers, norms, relus, output_layer):
    """Run backward pass through the deep network."""
    d = output_layer.backward(d_out)
    for layer, norm, relu in zip(reversed(layers), reversed(norms), reversed(relus)):
        d = relu.backward(d)
        if norm is not None:
            d = norm.backward(d)
        d = layer.backward(d)
    return d


def deep_network_update(layers, norms, output_layer, lr):
    for layer, norm in zip(layers, norms):
        layer.update(lr)
        if norm is not None:
            norm.update(lr)
    output_layer.update(lr)


def compare_convergence(num_layers: int = 8, epochs: int = 500):
    """
    Train the same deep network on a regression task (predict sin(x)).
    Compare: with LayerNorm vs without LayerNorm.
    Shows that without LayerNorm, deep networks converge much more slowly
    (or not at all) because gradients vanish or explode.

    Backend analogy: Both pipelines receive the same requests (inputs).
    The one with normalised middleware passes stable signals; the one without
    sees wildly varying scales that confuse each successive layer.
    """
    print("=" * 60)
    print(f"Convergence experiment: {num_layers}-layer network, {epochs} epochs")
    print("=" * 60)

    np.random.seed(7)
    n = 256
    features = 32
    X = np.random.randn(features, n)
    # Target: sine of the sum of features — a non-trivial mapping
    y = np.sin(X.sum(axis=0, keepdims=True))   # shape (1, n)

    results = {}
    for use_norm in [False, True]:
        label = "with LayerNorm" if use_norm else "without LayerNorm"
        layers, norms, relus, out_layer = make_deep_network(
            num_layers, features, use_layernorm=use_norm, seed=42
        )
        losses = []
        lr = 0.002

        for epoch in range(epochs):
            pred = deep_network_forward(X, layers, norms, relus, out_layer)
            loss = float(np.mean((pred - y) ** 2))
            losses.append(loss)

            d_pred = 2 * (pred - y) / n
            deep_network_backward(d_pred, layers, norms, relus, out_layer)
            deep_network_update(layers, norms, out_layer, lr)

        results[label] = losses
        print(f"  {label}: start={losses[0]:.4f}, end={losses[-1]:.4f}")

    return results


def demo_residual_block():
    """Show that a residual block produces non-zero gradients even for very deep chains."""
    print("\n" + "=" * 60)
    print("Residual block demo")
    print("=" * 60)

    np.random.seed(0)
    features = 16
    n = 8

    X = np.random.randn(features, n)
    y = np.random.randn(1, n)

    # Stack 6 residual blocks + output layer
    blocks = [ResidualBlock(features, seed=i) for i in range(6)]
    output_layer = LinearLayer(features, 1, seed=99)

    losses = []
    lr = 0.01

    for epoch in range(300):
        x = X.copy()
        for block in blocks:
            x = block.forward(x)
        pred = output_layer.forward(x)
        loss = float(np.mean((pred - y) ** 2))
        losses.append(loss)

        d = output_layer.backward(2 * (pred - y) / n)
        for block in reversed(blocks):
            d = block.backward(d)
        for block in blocks:
            block.update(lr)
        output_layer.update(lr)

    print(f"  6 residual blocks: start loss={losses[0]:.4f}, end loss={losses[-1]:.4f}")
    print(f"  Gradient flows cleanly through skip connections!")
    return losses


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(convergence_results: dict, residual_losses: list):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: convergence comparison
    colors = {"with LayerNorm": "steelblue", "without LayerNorm": "crimson"}
    for label, losses in convergence_results.items():
        axes[0].plot(losses, label=label, color=colors[label])
    axes[0].set_title("Deep Network: LayerNorm vs No Norm")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[0].legend()
    axes[0].set_yscale("log")
    axes[0].grid(True, alpha=0.3)

    # Right: residual block convergence
    axes[1].plot(residual_losses, color="seagreen")
    axes[1].set_title("6-Layer Residual Network\n(LayerNorm + skip connections)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MSE Loss")
    axes[1].set_yscale("log")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("module-08-normalization.png", dpi=120)
    plt.show()
    print("Plot saved to module-08-normalization.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Show BatchNorm in action
    print("BatchNorm sanity check:")
    rng = np.random.default_rng(0)
    x_test = rng.standard_normal((4, 8)) * 10 + 5   # features not centred
    bn = BatchNorm(num_features=4)
    out_bn = bn.forward(x_test)
    print(f"  Input  mean per feature: {x_test.mean(axis=1).round(2)}")
    print(f"  Output mean per feature: {out_bn.mean(axis=1).round(3)}  (≈ 0)")
    print(f"  Output std  per feature: {out_bn.std(axis=1).round(3)}   (≈ 1)\n")

    # Show LayerNorm in action
    print("LayerNorm sanity check:")
    ln = LayerNorm(num_features=4)
    out_ln = ln.forward(x_test)
    print(f"  Input  mean per sample: {x_test.mean(axis=0).round(2)}")
    print(f"  Output mean per sample: {out_ln.mean(axis=0).round(3)}  (≈ 0)")
    print(f"  Output std  per sample: {out_ln.std(axis=0).round(3)}   (≈ 1)\n")

    # Convergence comparison
    conv_results = compare_convergence(num_layers=8, epochs=500)

    # Residual block demo
    residual_losses = demo_residual_block()

    # Plot
    plot_results(conv_results, residual_losses)
