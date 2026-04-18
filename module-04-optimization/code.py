"""
Module 04: Optimization Algorithms — Code Demonstrations
=========================================================
Covers:
  - Vanilla gradient descent on a 2D loss surface
  - SGD with mini-batches on a linear regression problem
  - Adam optimizer implemented from scratch
  - SGD vs Adam convergence comparison
  - Loss curve visualizations

Run this file directly: python code.py
"""

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 1. Vanilla Gradient Descent on a 2D Loss Surface
# ---------------------------------------------------------------------------

def demo_vanilla_gradient_descent():
    """
    Minimize f(x, y) = (x - 3)^2 + (y + 2)^2
    True minimum: (x=3, y=-2)  — an obvious bowl in 2D

    Backend analogy: imagine x and y are two config knobs on your service.
    f(x,y) is the measured p99 latency.  We want the knob settings that give
    the lowest latency, but we can only observe the gradient at one point
    at a time (like A/B testing a single config change).
    """
    print("=" * 60)
    print("1. Vanilla Gradient Descent on f(x,y) = (x-3)^2 + (y+2)^2")
    print("=" * 60)

    def loss(x, y):
        return (x - 3) ** 2 + (y + 2) ** 2

    def grad(x, y):
        # d/dx = 2(x-3),  d/dy = 2(y+2)
        return np.array([2 * (x - 3), 2 * (y + 2)])

    # Start far from the minimum
    params = np.array([-4.0, 6.0])
    lr = 0.1
    n_steps = 60

    history = [params.copy()]
    losses = []

    for step in range(n_steps):
        x, y = params
        g = grad(x, y)
        params = params - lr * g          # θ_new = θ_old - lr * ∇L
        history.append(params.copy())
        losses.append(loss(*params))

    history = np.array(history)
    print(f"  Start : ({history[0,0]:.2f}, {history[0,1]:.2f})  loss={loss(*history[0]):.4f}")
    print(f"  End   : ({history[-1,0]:.2f}, {history[-1,1]:.2f})  loss={loss(*history[-1]):.6f}")
    print(f"  True min: (3.00, -2.00)")

    # -- Plot --
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Contour of the loss surface
    xs = np.linspace(-5, 6, 200)
    ys = np.linspace(-5, 6, 200)
    X, Y = np.meshgrid(xs, ys)
    Z = loss(X, Y)
    axes[0].contour(X, Y, Z, levels=25, cmap="RdYlGn_r")
    axes[0].plot(history[:, 0], history[:, 1], "bo-", markersize=3, label="GD path")
    axes[0].plot(3, -2, "r*", markersize=15, label="True min")
    axes[0].set_title("Gradient Descent Path on Loss Surface")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].legend()

    # Loss over steps
    axes[1].plot(losses)
    axes[1].set_title("Loss over Steps (Vanilla GD)")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Loss")
    axes[1].set_yscale("log")
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig("plot_01_vanilla_gd.png", dpi=100)
    print("  Saved: plot_01_vanilla_gd.png\n")


# ---------------------------------------------------------------------------
# 2. SGD with Mini-Batches on Linear Regression
# ---------------------------------------------------------------------------

def demo_sgd_minibatch():
    """
    Linear regression with SGD:  y = 2x + 1 + noise
    We learn weights w, b so that w*x + b ≈ y.

    Backend analogy: instead of computing average latency over ALL requests
    before adjusting your load balancer (expensive), you sample a random
    batch of 32 requests per second and adjust based on that sample.
    The estimate is noisier but you adjust 1000x more often.
    """
    print("=" * 60)
    print("2. SGD with Mini-Batches on Linear Regression  y = 2x + 1")
    print("=" * 60)

    rng = np.random.default_rng(42)

    # True relationship: y = 2x + 1 + noise
    N = 500
    x = rng.uniform(-3, 3, N)
    y = 2.0 * x + 1.0 + rng.normal(0, 0.5, N)

    # Initialise weights randomly
    w = rng.normal(0, 0.1)
    b = 0.0
    lr = 0.01
    batch_size = 32
    n_epochs = 10

    losses = []

    for epoch in range(n_epochs):
        # Shuffle data each epoch — key step in SGD
        idx = rng.permutation(N)
        x_shuf, y_shuf = x[idx], y[idx]

        for start in range(0, N, batch_size):
            xb = x_shuf[start: start + batch_size]
            yb = y_shuf[start: start + batch_size]

            # Forward: predictions and loss (MSE)
            y_pred = w * xb + b
            error  = y_pred - yb          # residual
            mse    = np.mean(error ** 2)

            # Gradients of MSE w.r.t. w and b
            dw = np.mean(2 * error * xb)
            db = np.mean(2 * error)

            # Gradient descent step
            w -= lr * dw
            b -= lr * db

        # Full-dataset loss after each epoch
        full_loss = np.mean((w * x + b - y) ** 2)
        losses.append(full_loss)

    print(f"  Learned: w={w:.4f}  b={b:.4f}  (true: w=2.0, b=1.0)")
    print(f"  Final MSE: {losses[-1]:.4f}\n")

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.scatter(x[:100], y[:100], s=10, alpha=0.5, label="data")
    xs_line = np.linspace(-3, 3, 100)
    plt.plot(xs_line, w * xs_line + b, "r-", linewidth=2, label=f"fit: y={w:.2f}x+{b:.2f}")
    plt.plot(xs_line, 2 * xs_line + 1, "g--", linewidth=2, label="true: y=2x+1")
    plt.legend()
    plt.title("SGD Linear Regression Fit")
    plt.xlabel("x")
    plt.ylabel("y")

    plt.subplot(1, 2, 2)
    plt.plot(losses, marker="o")
    plt.title("MSE Loss per Epoch (Mini-Batch SGD)")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("plot_02_sgd_linear_regression.png", dpi=100)
    print("  Saved: plot_02_sgd_linear_regression.png\n")


# ---------------------------------------------------------------------------
# 3. Adam Optimizer from Scratch
# ---------------------------------------------------------------------------

class AdamOptimizer:
    """
    Adam: Adaptive Moment Estimation
    Combines momentum (1st moment) with RMSProp (2nd moment).

    Full algorithm:
        m_t = beta1 * m_{t-1} + (1-beta1) * g_t       # momentum
        v_t = beta2 * v_{t-1} + (1-beta2) * g_t^2     # adaptive rate
        m_hat = m_t / (1 - beta1^t)                    # bias correction
        v_hat = v_t / (1 - beta2^t)                    # bias correction
        theta = theta - lr * m_hat / (sqrt(v_hat) + eps)

    Backend analogy: per-parameter exponential moving average of gradients
    (momentum) and squared gradients (normaliser) — like per-endpoint rate
    limiting that auto-adjusts to each endpoint's traffic volatility.
    """

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr    = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps   = eps
        self.m     = None   # first moment (momentum)
        self.v     = None   # second moment (adaptive)
        self.t     = 0      # timestep

    def step(self, params, grads):
        """
        params: numpy array of parameters
        grads:  numpy array of gradients (same shape)
        Returns: updated params
        """
        if self.m is None:
            # Initialise state with zeros on first call
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)

        self.t += 1

        # Update biased first and second moment estimates
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * grads ** 2

        # Bias-corrected estimates
        # At t=1 with beta1=0.9: m_hat = m / (1 - 0.9^1) = m / 0.1 — large correction
        # At t=1000:              m_hat ≈ m / 1.0          — correction vanishes
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        # Parameter update
        return params - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def demo_adam_from_scratch():
    """
    Minimise f(θ) = θ^2  (simple parabola) using Adam.
    We add noise to the gradient to simulate a realistic stochastic gradient.
    """
    print("=" * 60)
    print("3. Adam Optimizer from Scratch on f(θ) = θ^2  (noisy grads)")
    print("=" * 60)

    rng = np.random.default_rng(0)

    theta = np.array([10.0])    # start far from minimum (θ=0)
    optimizer = AdamOptimizer(lr=0.1)

    losses = []
    for step in range(200):
        # True gradient of θ^2 is 2θ, plus noise
        grad = 2 * theta + rng.normal(0, 0.5, size=theta.shape)
        theta = optimizer.step(theta, grad)
        losses.append(float(theta[0] ** 2))

    print(f"  Final θ = {theta[0]:.6f}  (true min = 0.0)")
    print(f"  Final loss = {losses[-1]:.8f}\n")

    plt.figure(figsize=(6, 4))
    plt.plot(losses)
    plt.title("Adam Convergence on f(θ)=θ² (noisy gradients)")
    plt.xlabel("Step")
    plt.ylabel("Loss (θ²)")
    plt.yscale("log")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plot_03_adam_scratch.png", dpi=100)
    print("  Saved: plot_03_adam_scratch.png\n")


# ---------------------------------------------------------------------------
# 4. SGD vs Adam Convergence Comparison
# ---------------------------------------------------------------------------

def sgd_step(params, grads, lr):
    """Plain SGD: θ = θ - lr * ∇L"""
    return params - lr * grads


def demo_sgd_vs_adam():
    """
    Compare SGD and Adam on a noisy quadratic problem.

    The problem:  minimize  sum((Wh - y)^2)
    where W is a weight matrix, h is a fixed input, y is a target.

    We add significant gradient noise to simulate real mini-batch training.
    Adam should converge faster and more smoothly thanks to:
      1) Momentum: smoother trajectory
      2) Adaptive rates: self-tuning per parameter
      3) Bias correction: reliable early steps
    """
    print("=" * 60)
    print("4. SGD vs Adam Convergence Comparison")
    print("=" * 60)

    rng = np.random.default_rng(7)

    # A simple quadratic loss in 50 dimensions
    # True solution: W* = target
    dim = 50
    target = rng.normal(0, 1, dim)

    def compute_loss_and_grad(W, noise_scale=0.3):
        residual = W - target
        loss     = np.sum(residual ** 2)
        grad     = 2 * residual + rng.normal(0, noise_scale, size=W.shape)
        return loss, grad

    # --- SGD run ---
    W_sgd = np.zeros(dim)
    lr_sgd = 0.01
    sgd_losses = []

    for _ in range(500):
        loss, grad = compute_loss_and_grad(W_sgd)
        W_sgd = sgd_step(W_sgd, grad, lr_sgd)
        sgd_losses.append(loss)

    # --- Adam run ---
    W_adam = np.zeros(dim)
    adam = AdamOptimizer(lr=0.1)
    adam_losses = []

    for _ in range(500):
        loss, grad = compute_loss_and_grad(W_adam)
        W_adam = adam.step(W_adam, grad)
        adam_losses.append(loss)

    print(f"  SGD  final loss: {sgd_losses[-1]:.4f}")
    print(f"  Adam final loss: {adam_losses[-1]:.4f}")

    # Find step where each first drops below 10% of initial loss
    sgd_threshold  = sgd_losses[0]  * 0.1
    adam_threshold = adam_losses[0] * 0.1
    sgd_steps_to_10pct  = next((i for i, l in enumerate(sgd_losses)  if l < sgd_threshold),  500)
    adam_steps_to_10pct = next((i for i, l in enumerate(adam_losses) if l < adam_threshold), 500)
    print(f"  Steps to reach 10% of initial loss:")
    print(f"    SGD:  {sgd_steps_to_10pct}")
    print(f"    Adam: {adam_steps_to_10pct}")
    print()

    plt.figure(figsize=(8, 4))
    plt.plot(sgd_losses,  label=f"SGD  (lr={lr_sgd})", alpha=0.8)
    plt.plot(adam_losses, label="Adam (lr=0.1)",  alpha=0.8)
    plt.title("SGD vs Adam: Loss Convergence (noisy quadratic, 50 dims)")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plot_04_sgd_vs_adam.png", dpi=100)
    print("  Saved: plot_04_sgd_vs_adam.png\n")


# ---------------------------------------------------------------------------
# 5. Learning Rate Effect Visualisation
# ---------------------------------------------------------------------------

def demo_learning_rate_effect():
    """
    Show how different learning rates behave on a simple 1D parabola.
    Backend analogy: like tuning retry-backoff initial wait time —
    too high = chaotic, too low = very slow, just right = smooth.
    """
    print("=" * 60)
    print("5. Effect of Different Learning Rates on f(θ) = θ^2")
    print("=" * 60)

    def loss_fn(theta):  return theta ** 2
    def grad_fn(theta):  return 2 * theta

    learning_rates = [0.001, 0.1, 0.9, 1.05]
    colors         = ["blue", "green", "orange", "red"]
    labels         = ["0.001 (too slow)", "0.1 (good)", "0.9 (oscillating)", "1.05 (diverges)"]
    n_steps        = 40

    plt.figure(figsize=(10, 4))

    for lr, color, label in zip(learning_rates, colors, labels):
        theta = 8.0
        losses = []
        for _ in range(n_steps):
            theta = theta - lr * grad_fn(theta)
            losses.append(loss_fn(theta))
            if abs(theta) > 1e6:   # diverged
                losses.extend([np.nan] * (n_steps - len(losses)))
                break
        plt.plot(losses, color=color, label=label, linewidth=2)

    plt.title("Effect of Learning Rate on f(θ)=θ²")
    plt.xlabel("Step")
    plt.ylabel("Loss (θ²)")
    plt.yscale("symlog", linthresh=1e-4)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plot_05_learning_rate_effect.png", dpi=100)
    print("  Saved: plot_05_learning_rate_effect.png\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo_vanilla_gradient_descent()
    demo_sgd_minibatch()
    demo_adam_from_scratch()
    demo_sgd_vs_adam()
    demo_learning_rate_effect()

    print("All demos complete.  PNG plots saved to the current directory.")
