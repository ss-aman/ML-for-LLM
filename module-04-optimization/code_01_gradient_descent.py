"""
Module 04 — Code 01: Gradient Descent Variants
================================================
Implement and compare full-batch GD, SGD, and mini-batch SGD.
Visualize the loss landscape and optimization paths.

Run: python code_01_gradient_descent.py
"""

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# SECTION 1: Visualizing the Loss Landscape
# =============================================================================

def section_loss_landscape():
    print("=" * 55)
    print("SECTION 1: Loss Landscape Visualization")
    print("=" * 55)

    # An elongated bowl: different curvature in each dimension
    # f(w1, w2) = w1^2 + 10*w2^2
    # Gradient: [2*w1, 20*w2]
    # Minimum: (0, 0)
    def loss(w):
        return float(w[0]**2 + 10 * w[1]**2)

    def gradient(w):
        return np.array([2 * w[0], 20 * w[1]])

    w1_range = np.linspace(-5, 5, 200)
    w2_range = np.linspace(-1.5, 1.5, 200)
    W1, W2   = np.meshgrid(w1_range, w2_range)
    Z        = W1**2 + 10 * W2**2

    # Run gradient descent from starting point
    w_start = np.array([4.0, 1.2])
    w       = w_start.copy()
    lr      = 0.08
    path    = [w.copy()]
    losses  = [loss(w)]

    for _ in range(30):
        g = gradient(w)
        w = w - lr * g
        path.append(w.copy())
        losses.append(loss(w))

    path = np.array(path)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss contour with path
    axes[0].contourf(W1, W2, Z, levels=20, cmap='Blues', alpha=0.6)
    axes[0].contour(W1, W2, Z, levels=20, colors='gray', alpha=0.4, linewidths=0.5)
    axes[0].plot(path[:, 0], path[:, 1], 'ro-', markersize=4, linewidth=2, label='GD path')
    axes[0].plot(path[0, 0], path[0, 1], 'g^', markersize=10, label='Start')
    axes[0].plot(0, 0, 'k*', markersize=12, label='Minimum')
    axes[0].set_xlabel('w₁')
    axes[0].set_ylabel('w₂')
    axes[0].set_title('f(w) = w₁² + 10w₂² (elongated bowl)')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Loss curve
    axes[1].semilogy(losses, 'b-o', markersize=4, linewidth=2)
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('Loss (log scale)')
    axes[1].set_title('Loss vs Step')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('loss_landscape.png', dpi=100)
    plt.close()

    print(f"\nElongated bowl: f(w) = w1² + 10·w2²")
    print(f"  Minimum at: (0, 0)")
    print(f"  Start: {w_start}")
    print(f"  After 30 steps (lr={lr}): ({path[-1,0]:.5f}, {path[-1,1]:.5f})")
    print(f"  Final loss: {losses[-1]:.8f}")
    print(f"\nObservation: gradient descent oscillates in the steep w2 direction")
    print(f"  because steps overshoot in the high-curvature dimension.")
    print(f"Saved: loss_landscape.png")


# =============================================================================
# SECTION 2: Full-Batch vs Mini-Batch SGD
# =============================================================================

def section_batch_comparison():
    print("\n" + "=" * 55)
    print("SECTION 2: Full-Batch vs Mini-Batch SGD")
    print("=" * 55)

    np.random.seed(42)

    # True linear relationship: y = 3x - 2 + noise
    n_samples  = 1000
    X_true     = np.random.uniform(-3, 3, n_samples)
    y_true     = 3.0 * X_true - 2.0 + np.random.normal(0, 0.5, n_samples)
    true_w, true_b = 3.0, -2.0

    def mse_loss(w, b, X, y):
        return np.mean((w * X + b - y) ** 2)

    def mse_grad(w, b, X, y):
        err = w * X + b - y
        dw  = 2 * np.mean(err * X)
        db  = 2 * np.mean(err)
        return dw, db

    lr = 0.01
    n_steps = 200

    # --- Full-batch GD ---
    w_fb, b_fb = 0.0, 0.0
    losses_fullbatch = []
    for _ in range(n_steps):
        dw, db = mse_grad(w_fb, b_fb, X_true, y_true)
        w_fb  -= lr * dw
        b_fb  -= lr * db
        losses_fullbatch.append(mse_loss(w_fb, b_fb, X_true, y_true))

    # --- Mini-batch SGD (batch_size=32) ---
    w_mb, b_mb = 0.0, 0.0
    losses_minibatch = []
    batch_size = 32
    for step in range(n_steps):
        idx = np.random.choice(n_samples, size=batch_size, replace=False)
        Xb, yb = X_true[idx], y_true[idx]
        dw, db = mse_grad(w_mb, b_mb, Xb, yb)
        w_mb  -= lr * dw
        b_mb  -= lr * db
        losses_minibatch.append(mse_loss(w_mb, b_mb, X_true, y_true))

    # --- Single-sample SGD (batch_size=1) ---
    w_sg, b_sg = 0.0, 0.0
    losses_sgd1 = []
    for _ in range(n_steps):
        i = np.random.randint(0, n_samples)
        dw, db = mse_grad(w_sg, b_sg, X_true[i:i+1], y_true[i:i+1])
        w_sg  -= lr * dw
        b_sg  -= lr * db
        losses_sgd1.append(mse_loss(w_sg, b_sg, X_true, y_true))

    print(f"\nLinear regression: y = 3x - 2 + noise,  {n_samples} samples")
    print(f"\n{'Method':25s}  {'Final w':>8}  {'Final b':>8}  {'Final loss':>12}")
    print("-" * 60)
    for name, w, b, losses in [
        ("Full-batch GD",       w_fb, b_fb, losses_fullbatch),
        ("Mini-batch (bs=32)",  w_mb, b_mb, losses_minibatch),
        ("SGD (bs=1)",          w_sg, b_sg, losses_sgd1),
    ]:
        print(f"  {name:25s}  {w:>8.4f}  {b:>8.4f}  {losses[-1]:>12.6f}")
    print(f"\n  True values:               w=3.0000  b=-2.0000")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    steps = range(n_steps)
    axes[0].semilogy(steps, losses_fullbatch, 'b-', label='Full-batch GD', linewidth=2)
    axes[0].semilogy(steps, losses_minibatch, 'g-', label='Mini-batch (bs=32)', linewidth=1.5, alpha=0.8)
    axes[0].semilogy(steps, losses_sgd1,      'r-', label='SGD (bs=1)',    linewidth=1, alpha=0.6)
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('MSE Loss (log scale)')
    axes[0].set_title('Convergence: Full-batch vs Mini-batch vs SGD')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    x_plot = np.linspace(-3, 3, 100)
    axes[1].scatter(X_true[:100], y_true[:100], s=10, alpha=0.4, label='Data')
    axes[1].plot(x_plot, true_w * x_plot + true_b, 'k--', linewidth=2, label=f'True: y=3x-2')
    axes[1].plot(x_plot, w_fb * x_plot + b_fb, 'b-', linewidth=2, label=f'Full-batch: y={w_fb:.2f}x{b_fb:+.2f}')
    axes[1].plot(x_plot, w_mb * x_plot + b_mb, 'g-', linewidth=1.5, label=f'Mini-batch: y={w_mb:.2f}x{b_mb:+.2f}')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    axes[1].set_title('Fitted Line Comparison')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('batch_comparison.png', dpi=100)
    plt.close()
    print("\nSaved: batch_comparison.png")
    print("\nKey insight: Mini-batch and full-batch converge to similar solutions.")
    print("  Mini-batch loss is noisier but often reaches good solution faster")
    print("  due to the regularization effect of gradient noise.")


# =============================================================================
# SECTION 3: Learning Rate Effect
# =============================================================================

def section_learning_rate():
    print("\n" + "=" * 55)
    print("SECTION 3: Learning Rate Effect")
    print("=" * 55)

    # f(θ) = θ^2,  gradient = 2θ,  minimum at θ=0
    def loss_fn(theta): return theta ** 2
    def grad_fn(theta): return 2 * theta

    start  = 8.0
    n_steps = 50
    configs = [
        (0.001, 'blue',   'lr=0.001 (too slow)'),
        (0.1,   'green',  'lr=0.1   (good)'),
        (0.9,   'orange', 'lr=0.9   (oscillating)'),
        (1.05,  'red',    'lr=1.05  (diverges)'),
    ]

    print(f"\nf(θ) = θ²,  start=8.0,  {n_steps} steps")
    print(f"\n{'Config':30s}  {'Final θ':>10}  {'Final loss':>12}  {'Status':>12}")
    print("-" * 70)

    plt.figure(figsize=(10, 4))
    for lr, color, label in configs:
        theta  = start
        losses = []
        for _ in range(n_steps):
            theta  = theta - lr * grad_fn(theta)
            losses.append(loss_fn(theta))
            if abs(theta) > 1e6:
                losses.extend([np.nan] * (n_steps - len(losses)))
                break

        final_loss  = losses[-1] if not np.isnan(losses[-1]) else float('inf')
        status      = "DIVERGED" if np.isnan(losses[-1]) else f"loss={final_loss:.6f}"
        print(f"  {label:30s}  {theta if abs(theta)<1e6 else float('inf'):>10.4f}  {final_loss:>12}  {status:>12}")

        plt.plot([x if not np.isnan(x) else 0 for x in losses],
                 color=color, label=label, linewidth=2)

    plt.xlabel('Step')
    plt.ylabel('Loss (θ²)')
    plt.title('Learning Rate Effect on Convergence')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig('learning_rate_effect.png', dpi=100)
    plt.close()
    print("\nSaved: learning_rate_effect.png")
    print("\nThe stability condition for this problem: lr < 1/max_eigenvalue = 1/2 = 0.5")
    print("  lr=0.9: technically stable but oscillating (2 - 2*0.9 = 0.2 damping)")
    print("  lr=1.05: 2*1.05 = 2.1 > 2, so each step amplifies error → diverges")


# =============================================================================
# SECTION 4: Noise Helps Escape Saddle Points
# =============================================================================

def section_noise_and_saddle():
    print("\n" + "=" * 55)
    print("SECTION 4: How SGD Noise Escapes Saddle Points")
    print("=" * 55)

    # Saddle point function: f(x,y) = x^2 - y^2
    # Gradient: [2x, -2y]
    # Saddle at (0,0): gradient=0 but NOT a minimum
    def saddle_loss(w):
        return float(w[0]**2 - w[1]**2)

    def saddle_grad(w):
        return np.array([2 * w[0], -2 * w[1]])

    print("\nFunction: f(x,y) = x² - y²")
    print("Saddle point at (0,0): gradient=0 but f goes DOWN in y direction")

    np.random.seed(0)
    lr       = 0.01
    n_steps  = 200
    start    = np.array([0.01, 0.01])  # start near the saddle

    # Full-batch GD: gets stuck at saddle
    w_gd  = start.copy()
    path_gd = [w_gd.copy()]
    for _ in range(n_steps):
        g    = saddle_grad(w_gd)
        w_gd = w_gd - lr * g
        path_gd.append(w_gd.copy())
    path_gd = np.array(path_gd)

    # SGD with noise: can escape saddle
    w_sgd    = start.copy()
    path_sgd = [w_sgd.copy()]
    for _ in range(n_steps):
        noisy_g = saddle_grad(w_sgd) + np.random.normal(0, 0.1, 2)
        w_sgd   = w_sgd - lr * noisy_g
        path_sgd.append(w_sgd.copy())
    path_sgd = np.array(path_sgd)

    print(f"\n  Full-batch GD:  starts at {start}, ends at ({path_gd[-1,0]:.4f}, {path_gd[-1,1]:.4f})")
    print(f"  → stuck near saddle (gradient ≈ 0, tiny updates only)")
    print(f"\n  SGD with noise: starts at {start}, ends at ({path_sgd[-1,0]:.4f}, {path_sgd[-1,1]:.4f})")
    print(f"  → escaped saddle due to gradient noise")

    # Plot
    x_range = np.linspace(-1, 1, 100)
    y_range = np.linspace(-1, 1, 100)
    X, Y    = np.meshgrid(x_range, y_range)
    Z       = X**2 - Y**2

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, path, title in [
        (axes[0], path_gd,  'Full-Batch GD (stuck at saddle)'),
        (axes[1], path_sgd, 'SGD with noise (escapes saddle)'),
    ]:
        ax.contourf(X, Y, Z, levels=20, cmap='RdBu', alpha=0.6)
        ax.contour(X, Y, Z, levels=[0], colors='black', linewidths=2)  # saddle line
        ax.plot(path[:, 0], path[:, 1], 'w-o', markersize=3, linewidth=1.5)
        ax.plot(path[0, 0], path[0, 1], 'g^', markersize=10)
        ax.plot(0, 0, 'k*', markersize=12, label='Saddle point')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig('saddle_escape.png', dpi=100)
    plt.close()
    print("\nSaved: saddle_escape.png")


# =============================================================================
# RUN ALL SECTIONS
# =============================================================================

if __name__ == '__main__':
    section_loss_landscape()
    section_batch_comparison()
    section_learning_rate()
    section_noise_and_saddle()

    print("\n" + "=" * 55)
    print("Key takeaways:")
    print("  1. Loss landscape: elongated bowls cause oscillation")
    print("  2. Mini-batch ≈ full-batch in quality, faster in practice")
    print("  3. Learning rate: too high diverges, too low wastes compute")
    print("  4. SGD noise helps escape saddle points (common in deep nets)")
    print("=" * 55)
