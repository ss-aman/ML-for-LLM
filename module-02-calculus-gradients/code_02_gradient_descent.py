"""
Module 02 — Code 02: Gradient Descent
=======================================
Implement gradient descent from scratch.
Show learning rate effects, momentum, and convergence.

Run: python code_02_gradient_descent.py
"""

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# THE LOSS FUNCTION WE'LL OPTIMIZE
# =============================================================================

# f(w1, w2) = (w1 - 3)² + (w2 - 5)²
# Bowl-shaped surface, minimum at [3, 5].
# Easy to visualize in 2D and verify convergence.

def loss(w: np.ndarray) -> float:
    return float((w[0] - 3)**2 + (w[1] - 5)**2)


def gradient(w: np.ndarray) -> np.ndarray:
    return np.array([2*(w[0] - 3), 2*(w[1] - 5)])


# =============================================================================
# GRADIENT DESCENT VARIANTS
# =============================================================================

def vanilla_gradient_descent(w_init, lr, n_steps):
    """
    Pure gradient descent: w = w - lr * ∇L(w)
    Returns history of (w, loss) at each step.
    """
    w = np.array(w_init, dtype=float)
    w_hist    = [w.copy()]
    loss_hist = [loss(w)]

    for _ in range(n_steps):
        g = gradient(w)
        w = w - lr * g
        w_hist.append(w.copy())
        loss_hist.append(loss(w))

    return w_hist, loss_hist


def momentum_gradient_descent(w_init, lr, beta, n_steps):
    """
    Gradient descent with momentum.
    Accumulates a 'velocity' in consistent gradient directions.

    velocity = beta * velocity - lr * gradient
    w        = w + velocity

    beta = 0.9 means: keep 90% of previous velocity each step.
    This smooths out noisy gradients and accelerates in consistent directions.
    """
    w        = np.array(w_init, dtype=float)
    velocity = np.zeros_like(w)
    w_hist    = [w.copy()]
    loss_hist = [loss(w)]

    for _ in range(n_steps):
        g        = gradient(w)
        velocity = beta * velocity - lr * g   # accumulate velocity
        w        = w + velocity                # step using velocity
        w_hist.append(w.copy())
        loss_hist.append(loss(w))

    return w_hist, loss_hist


def adam_gradient_descent(w_init, lr, n_steps, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    Adam optimizer — used in virtually all LLM training.

    Maintains:
      m  = exponential moving average of gradient (direction / momentum)
      v  = exponential moving average of gradient² (magnitude / scaling)

    Update:
      m = beta1 * m + (1 - beta1) * grad
      v = beta2 * v + (1 - beta2) * grad²
      w -= lr * m_corrected / (sqrt(v_corrected) + eps)

    Key insight: each parameter gets its own effective learning rate based
    on the history of its gradients. Rarely-updated params → larger effective lr.
    Frequently-updated params → smaller effective lr.
    """
    w        = np.array(w_init, dtype=float)
    m        = np.zeros_like(w)    # 1st moment (mean)
    v        = np.zeros_like(w)    # 2nd moment (variance)
    w_hist    = [w.copy()]
    loss_hist = [loss(w)]

    for t in range(1, n_steps + 1):
        g = gradient(w)

        # Update moment estimates
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g**2

        # Bias correction (important in early steps when m, v are close to 0)
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)

        # Adaptive update
        w = w - lr * m_hat / (np.sqrt(v_hat) + eps)

        w_hist.append(w.copy())
        loss_hist.append(loss(w))

    return w_hist, loss_hist


# =============================================================================
# SECTIONS
# =============================================================================

def section_learning_rate_effect():
    print("=" * 55)
    print("SECTION 1: Effect of Learning Rate")
    print("=" * 55)

    lr_configs = [
        (0.005, 'blue',   'lr=0.005 (too slow)'),
        (0.1,   'green',  'lr=0.1   (good)'),
        (0.5,   'orange', 'lr=0.5   (fast)'),
        (1.05,  'red',    'lr=1.05  (diverges!)'),
    ]

    plt.figure(figsize=(8, 5))

    for lr, color, label in lr_configs:
        _, loss_hist = vanilla_gradient_descent([0.0, 0.0], lr, n_steps=40)
        display = [min(l, 250) for l in loss_hist]   # cap for readability
        plt.plot(display, color=color, label=label, linewidth=2)

        final = loss_hist[-1]
        status = "DIVERGED" if final > 200 else f"final_loss={final:.4f}"
        print(f"  lr={lr:.3f}: {status}")

    plt.xlabel('Iteration'); plt.ylabel('Loss (capped at 250)')
    plt.title('Learning Rate Effect on Gradient Descent')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('learning_rate_effect.png', dpi=100)
    plt.close()
    print("\n  Key observations:")
    print("  - Too small (0.005): converges but requires many iterations")
    print("  - Too large (1.05) : overshoots, loss INCREASES — diverges")
    print("  - Just right (0.1) : smooth convergence in ~30 steps")
    print("  Saved: learning_rate_effect.png")


def section_comparing_optimizers():
    print("\n" + "=" * 55)
    print("SECTION 2: Vanilla GD vs Momentum vs Adam")
    print("=" * 55)

    n_steps = 30
    start   = [0.0, 0.0]

    _, loss_vanilla   = vanilla_gradient_descent(start, lr=0.1,   n_steps=n_steps)
    _, loss_momentum  = momentum_gradient_descent(start, lr=0.1, beta=0.9, n_steps=n_steps)
    _, loss_adam      = adam_gradient_descent(start, lr=0.5, n_steps=n_steps)

    print(f"\n  {'Step':>5}  {'Vanilla GD':>14}  {'Momentum':>14}  {'Adam':>12}")
    print("  " + "-" * 50)
    for i in [0, 5, 10, 15, 20, 25, 30]:
        print(f"  {i:>5}  {loss_vanilla[i]:>14.6f}  {loss_momentum[i]:>14.6f}  {loss_adam[i]:>12.6f}")

    # Plot convergence curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    steps = list(range(n_steps + 1))
    axes[0].semilogy(steps, loss_vanilla,  'b-o', markersize=3, label='Vanilla GD', linewidth=2)
    axes[0].semilogy(steps, loss_momentum, 'g-s', markersize=3, label='Momentum',   linewidth=2)
    axes[0].semilogy(steps, loss_adam,     'r-^', markersize=3, label='Adam',       linewidth=2)
    axes[0].set_xlabel('Iteration'); axes[0].set_ylabel('Loss (log scale)')
    axes[0].set_title('Convergence Comparison')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    # Path in parameter space
    w_hist_vanilla,  _ = vanilla_gradient_descent(start, lr=0.1, n_steps=n_steps)
    w_hist_momentum, _ = momentum_gradient_descent(start, lr=0.1, beta=0.9, n_steps=n_steps)
    w_hist_adam,     _ = adam_gradient_descent(start, lr=0.5, n_steps=n_steps)

    w1_grid = np.linspace(-1, 6, 100)
    w2_grid = np.linspace(-1, 8, 100)
    W1, W2  = np.meshgrid(w1_grid, w2_grid)
    Z       = (W1 - 3)**2 + (W2 - 5)**2

    ax = axes[1]
    ax.contourf(W1, W2, Z, levels=15, cmap='Blues', alpha=0.5)
    ax.contour( W1, W2, Z, levels=15, colors='gray', alpha=0.4, linewidths=0.5)

    for w_hist, label, color, marker in [
        (w_hist_vanilla,  'Vanilla GD', 'blue',   'o'),
        (w_hist_momentum, 'Momentum',   'green',  's'),
        (w_hist_adam,     'Adam',       'red',    '^'),
    ]:
        xs = [w[0] for w in w_hist]
        ys = [w[1] for w in w_hist]
        ax.plot(xs, ys, f'{color[0]}-', label=label, linewidth=2)
        ax.plot(xs[0], ys[0], f'{color[0]}{marker}', markersize=8)

    ax.plot(3, 5, 'k*', markersize=15, label='Minimum [3,5]', zorder=5)
    ax.set_xlabel('w₁'); ax.set_ylabel('w₂')
    ax.set_title('Optimization Paths in Parameter Space')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('optimizer_comparison.png', dpi=100)
    plt.close()
    print("\n  Saved: optimizer_comparison.png")
    print("\n  Key insight: Adam converges faster and more reliably.")
    print("  That's why it's the default for training all modern LLMs.")


def section_learning_rate_schedule():
    print("\n" + "=" * 55)
    print("SECTION 3: Learning Rate Schedule (Warmup + Cosine Decay)")
    print("=" * 55)
    print("This is the actual schedule used in GPT, LLaMA, etc.")

    total_steps   = 100
    warmup_steps  = 10
    lr_max        = 0.1
    lr_min        = 0.01

    def cosine_schedule(step):
        if step < warmup_steps:
            return lr_max * (step / warmup_steps)
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * progress))

    steps    = np.arange(total_steps + 1)
    lr_sched = np.array([cosine_schedule(s) for s in steps])

    # Run GD with scheduled lr
    w = np.array([0.0, 0.0])
    loss_hist_sched = [loss(w)]
    for step in range(total_steps):
        lr_now = cosine_schedule(step)
        g      = gradient(w)
        w      = w - lr_now * g
        loss_hist_sched.append(loss(w))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(steps, lr_sched, 'b-', linewidth=2)
    axes[0].axvspan(0, warmup_steps, alpha=0.15, color='green', label='warmup')
    axes[0].axvspan(warmup_steps, total_steps, alpha=0.1, color='orange', label='cosine decay')
    axes[0].set_xlabel('Step'); axes[0].set_ylabel('Learning Rate')
    axes[0].set_title('Warmup + Cosine Decay Schedule')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].semilogy(range(total_steps + 1), loss_hist_sched, 'r-', linewidth=2, label='Scheduled LR')
    axes[1].set_xlabel('Step'); axes[1].set_ylabel('Loss (log scale)')
    axes[1].set_title('Loss with Learning Rate Schedule')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('lr_schedule.png', dpi=100)
    plt.close()

    print(f"\n  Schedule parameters: warmup={warmup_steps} steps, max_lr={lr_max}, min_lr={lr_min}")
    print(f"  Step  0  (warmup start): lr = {cosine_schedule(0):.4f}")
    print(f"  Step  5  (mid warmup):   lr = {cosine_schedule(5):.4f}")
    print(f"  Step 10  (end warmup):   lr = {cosine_schedule(10):.4f}")
    print(f"  Step 50  (mid decay):    lr = {cosine_schedule(50):.4f}")
    print(f"  Step 100 (end):          lr = {cosine_schedule(100):.4f}")
    print("\n  Saved: lr_schedule.png")


if __name__ == '__main__':
    section_learning_rate_effect()
    section_comparing_optimizers()
    section_learning_rate_schedule()
    print("\nAll sections complete.")
    print("Generated: learning_rate_effect.png, optimizer_comparison.png, lr_schedule.png")
