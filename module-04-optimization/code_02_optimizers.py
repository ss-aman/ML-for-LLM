"""
Module 04 — Code 02: Optimizers from Scratch
=============================================
Implement SGD, Momentum, RMSprop, Adam, and AdamW from scratch.
Compare their convergence on a challenging problem.

Run: python code_02_optimizers.py
"""

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# OPTIMIZER IMPLEMENTATIONS
# =============================================================================

class SGD:
    """Vanilla stochastic gradient descent."""

    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, params, grads):
        return params - self.lr * grads


class SGDMomentum:
    """
    SGD with momentum.
    v = β·v + g       (accumulate velocity)
    w = w - lr·v      (update with velocity)
    """

    def __init__(self, lr=0.01, beta=0.9):
        self.lr   = lr
        self.beta = beta
        self.v    = None

    def step(self, params, grads):
        if self.v is None:
            self.v = np.zeros_like(params)
        self.v = self.beta * self.v + grads         # NOT (1-beta)*grads — standard form
        return params - self.lr * self.v


class RMSprop:
    """
    Root Mean Square Propagation.
    s = β·s + (1-β)·g²    (EMA of squared gradients)
    w = w - lr·g/√(s+ε)   (adaptive update)
    """

    def __init__(self, lr=0.01, beta=0.999, eps=1e-8):
        self.lr   = lr
        self.beta = beta
        self.eps  = eps
        self.s    = None

    def step(self, params, grads):
        if self.s is None:
            self.s = np.zeros_like(params)
        self.s = self.beta * self.s + (1 - self.beta) * grads ** 2
        return params - self.lr * grads / (np.sqrt(self.s) + self.eps)


class Adam:
    """
    Adam: Adaptive Moment Estimation (Kingma & Ba, 2014)
    m = β1·m + (1-β1)·g         (first moment: momentum)
    v = β2·v + (1-β2)·g²        (second moment: adaptive)
    m̂ = m/(1-β1^t)              (bias correction)
    v̂ = v/(1-β2^t)              (bias correction)
    w = w - lr·m̂/√(v̂+ε)
    """

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr    = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps   = eps
        self.m     = None
        self.v     = None
        self.t     = 0

    def step(self, params, grads):
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
        self.t += 1
        self.m  = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v  = self.beta2 * self.v + (1 - self.beta2) * grads ** 2
        m_hat   = self.m / (1 - self.beta1 ** self.t)
        v_hat   = self.v / (1 - self.beta2 ** self.t)
        return params - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class AdamW:
    """
    AdamW: Adam with Decoupled Weight Decay (Loshchilov & Hutter, 2017)

    Difference from Adam:
      - Weight decay is applied SEPARATELY from the gradient update
      - w = w - lr·m̂/√(v̂+ε) - lr·λ·w   (last term is decoupled weight decay)
      - This ensures all parameters get the same relative regularization strength
        regardless of gradient magnitude (fixing the L2 regularization bug in Adam)
    """

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01):
        self.lr           = lr
        self.beta1        = beta1
        self.beta2        = beta2
        self.eps          = eps
        self.weight_decay = weight_decay
        self.m            = None
        self.v            = None
        self.t            = 0

    def step(self, params, grads):
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
        self.t += 1

        # Gradient-based update (same as Adam, no weight decay here)
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * grads ** 2
        m_hat  = self.m / (1 - self.beta1 ** self.t)
        v_hat  = self.v / (1 - self.beta2 ** self.t)
        params = params - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

        # Decoupled weight decay (separate from gradient update)
        params = params - self.lr * self.weight_decay * params

        return params


# =============================================================================
# SECTION 1: Convergence on the Elongated Bowl
# =============================================================================

def section_elongated_bowl():
    print("=" * 55)
    print("SECTION 1: Convergence on Elongated Bowl")
    print("=" * 55)
    print("f(w) = w1² + 100·w2²  (high curvature ratio = 100)")

    # f(w) = w1^2 + 100*w2^2
    # Gradient: [2*w1, 200*w2]
    def loss(w):
        return float(w[0]**2 + 100 * w[1]**2)

    def gradient(w):
        return np.array([2 * w[0], 200 * w[1]])

    w_start  = np.array([4.0, 0.5])
    n_steps  = 200

    optimizers = [
        ("SGD (lr=0.001)",        SGD(lr=0.001)),
        ("Momentum (β=0.9)",      SGDMomentum(lr=0.001, beta=0.9)),
        ("RMSprop",               RMSprop(lr=0.01, beta=0.999)),
        ("Adam",                  Adam(lr=0.05, beta1=0.9, beta2=0.999)),
        ("AdamW (wd=0.01)",       AdamW(lr=0.05, beta1=0.9, beta2=0.999, weight_decay=0.01)),
    ]

    colors = ['red', 'orange', 'green', 'blue', 'purple']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    print(f"\n{'Optimizer':25s}  {'Final loss':>14}  {'Steps to 1%':>12}")
    print("-" * 55)

    for (name, opt), color in zip(optimizers, colors):
        w          = w_start.copy()
        losses_run = [loss(w)]
        paths      = [w.copy()]

        for _ in range(n_steps):
            g = gradient(w)
            w = opt.step(w, g)
            losses_run.append(loss(w))
            paths.append(w.copy())

        paths = np.array(paths)

        # Steps to reach 1% of initial loss
        initial   = losses_run[0]
        threshold = initial * 0.01
        steps_to  = next((i for i, l in enumerate(losses_run) if l < threshold), n_steps)

        print(f"  {name:25s}  {losses_run[-1]:>14.6f}  {steps_to:>12}")

        axes[0].semilogy(losses_run, color=color, label=name, linewidth=2)
        axes[1].plot(paths[:, 0], paths[:, 1], color=color,
                     marker='o', markersize=2, linewidth=1.5, label=name, alpha=0.8)

    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Loss (log scale)')
    axes[0].set_title('Convergence on Elongated Bowl')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    w1r = np.linspace(-4.5, 4.5, 200)
    w2r = np.linspace(-0.6, 0.6, 200)
    W1, W2 = np.meshgrid(w1r, w2r)
    Z      = W1**2 + 100 * W2**2
    axes[1].contourf(W1, W2, Z, levels=20, cmap='Blues', alpha=0.5)
    axes[1].plot(0, 0, 'k*', markersize=12, zorder=5, label='Minimum')
    axes[1].set_xlabel('w₁')
    axes[1].set_ylabel('w₂')
    axes[1].set_title('Optimization Paths')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('optimizer_comparison.png', dpi=100)
    plt.close()
    print("\nSaved: optimizer_comparison.png")


# =============================================================================
# SECTION 2: Adam Bias Correction — Why It Matters
# =============================================================================

def section_bias_correction():
    print("\n" + "=" * 55)
    print("SECTION 2: Adam Bias Correction at Early Steps")
    print("=" * 55)

    # Simulate the moment estimates at early steps
    grad_true = 1.0   # constant true gradient
    beta1     = 0.9
    beta2     = 0.999

    m, v  = 0.0, 0.0
    print("\nStep-by-step: m_raw vs m_corrected")
    print(f"  True gradient = {grad_true}, β1 = {beta1}")
    print(f"\n  {'Step':>5}  {'m_raw':>10}  {'m_corrected':>13}  {'Correction factor':>18}")
    print("  " + "-" * 52)

    for t in range(1, 8):
        m = beta1 * m + (1 - beta1) * grad_true
        v = beta2 * v + (1 - beta2) * grad_true**2
        m_hat        = m / (1 - beta1**t)
        correction   = 1 / (1 - beta1**t)
        print(f"  {t:>5}  {m:>10.6f}  {m_hat:>13.6f}  {correction:>18.4f}x")

    print(f"\n  At step 1: m_raw = {0.1:.4f} (heavily biased toward 0)")
    print(f"  After correction: m_hat ≈ {grad_true:.4f} (matches true gradient)")
    print(f"  By step 7: correction factor approaches 1.0 (correction fades)")

    print("""
Without bias correction:
  - Early steps are ~10x too small
  - Model barely moves in the first ~100 steps
  - Training appears "stuck" at the start

With bias correction:
  - First step uses the correct scale
  - Training starts learning immediately
""")


# =============================================================================
# SECTION 3: Adam vs AdamW — Weight Decay Difference
# =============================================================================

def section_adamw_vs_adam():
    print("=" * 55)
    print("SECTION 3: Adam vs AdamW — Weight Decay Effect")
    print("=" * 55)

    print("""
Scenario: a parameter that has been updated many times (large v)
vs a parameter that's rarely updated (small v).

With Adam + L2 regularization (wrong):
  Weight decay goes through adaptive scaling 1/√v
  → Frequently updated params: large v → small effective decay
  → Rarely updated params:     small v → large effective decay
  → Regularization is inconsistent across parameters

With AdamW (correct):
  Weight decay is applied directly: w -= lr·λ·w
  → Same relative decay for ALL parameters regardless of gradient history
""")

    # Demonstrate the difference numerically
    lambda_wd = 0.1
    lr        = 0.001
    grad      = np.array([1.0, 1.0])   # same gradient

    # Two parameters with different gradient histories
    v_high = 100.0   # param 1: updated many times (large accumulated v)
    v_low  = 0.01    # param 2: rarely updated (small v)

    # Adam with L2: weight decay goes through adaptive scaling
    adam_decay_1 = lr * lambda_wd / np.sqrt(v_high + 1e-8)   # param 1
    adam_decay_2 = lr * lambda_wd / np.sqrt(v_low  + 1e-8)   # param 2

    # AdamW: weight decay applied directly
    adamw_decay_1 = lr * lambda_wd    # param 1
    adamw_decay_2 = lr * lambda_wd    # param 2 (same!)

    print(f"Effective weight decay per parameter (lr={lr}, λ={lambda_wd}):")
    print(f"\n  {'':25s}  {'Param 1 (v=100)':>18}  {'Param 2 (v=0.01)':>18}  {'Ratio':>8}")
    print("  " + "-" * 75)
    print(f"  {'Adam+L2 (wrong):':25s}  {adam_decay_1:>18.6f}  {adam_decay_2:>18.6f}  {adam_decay_2/adam_decay_1:>8.1f}x")
    print(f"  {'AdamW (correct):':25s}  {adamw_decay_1:>18.6f}  {adamw_decay_2:>18.6f}  {adamw_decay_2/adamw_decay_1:>8.1f}x")

    print(f"""
  Adam+L2: Param 2 gets {adam_decay_2/adam_decay_1:.0f}x more weight decay than Param 1!
  → Inconsistent regularization: rarely-updated params are over-regularized

  AdamW: Both params get exactly the same weight decay ({adamw_decay_1:.4f})
  → Consistent regularization: all params equally controlled
""")


# =============================================================================
# SECTION 4: Convergence on Noisy Problem (LLM-Like)
# =============================================================================

def section_noisy_convergence():
    print("=" * 55)
    print("SECTION 4: Optimizer Comparison on Noisy Quadratic")
    print("=" * 55)

    print("Simulates training dynamics: true gradient + stochastic noise")

    np.random.seed(42)
    n_params = 100    # number of parameters
    n_steps  = 500

    # True minimum is at all-zeros
    # Gradient = 2*w + noise (noisy gradient)
    def true_loss(w):
        return float(np.sum(w**2))

    def noisy_gradient(w, noise_std=0.3):
        return 2 * w + np.random.normal(0, noise_std, size=w.shape)

    optimizers = [
        ("SGD (lr=0.01)",    SGD(lr=0.01)),
        ("Momentum",         SGDMomentum(lr=0.01, beta=0.9)),
        ("Adam",             Adam(lr=0.1, beta1=0.9, beta2=0.95)),
        ("AdamW (wd=0.01)",  AdamW(lr=0.1, beta1=0.9, beta2=0.95, weight_decay=0.01)),
    ]
    colors = ['red', 'orange', 'blue', 'purple']

    print(f"\n{'Optimizer':25s}  {'Initial loss':>14}  {'Final loss':>12}  {'Steps to 1%':>12}")
    print("-" * 68)

    plt.figure(figsize=(9, 4))
    for (name, opt), color in zip(optimizers, colors):
        np.random.seed(0)
        w          = np.random.randn(n_params)
        losses_run = [true_loss(w)]

        for _ in range(n_steps):
            g = noisy_gradient(w)
            w = opt.step(w, g)
            losses_run.append(true_loss(w))

        threshold = losses_run[0] * 0.01
        steps_to  = next((i for i, l in enumerate(losses_run) if l < threshold), n_steps)
        print(f"  {name:25s}  {losses_run[0]:>14.2f}  {losses_run[-1]:>12.4f}  {steps_to:>12}")

        plt.semilogy(losses_run, color=color, label=name, linewidth=2, alpha=0.85)

    plt.xlabel('Step')
    plt.ylabel('True Loss (log scale)')
    plt.title('Optimizer Convergence on Noisy Quadratic (100 params)')
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('noisy_convergence.png', dpi=100)
    plt.close()
    print("\nSaved: noisy_convergence.png")
    print("""
Key observations:
  - Adam/AdamW converge faster due to adaptive per-parameter learning rates
  - Momentum helps vs plain SGD but lacks the adaptive component
  - AdamW with weight decay adds regularization (useful for generalization)
  - All optimizers converge to similar final loss on this simple problem
""")


# =============================================================================
# RUN ALL SECTIONS
# =============================================================================

if __name__ == '__main__':
    section_elongated_bowl()
    section_bias_correction()
    section_adamw_vs_adam()
    section_noisy_convergence()

    print("\n" + "=" * 55)
    print("Key takeaways:")
    print("  1. Momentum: accumulates velocity, reduces oscillation")
    print("  2. RMSprop: adapts per-parameter, handles scale mismatch")
    print("  3. Adam: both simultaneously, with bias correction")
    print("  4. AdamW: fixed weight decay (what all LLMs use)")
    print("  5. Default LLM config: β1=0.9, β2=0.95, ε=1e-8, wd=0.1")
    print("=" * 55)
