"""
Module 02 — Code 01: Derivatives
==================================
Build intuition for derivatives and partial derivatives.
No ML framework needed — just numpy and math.

Run: python code_01_derivatives.py
"""

import math
import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# NUMERICAL DERIVATIVES
# =============================================================================

def derivative_forward(f, x, h=1e-5):
    """
    Forward difference: f'(x) ≈ (f(x+h) - f(x)) / h
    Simple but slightly inaccurate — the approximation has O(h) error.
    """
    return (f(x + h) - f(x)) / h


def derivative_centered(f, x, h=1e-5):
    """
    Centered difference: f'(x) ≈ (f(x+h) - f(x-h)) / (2h)
    More accurate — symmetric, O(h²) error.
    This is what you should use for gradient checking.
    """
    return (f(x + h) - f(x - h)) / (2 * h)


def section_basic_derivatives():
    print("=" * 55)
    print("SECTION 1: Numerical vs Analytical Derivatives")
    print("=" * 55)

    # f(x) = x²,  f'(x) = 2x
    f1 = lambda x: x**2
    f1_analytical = lambda x: 2 * x

    print("\nf(x) = x²  →  f'(x) = 2x")
    print(f"\n{'x':>6}  {'forward':>14}  {'centered':>14}  {'analytical':>12}  {'error_fwd':>10}")
    print("-" * 62)
    for x in [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]:
        fwd     = derivative_forward(f1, x)
        ctr     = derivative_centered(f1, x)
        exact   = f1_analytical(x)
        err_fwd = abs(fwd - exact)
        print(f"{x:>6.1f}  {fwd:>14.8f}  {ctr:>14.8f}  {exact:>12.4f}  {err_fwd:>10.2e}")

    # f(x) = sin(x),  f'(x) = cos(x)
    print("\n\nf(x) = sin(x)  →  f'(x) = cos(x)")
    print(f"\n{'x (radians)':>12}  {'centered':>14}  {'analytical':>12}  {'error':>10}")
    print("-" * 52)
    for x in [0, math.pi/6, math.pi/4, math.pi/2, math.pi]:
        ctr   = derivative_centered(math.sin, x)
        exact = math.cos(x)
        err   = abs(ctr - exact)
        print(f"{x:>12.4f}  {ctr:>14.8f}  {exact:>12.8f}  {err:>10.2e}")

    # f(x) = x³ - 2x + 1,  f'(x) = 3x² - 2
    print("\n\nf(x) = x³ - 2x + 1  →  f'(x) = 3x² - 2")
    f3 = lambda x: x**3 - 2*x + 1
    f3_analytical = lambda x: 3*x**2 - 2
    x = 2.0
    ctr   = derivative_centered(f3, x)
    exact = f3_analytical(x)
    print(f"  At x=2: centered = {ctr:.6f},  analytical = {exact:.6f}")
    print(f"  At x=0 (f'=−2, function is decreasing): {derivative_centered(f3, 0):.6f}")
    print(f"  At x=√(2/3)≈0.816 (f'=0, local minimum): {derivative_centered(f3, (2/3)**0.5):.6f}")


# =============================================================================
# ACTIVATION FUNCTION DERIVATIVES
# =============================================================================

def relu(x):
    return np.maximum(0, x)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def section_activation_derivatives():
    print("\n" + "=" * 55)
    print("SECTION 2: Activation Function Derivatives")
    print("=" * 55)
    print("These are used constantly in backprop.")

    x_vals = np.linspace(-4, 4, 200)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))

    # --- ReLU ---
    relu_vals     = relu(x_vals)
    relu_deriv    = (x_vals > 0).astype(float)   # analytical: 1 if x>0, else 0
    relu_num_deriv = np.array([derivative_centered(relu, x) for x in x_vals])

    ax = axes[0, 0]
    ax.plot(x_vals, relu_vals, 'b-', label='relu(x)', linewidth=2)
    ax.set_title("ReLU: f(x) = max(0, x)")
    ax.axhline(0, color='k', lw=0.5); ax.axvline(0, color='k', lw=0.5)
    ax.grid(True, alpha=0.3); ax.legend()

    ax = axes[0, 1]
    ax.plot(x_vals, relu_deriv, 'r-', label="relu'(x) = 0 or 1", linewidth=2)
    ax.set_title("ReLU Derivative")
    ax.set_ylim(-0.2, 1.4)
    ax.axhline(0, color='k', lw=0.5); ax.axvline(0, color='k', lw=0.5)
    ax.grid(True, alpha=0.3); ax.legend()

    # --- Sigmoid ---
    sig_vals     = sigmoid(x_vals)
    sig_deriv    = sig_vals * (1 - sig_vals)   # analytical: σ(x) * (1-σ(x))
    sig_num_deriv = np.array([derivative_centered(sigmoid, x) for x in x_vals])

    ax = axes[1, 0]
    ax.plot(x_vals, sig_vals, 'b-', label='σ(x)', linewidth=2)
    ax.set_title("Sigmoid: σ(x) = 1/(1+e^(-x))")
    ax.axhline(0, color='k', lw=0.5); ax.axvline(0, color='k', lw=0.5)
    ax.grid(True, alpha=0.3); ax.legend()

    ax = axes[1, 1]
    ax.plot(x_vals, sig_deriv, 'r-', label="σ'(x) = σ(1-σ)", linewidth=2)
    ax.plot(x_vals, sig_num_deriv, 'g--', label='numerical', alpha=0.7, linewidth=1.5)
    ax.set_title("Sigmoid Derivative (max at x=0 → 0.25)")
    ax.axhline(0, color='k', lw=0.5); ax.axvline(0, color='k', lw=0.5)
    ax.grid(True, alpha=0.3); ax.legend()

    plt.suptitle("Activation Functions and Their Derivatives", fontsize=12)
    plt.tight_layout()
    plt.savefig('activation_derivatives.png', dpi=100)
    plt.close()
    print("\nSaved: activation_derivatives.png")

    print("\nReLU derivative facts:")
    print("  - relu'(x) = 1 for x > 0  (gradient passes through unchanged)")
    print("  - relu'(x) = 0 for x < 0  (gradient is BLOCKED — 'dead neuron')")
    print("  - Dead neurons: if a neuron always outputs 0, it never learns")

    print("\nSigmoid derivative facts:")
    print("  - Maximum value is 0.25 at x=0")
    print("  - Near zero for large |x| → VANISHING GRADIENTS")
    print("  - This is why deep nets with sigmoid activations were hard to train")
    print("  - Modern LLMs use GELU/ReLU, not sigmoid")


# =============================================================================
# PARTIAL DERIVATIVES
# =============================================================================

def numerical_gradient(f, w, h=1e-5):
    """
    Compute gradient numerically: perturb each dimension independently.
    Returns a vector of partial derivatives.
    """
    grad = np.zeros_like(w, dtype=float)
    for i in range(len(w)):
        w_plus       = w.copy(); w_plus[i]  += h
        w_minus      = w.copy(); w_minus[i] -= h
        grad[i]      = (f(w_plus) - f(w_minus)) / (2 * h)
    return grad


def section_partial_derivatives():
    print("\n" + "=" * 55)
    print("SECTION 3: Partial Derivatives and the Gradient")
    print("=" * 55)

    # f(w1, w2) = (w1 - 3)² + (w2 - 5)²
    # ∂f/∂w1 = 2(w1 - 3),  ∂f/∂w2 = 2(w2 - 5)
    # Minimum at w = [3, 5]

    def f(w):
        return (w[0] - 3)**2 + (w[1] - 5)**2

    def analytical_gradient(w):
        return np.array([2*(w[0] - 3), 2*(w[1] - 5)])

    print("\nf(w1, w2) = (w1 - 3)² + (w2 - 5)²")
    print("∂f/∂w1 = 2(w1 - 3),  ∂f/∂w2 = 2(w2 - 5)")
    print("Minimum at w = [3, 5]  (where both partial derivatives = 0)")

    test_points = [
        np.array([0.0, 0.0]),
        np.array([3.0, 5.0]),    # at the minimum
        np.array([6.0, 5.0]),    # w1 too far right
        np.array([3.0, 0.0]),    # w2 too far down
    ]

    print(f"\n{'w':>14}  {'f(w)':>8}  {'grad (num)':>22}  {'grad (analyt)':>22}  match")
    print("-" * 82)
    for w in test_points:
        fval  = f(w)
        g_num = numerical_gradient(f, w)
        g_ana = analytical_gradient(w)
        ok    = np.allclose(g_num, g_ana, atol=1e-4)
        print(f"{str(w.tolist()):>14}  {fval:>8.2f}  {str(g_num.round(4).tolist()):>22}  "
              f"{str(g_ana.tolist()):>22}  {'✓' if ok else '✗'}")

    print("\nNote: at the minimum [3,5], gradient = [0, 0]  → no direction to improve")
    print("Gradient points uphill  →  -gradient points downhill  →  that's gradient descent")


if __name__ == '__main__':
    section_basic_derivatives()
    section_activation_derivatives()
    section_partial_derivatives()
    print("\nAll sections complete.")
