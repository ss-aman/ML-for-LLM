"""
Module 02: Calculus and Gradients — Python Implementations
===========================================================
Run this file directly:  python code.py

Each section demonstrates a core calculus concept with intuitive comments
aimed at Python/backend developers.

The key theme: derivatives and gradients tell us *which direction to move*
to minimize a function — which is how neural networks learn.
"""

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# SECTION 1: Numerical Derivative Approximation
#
# We don't need symbolic math to compute derivatives.
# The "finite difference" method estimates the derivative by measuring
# what happens when we nudge x by a tiny amount h.
#
# f'(x) ≈ (f(x+h) - f(x)) / h
#
# Backend analogy: this is like A/B testing a config change.
# "If I increase timeout by 1ms, what happens to error rate?"
# =============================================================================

def numerical_derivative(f, x, h=1e-5):
    """
    Estimate f'(x) using the finite difference method.

    f'(x) ≈ (f(x+h) - f(x)) / h

    Smaller h = more accurate, but too small causes floating-point errors.
    h=1e-5 is a good practical default.
    """
    return (f(x + h) - f(x)) / h


def demo_numerical_derivative():
    """
    Compare numerical derivatives to analytical (exact) derivatives.
    """
    print("=== Numerical Derivative Demo ===")

    # Example 1: f(x) = x^2
    # Analytical: f'(x) = 2x
    f = lambda x: x ** 2
    f_prime_analytical = lambda x: 2 * x

    for x in [0, 1, 2, 3, -1.5]:
        numerical = numerical_derivative(f, x)
        analytical = f_prime_analytical(x)
        error = abs(numerical - analytical)
        print(f"  x={x:5.1f}: numerical={numerical:.6f}  analytical={analytical:.6f}  error={error:.2e}")

    print()

    # Example 2: f(x) = sin(x)
    # Analytical: f'(x) = cos(x)
    print("  f(x) = sin(x), f'(x) = cos(x):")
    import math
    f2 = math.sin
    for x in [0, math.pi / 4, math.pi / 2]:
        numerical = numerical_derivative(f2, x)
        analytical = math.cos(x)
        print(f"  x={x:.4f}: numerical={numerical:.6f}  analytical={analytical:.6f}")


# =============================================================================
# SECTION 2: Gradient of a Multi-Variable Loss Function
#
# The gradient is the vector of all partial derivatives.
# For a function with 2 parameters w1 and w2:
#   ∇L = [ ∂L/∂w1, ∂L/∂w2 ]
#
# Our example loss function:
#   L(w1, w2) = (w1 - 3)^2 + (w2 - 5)^2
#
# This is a bowl-shaped surface with a minimum at (w1=3, w2=5).
# The gradient at any point (w1, w2) is:
#   ∇L = [ 2*(w1-3), 2*(w2-5) ]
#
# Backend analogy: if L is latency and (w1, w2) are config parameters
# (thread_pool_size, cache_ttl), the gradient tells you how each
# parameter affects latency at the current setting.
# =============================================================================

def loss_function(w):
    """
    L(w1, w2) = (w1 - 3)^2 + (w2 - 5)^2
    Minimum is at w = [3, 5] where L = 0.
    """
    w1, w2 = w[0], w[1]
    return (w1 - 3) ** 2 + (w2 - 5) ** 2


def analytical_gradient(w):
    """
    Exact gradient of our loss function:
    ∂L/∂w1 = 2*(w1 - 3)
    ∂L/∂w2 = 2*(w2 - 5)
    """
    w1, w2 = w[0], w[1]
    return np.array([2 * (w1 - 3), 2 * (w2 - 5)])


def numerical_gradient(f, w, h=1e-5):
    """
    Compute the gradient numerically by perturbing each dimension.

    For each parameter w_i, compute:
        ∂f/∂w_i ≈ (f(w + h*e_i) - f(w)) / h
    where e_i is the unit vector in the i-th direction.

    This is how automatic differentiation works conceptually —
    though in practice PyTorch uses backpropagation (chain rule),
    not finite differences.
    """
    grad = np.zeros_like(w, dtype=float)
    for i in range(len(w)):
        w_plus = w.copy()
        w_plus[i] += h
        grad[i] = (f(w_plus) - f(w)) / h
    return grad


def demo_gradient():
    """Compare numerical and analytical gradients."""
    print("\n=== Gradient Demo ===")
    print("Loss: L(w1, w2) = (w1-3)^2 + (w2-5)^2")
    print("Minimum at [3, 5]")

    test_points = [
        np.array([0.0, 0.0]),
        np.array([3.0, 5.0]),   # at the minimum
        np.array([6.0, 8.0]),
        np.array([1.5, 2.0]),
    ]

    for w in test_points:
        loss = loss_function(w)
        grad_analytical = analytical_gradient(w)
        grad_numerical = numerical_gradient(loss_function, w)
        match = np.allclose(grad_analytical, grad_numerical, atol=1e-4)
        print(f"\n  w = {w}  →  Loss = {loss:.4f}")
        print(f"    analytical gradient: {grad_analytical}")
        print(f"    numerical gradient:  {grad_numerical.round(6)}")
        print(f"    match: {match}")


# =============================================================================
# SECTION 3: Gradient Descent Loop
#
# The core training algorithm:
#   w = w - learning_rate * ∇Loss(w)
#
# Repeat until convergence. Watch the loss decrease toward zero.
#
# Backend analogy: like auto-tuning server config by:
#   1. Measure current performance metric (compute loss)
#   2. Try tiny changes to each parameter (compute gradient)
#   3. Move all parameters in the direction that improves performance
#   4. Repeat until stable
# =============================================================================

def gradient_descent(
    loss_fn,
    grad_fn,
    w_init,
    learning_rate=0.1,
    n_iterations=50
):
    """
    Run gradient descent starting from w_init.

    Args:
        loss_fn: the function to minimize, takes w as input
        grad_fn: function to compute the gradient of loss_fn
        w_init: starting parameter values
        learning_rate: step size (how aggressively to move)
        n_iterations: how many steps to take

    Returns:
        w_history: list of parameter values at each step
        loss_history: list of loss values at each step
    """
    w = np.array(w_init, dtype=float)
    w_history = [w.copy()]
    loss_history = [loss_fn(w)]

    for step in range(n_iterations):
        grad = grad_fn(w)
        w = w - learning_rate * grad          # the core update rule
        w_history.append(w.copy())
        loss_history.append(loss_fn(w))

    return w_history, loss_history


def demo_gradient_descent():
    """Run gradient descent on our bowl-shaped loss and plot convergence."""
    print("\n=== Gradient Descent Demo ===")

    w_start = np.array([0.0, 0.0])
    w_history, loss_history = gradient_descent(
        loss_fn=loss_function,
        grad_fn=analytical_gradient,
        w_init=w_start,
        learning_rate=0.2,
        n_iterations=30
    )

    print(f"Start:     w = {w_history[0]},   Loss = {loss_history[0]:.4f}")
    print(f"After 10:  w = {w_history[10].round(4)}, Loss = {loss_history[10]:.6f}")
    print(f"After 20:  w = {w_history[20].round(4)}, Loss = {loss_history[20]:.8f}")
    print(f"Final:     w = {w_history[-1].round(6)}, Loss = {loss_history[-1]:.2e}")
    print(f"Expected minimum: w = [3, 5]")

    # ----- Plot 1: Loss over iterations -----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(loss_history, 'b-o', markersize=4, linewidth=2)
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Loss L(w1, w2)')
    axes[0].set_title('Gradient Descent: Loss over Iterations')
    axes[0].set_yscale('log')
    axes[0].grid(True, alpha=0.3)
    axes[0].annotate(
        f'Converges to\nw≈[3,5], L≈0',
        xy=(len(loss_history) - 1, loss_history[-1]),
        xytext=(20, loss_history[5]),
        arrowprops=dict(arrowstyle='->', color='black'),
        fontsize=9
    )

    # ----- Plot 2: Path in parameter space -----
    w1_vals = [w[0] for w in w_history]
    w2_vals = [w[1] for w in w_history]

    # Draw the loss surface as a contour
    w1_grid = np.linspace(-1, 6, 100)
    w2_grid = np.linspace(-1, 8, 100)
    W1, W2 = np.meshgrid(w1_grid, w2_grid)
    Z = (W1 - 3) ** 2 + (W2 - 5) ** 2

    axes[1].contour(W1, W2, Z, levels=20, cmap='Blues', alpha=0.6)
    axes[1].plot(w1_vals, w2_vals, 'r-o', markersize=5, linewidth=2, label='Descent path')
    axes[1].plot(3, 5, 'g*', markersize=15, label='Minimum [3,5]')
    axes[1].plot(w1_vals[0], w2_vals[0], 'bs', markersize=10, label='Start [0,0]')
    axes[1].set_xlabel('w1')
    axes[1].set_ylabel('w2')
    axes[1].set_title('Gradient Descent Path in Parameter Space')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('gradient_descent.png', dpi=100)
    plt.close()
    print("Saved: gradient_descent.png")


# =============================================================================
# SECTION 4: Chain Rule Demo
#
# The chain rule: d/dx [f(g(x))] = f'(g(x)) * g'(x)
#
# We demonstrate this for h(x) = sin(x^2):
#   g(x) = x^2           g'(x) = 2x
#   f(u) = sin(u)         f'(u) = cos(u)
#   h'(x) = cos(x^2) * 2x
#
# Backend analogy: tracing how a latency spike in service C propagates
# back through service B and service A. The total sensitivity of end-to-end
# latency to a change in C is the product of sensitivities through each hop.
# =============================================================================

def demo_chain_rule():
    """
    Verify the chain rule numerically for h(x) = sin(x^2).
    """
    import math

    print("\n=== Chain Rule Demo: h(x) = sin(x^2) ===")
    print("Analytical: h'(x) = cos(x^2) * 2x")

    h = lambda x: math.sin(x ** 2)
    h_prime_analytical = lambda x: math.cos(x ** 2) * 2 * x

    print(f"\n{'x':>6}  {'numerical h\'(x)':>18}  {'analytical h\'(x)':>18}  {'error':>12}")
    print("-" * 62)
    for x in [0.5, 1.0, 1.5, 2.0, -1.0]:
        numerical = numerical_derivative(h, x)
        analytical = h_prime_analytical(x)
        error = abs(numerical - analytical)
        print(f"{x:>6.2f}  {numerical:>18.8f}  {analytical:>18.8f}  {error:>12.2e}")

    print("\nChain rule works — numerical and analytical derivatives match closely.")
    print("This same principle, applied layer by layer, is backpropagation.")


# =============================================================================
# SECTION 5: Effect of Learning Rate
#
# Shows how learning rate affects convergence.
# Too large → oscillates or diverges.
# Too small → converges slowly.
# =============================================================================

def demo_learning_rate_effect():
    """
    Run gradient descent with different learning rates and compare convergence.
    """
    print("\n=== Learning Rate Effect Demo ===")

    learning_rates = [0.01, 0.1, 0.5, 1.1]   # 1.1 is too large — will diverge
    colors = ['blue', 'green', 'orange', 'red']
    labels = ['0.01 (too slow)', '0.1 (good)', '0.5 (fast)', '1.1 (diverges!)']

    plt.figure(figsize=(8, 5))
    for lr, color, label in zip(learning_rates, colors, labels):
        _, loss_history = gradient_descent(
            loss_function,
            analytical_gradient,
            w_init=[0.0, 0.0],
            learning_rate=lr,
            n_iterations=40
        )
        # Cap display at 200 for readability
        display = [min(l, 200) for l in loss_history]
        plt.plot(display, color=color, label=f'lr={label}', linewidth=2)
        final_loss = loss_history[-1]
        print(f"  lr={lr:.2f}: final loss = {final_loss:.4f} ({'DIVERGED' if final_loss > 100 else 'OK'})")

    plt.xlabel('Iteration')
    plt.ylabel('Loss (capped at 200)')
    plt.title('Effect of Learning Rate on Convergence')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('learning_rate_effect.png', dpi=100)
    plt.close()
    print("Saved: learning_rate_effect.png")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("MODULE 02: Calculus and Gradients Demos")
    print("=" * 60)

    demo_numerical_derivative()
    demo_gradient()
    demo_gradient_descent()
    demo_chain_rule()
    demo_learning_rate_effect()

    print("\n" + "=" * 60)
    print("All demos complete. Check the generated PNG files.")
    print("  gradient_descent.png     — loss curve + path in parameter space")
    print("  learning_rate_effect.png — how lr choice affects convergence")
    print("=" * 60)
