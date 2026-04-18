"""
Module 02: Calculus and Gradients — Exercises
==============================================
Run this file directly:  python exercise.py

These exercises build hands-on intuition for derivatives, gradients,
and gradient descent. Each exercise is a function with a clear docstring.
The check() function at the bottom runs all exercises and prints results.

Target: someone who knows Python well but has never studied ML math.
"""

import math
import numpy as np


# =============================================================================
# EXERCISE 1: Numerical Gradient for f(x) = x^3 - 2x + 1
#
# Implement the finite difference method to estimate the derivative,
# then verify it matches the analytical derivative.
# =============================================================================

def exercise_1_numerical_gradient(x: float, h: float = 1e-5) -> float:
    """
    Compute the numerical derivative of f(x) = x^3 - 2x + 1 at point x.

    Use the finite difference formula:
        f'(x) ≈ (f(x + h) - f(x)) / h

    Then verify against the analytical derivative:
        f'(x) = 3x^2 - 2    (power rule: d/dx[x^3] = 3x^2, d/dx[-2x] = -2, d/dx[1] = 0)

    Backend analogy: this is like measuring how API error rate changes
    when you increase concurrency by a tiny amount (h). You don't need
    to know the formula — just measure the effect.

    Args:
        x: the point at which to evaluate the derivative
        h: the step size for finite differences (default: 1e-5)

    Returns:
        The estimated derivative f'(x) as a float.

    TODO: Define f(x), then apply the finite difference formula.
    """
    # Define the function
    def f(x):
        return x ** 3 - 2 * x + 1

    # YOUR CODE HERE: apply finite difference formula
    return (f(x + h) - f(x)) / h


def analytical_derivative_ex1(x: float) -> float:
    """Exact derivative: f'(x) = 3x^2 - 2"""
    return 3 * x ** 2 - 2


# =============================================================================
# EXERCISE 2: Gradient Descent on f(x,y) = x^2 + y^2 + 2x
#
# Find the minimum by repeatedly stepping in the direction of -gradient.
# The minimum is at x = -1, y = 0 (where the gradient = [0, 0]).
# =============================================================================

def exercise_2_gradient_descent(
    starting_point: list,
    learning_rate: float = 0.1,
    n_iterations: int = 100
) -> tuple:
    """
    Run gradient descent on f(x, y) = x^2 + y^2 + 2x.

    Analytical gradient:
        ∂f/∂x = 2x + 2
        ∂f/∂y = 2y

    Setting gradient to zero:
        2x + 2 = 0  →  x = -1
        2y = 0       →  y = 0
    Minimum is at (-1, 0), where f(-1, 0) = 1 + 0 - 2 = -1.

    Backend analogy: imagine (x, y) are two config parameters (e.g.,
    timeout_ms and retry_count), and f is the average error rate.
    Gradient descent automatically finds the optimal settings.

    Args:
        starting_point: [x0, y0] initial parameter values
        learning_rate: step size (try 0.1)
        n_iterations: number of steps to take

    Returns:
        (final_x, final_y): the found minimum as a tuple of floats.

    TODO: Implement the gradient descent update loop.
          At each step: compute gradient, then update: w = w - lr * grad
    """
    x, y = float(starting_point[0]), float(starting_point[1])

    # YOUR CODE HERE
    for _ in range(n_iterations):
        # Gradient of f(x, y) = x^2 + y^2 + 2x
        grad_x = 2 * x + 2   # ∂f/∂x
        grad_y = 2 * y        # ∂f/∂y

        # Gradient descent update
        x = x - learning_rate * grad_x
        y = y - learning_rate * grad_y

    return (x, y)


# =============================================================================
# EXERCISE 3 (Challenge): Gradient Descent with Momentum
#
# Standard gradient descent can be slow in narrow valleys or noisy gradients.
# Momentum adds a "velocity" term that accumulates past gradient directions,
# like a ball rolling downhill picking up speed.
#
# Update rule:
#   velocity = beta * velocity - learning_rate * gradient
#   weights  = weights + velocity
#
# Backend analogy: like TCP slow start, but in reverse — you accelerate
# toward the minimum when gradients consistently point the same direction,
# rather than starting slow and backing off.
# =============================================================================

def exercise_3_momentum_gradient_descent(
    starting_point: list,
    learning_rate: float = 0.1,
    beta: float = 0.9,
    n_iterations: int = 50
) -> tuple:
    """
    Run gradient descent WITH momentum on f(x, y) = x^2 + y^2 + 2x.

    Momentum update rule:
        velocity = beta * velocity - learning_rate * gradient
        weights  = weights + velocity

    beta (0.9 is typical): controls how much of the previous velocity to keep.
    A higher beta means more momentum — past gradients influence current step more.

    The same loss function as Exercise 2:
        f(x, y) = x^2 + y^2 + 2x
        ∂f/∂x = 2x + 2,  ∂f/∂y = 2y
        Minimum at (-1, 0)

    Backend analogy: TCP slow start increases the congestion window
    multiplicatively when things go well. Momentum similarly "accelerates"
    when the gradient consistently points in the same direction, letting
    you converge faster than plain gradient descent.

    Args:
        starting_point: [x0, y0] initial parameter values
        learning_rate: step size
        beta: momentum coefficient (0 = no momentum, 1 = pure inertia)
        n_iterations: number of update steps

    Returns:
        (final_x, final_y): the found minimum as a tuple of floats.
        Should converge faster than exercise_2 with the same learning_rate.

    TODO: Add a velocity vector (starts at zero), update it with momentum,
          then add it to the weights each step.
    """
    x, y = float(starting_point[0]), float(starting_point[1])

    # Initialize velocity at zero
    vx, vy = 0.0, 0.0

    # YOUR CODE HERE
    for _ in range(n_iterations):
        # Compute gradient
        grad_x = 2 * x + 2
        grad_y = 2 * y

        # Update velocity with momentum
        vx = beta * vx - learning_rate * grad_x
        vy = beta * vy - learning_rate * grad_y

        # Update weights using velocity
        x = x + vx
        y = y + vy

    return (x, y)


# =============================================================================
# CHECK FUNCTION — runs all exercises and prints results
# =============================================================================

def check():
    print("=" * 65)
    print("MODULE 02 EXERCISES — Results")
    print("=" * 65)

    # ---- Exercise 1 ----
    print("\n[Exercise 1] Numerical gradient for f(x) = x^3 - 2x + 1")
    print(f"  Analytical derivative: f'(x) = 3x^2 - 2")
    all_pass = True
    test_xs = [0.0, 1.0, 2.0, -1.0, 3.0]
    print(f"\n  {'x':>6}  {'numerical':>14}  {'analytical':>14}  {'error':>12}  result")
    print(f"  {'-'*6}  {'-'*14}  {'-'*14}  {'-'*12}  ------")
    for x in test_xs:
        numerical = exercise_1_numerical_gradient(x)
        analytical = analytical_derivative_ex1(x)
        error = abs(numerical - analytical)
        ok = error < 1e-3
        print(f"  {x:>6.2f}  {numerical:>14.8f}  {analytical:>14.8f}  {error:>12.2e}  {'PASS' if ok else 'FAIL'}")
        if not ok:
            all_pass = False
    print(f"\n  Exercise 1: {'ALL PASS' if all_pass else 'SOME FAILURES'}")

    # ---- Exercise 2 ----
    print("\n[Exercise 2] Gradient descent on f(x,y) = x^2 + y^2 + 2x")
    print("  Expected minimum: (-1.0, 0.0), where f(-1,0) = -1.0")
    start = [5.0, 5.0]
    result_x, result_y = exercise_2_gradient_descent(start, learning_rate=0.1, n_iterations=100)
    f_at_min = result_x**2 + result_y**2 + 2*result_x
    expected_x, expected_y = -1.0, 0.0
    ok_x = math.isclose(result_x, expected_x, abs_tol=1e-4)
    ok_y = math.isclose(result_y, expected_y, abs_tol=1e-4)
    print(f"  Start:  ({start[0]}, {start[1]})")
    print(f"  Found:  ({result_x:.6f}, {result_y:.6f})")
    print(f"  f at found point: {f_at_min:.6f} (expected -1.0)")
    print(f"  x converged? {'PASS' if ok_x else 'FAIL'}  |  y converged? {'PASS' if ok_y else 'FAIL'}")

    # ---- Exercise 3 ----
    print("\n[Exercise 3] Momentum gradient descent on same function")
    print("  Expected: same minimum (-1.0, 0.0), but converges faster")
    # Compare iterations needed for both methods
    convergence_results = {}
    for label, fn, kwargs in [
        ("plain GD",    exercise_2_gradient_descent,  {"learning_rate": 0.1}),
        ("with momentum", exercise_3_momentum_gradient_descent, {"learning_rate": 0.1, "beta": 0.9}),
    ]:
        losses_at_step = []
        x_cur, y_cur = 5.0, 5.0
        for n in [5, 10, 20, 50]:
            rx, ry = fn([5.0, 5.0], n_iterations=n, **kwargs)
            loss = rx**2 + ry**2 + 2*rx
            losses_at_step.append((n, loss))
        convergence_results[label] = losses_at_step

    print(f"\n  {'Iterations':>12}  {'Plain GD loss':>16}  {'Momentum loss':>16}")
    print(f"  {'-'*12}  {'-'*16}  {'-'*16}")
    for (n_plain, loss_plain), (n_mom, loss_mom) in zip(
        convergence_results["plain GD"],
        convergence_results["with momentum"]
    ):
        faster = "← faster" if loss_mom < loss_plain else ""
        print(f"  {n_plain:>12}  {loss_plain:>16.8f}  {loss_mom:>16.8f}  {faster}")

    mx, my = exercise_3_momentum_gradient_descent([5.0, 5.0], learning_rate=0.1, beta=0.9, n_iterations=50)
    ok_mx = math.isclose(mx, -1.0, abs_tol=1e-3)
    ok_my = math.isclose(my, 0.0, abs_tol=1e-3)
    print(f"\n  Final: ({mx:.6f}, {my:.6f})")
    print(f"  x correct? {'PASS' if ok_mx else 'FAIL'}  |  y correct? {'PASS' if ok_my else 'FAIL'}")
    print(f"\n  Key insight: momentum accumulates velocity in a consistent direction,")
    print(f"  so it converges faster — especially in elongated loss surfaces.")

    print("\n" + "=" * 65)
    print("Tip: Try changing the learning_rate and beta in Exercise 3 to see")
    print("how they affect convergence speed.")
    print("=" * 65)


if __name__ == '__main__':
    check()
