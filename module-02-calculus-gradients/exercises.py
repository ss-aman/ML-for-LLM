"""
Module 02 — Exercises: Calculus & Gradients
=============================================
Implement each function. Do NOT look at solutions.py until you've tried.

Run: python exercises.py
"""

import math
import numpy as np


# =============================================================================
# EXERCISE 1: Numerical Derivative
# Difficulty: Easy
# =============================================================================

def numerical_derivative(f, x: float, h: float = 1e-5) -> float:
    """
    Estimate the derivative of f at point x using the centered difference:

        f'(x) ≈ (f(x+h) - f(x-h)) / (2h)

    This is more accurate than the forward difference (f(x+h) - f(x)) / h.

    Example:
        f = lambda x: x**2
        numerical_derivative(f, 3.0)  → ~6.0   (exact: 2*3 = 6)
        numerical_derivative(f, -2.0) → ~-4.0  (exact: 2*(-2) = -4)
    """
    # TODO: one line using the centered difference formula
    raise NotImplementedError


# =============================================================================
# EXERCISE 2: Numerical Gradient
# Difficulty: Easy-Medium
# =============================================================================

def numerical_gradient(f, w: np.ndarray, h: float = 1e-5) -> np.ndarray:
    """
    Compute the gradient of f at w numerically.

    For each dimension i, perturb only w[i] and compute the partial derivative:
        ∂f/∂wᵢ ≈ (f(w + h*eᵢ) - f(w - h*eᵢ)) / (2h)
    where eᵢ is a vector with 1 at position i, 0 elsewhere.

    Args:
        f: function that takes a numpy array and returns a scalar
        w: numpy array of parameters (do not modify in place!)
        h: step size for finite difference

    Returns:
        grad: numpy array, same shape as w, containing all partial derivatives

    Example:
        f = lambda w: (w[0]-3)**2 + (w[1]-5)**2
        numerical_gradient(f, np.array([0., 0.]))
        → approximately [-6., -10.]   (exact: [2*(0-3), 2*(0-5)])
    """
    grad = np.zeros_like(w, dtype=float)
    for i in range(len(w)):
        # TODO: compute grad[i] using centered difference
        # Hint: create w_plus and w_minus as copies of w, modify only index i
        raise NotImplementedError
    return grad


# =============================================================================
# EXERCISE 3: Gradient Descent
# Difficulty: Medium
# =============================================================================

def gradient_descent(
    loss_fn,
    grad_fn,
    w_init: list,
    lr: float,
    n_steps: int
) -> tuple:
    """
    Run gradient descent starting from w_init.

    Update rule (repeat n_steps times):
        w = w - lr * grad_fn(w)

    Args:
        loss_fn: function w → scalar loss
        grad_fn: function w → gradient vector (same shape as w)
        w_init:  initial parameter values as a list
        lr:      learning rate (step size)
        n_steps: number of gradient steps to take

    Returns:
        (final_w, loss_history)
        final_w:      numpy array of final parameter values
        loss_history: list of loss values at each step (including step 0)

    Example: minimizing f(w1, w2) = (w1-3)^2 + (w2-5)^2
        Starting at [0, 0], after 50 steps with lr=0.1:
        final_w should be close to [3.0, 5.0]
    """
    w = np.array(w_init, dtype=float)
    loss_history = [float(loss_fn(w))]

    for _ in range(n_steps):
        # TODO: compute gradient and update w
        raise NotImplementedError

    return w, loss_history


# =============================================================================
# EXERCISE 4: ReLU Backward
# Difficulty: Medium
# =============================================================================

def relu_backward(grad_output: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    Compute the gradient flowing backward through a ReLU activation.

    Forward: a = relu(z) = max(0, z)
    Backward: given dL/da (grad_output), return dL/dz

    Key insight:
      - Where z > 0: relu was "active", gradient passes through unchanged
      - Where z ≤ 0: relu output was 0, gradient is BLOCKED (= 0)

    This is the chain rule applied to max(0, z):
        dL/dz = dL/da * d(relu)/dz
        d(relu)/dz = 1 if z > 0, else 0

    Args:
        grad_output: dL/da, gradient of loss w.r.t. relu output (same shape as z)
        z:           the pre-activation values (stored during forward pass)

    Returns:
        dL/dz: gradient of loss w.r.t. pre-activation (same shape as z)

    Example:
        z            = np.array([ 2.0, -1.0,  0.5, -3.0])
        grad_output  = np.array([ 1.0,  1.0,  1.0,  1.0])
        relu_backward → [1.0, 0.0, 1.0, 0.0]
        (blocked at positions where z ≤ 0)
    """
    # TODO: one line — multiply grad_output by the indicator (z > 0)
    raise NotImplementedError


# =============================================================================
# EXERCISE 5: Linear Layer Backward
# Difficulty: Hard (this is the core of backprop)
# =============================================================================

def linear_backward(grad_output: np.ndarray, x: np.ndarray, W: np.ndarray) -> tuple:
    """
    Compute gradients for a linear layer: z = W @ x + b

    Given the gradient dL/dz (coming from the layer above), compute:
        dL/dW — gradient for the weight matrix
        dL/db — gradient for the bias
        dL/dx — gradient to pass backward to the previous layer

    Formulas (derived from the chain rule):
        dL/dW = outer(dL/dz, x)    i.e. dL/dz[:, None] @ x[None, :]
        dL/db = dL/dz
        dL/dx = W.T @ dL/dz

    Args:
        grad_output: dL/dz, shape (d_out,)
        x:           input to this layer, shape (d_in,)
        W:           weight matrix, shape (d_out, d_in)

    Returns:
        (dL_dW, dL_db, dL_dx) as numpy arrays

    Example:
        W = np.array([[1.,2.],[3.,4.]])   # (2, 2)
        x = np.array([1., 2.])            # (2,)
        dL_dz = np.array([1., 1.])        # (2,) — gradient from above

        dL_dW → [[1,2],[1,2]]    (outer product of dL_dz and x)
        dL_db → [1, 1]            (same as dL_dz)
        dL_dx → [4, 6]            (W.T @ dL_dz = [[1,3],[2,4]] @ [1,1])
    """
    # TODO: implement all three gradients using the formulas above
    raise NotImplementedError


# =============================================================================
# RUN ALL EXERCISES
# =============================================================================

def check():
    print("=" * 55)
    print("Module 02 — Exercise Results")
    print("=" * 55)

    passed = 0
    total  = 0
    skipped = 0

    def test(name, got, expected, tol=1e-4):
        nonlocal passed, total
        total += 1
        if isinstance(expected, float) or isinstance(expected, int):
            ok = abs(float(got) - float(expected)) < tol
        else:
            ok = np.allclose(got, expected, atol=tol)
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        if ok:
            passed += 1

    # ── Exercise 1: numerical_derivative ──────────────────────────────────
    print("\nExercise 1: numerical_derivative")
    try:
        f = lambda x: x**2
        test("f(x)=x², f'(3) = 6",        numerical_derivative(f, 3.0),  6.0)
        test("f(x)=x², f'(-2) = -4",       numerical_derivative(f, -2.0), -4.0)
        test("f(x)=x², f'(0) = 0",         numerical_derivative(f, 0.0),  0.0)
        g = lambda x: x**3 - 2*x + 1
        test("f(x)=x³-2x+1, f'(2) = 10",  numerical_derivative(g, 2.0),  10.0)
    except NotImplementedError:
        skipped += 1
        print("  [SKIP] Not implemented yet")

    # ── Exercise 2: numerical_gradient ────────────────────────────────────
    print("\nExercise 2: numerical_gradient")
    try:
        f = lambda w: (w[0]-3)**2 + (w[1]-5)**2
        grad = numerical_gradient(f, np.array([0., 0.]))
        test("∂f/∂w1 at [0,0] = -6",  grad[0], -6.0)
        test("∂f/∂w2 at [0,0] = -10", grad[1], -10.0)
        grad2 = numerical_gradient(f, np.array([3., 5.]))
        test("gradient at minimum ≈ [0,0]", grad2, np.array([0., 0.]))
    except NotImplementedError:
        skipped += 1
        print("  [SKIP] Not implemented yet")

    # ── Exercise 3: gradient_descent ──────────────────────────────────────
    print("\nExercise 3: gradient_descent")
    try:
        f    = lambda w: (w[0]-3)**2 + (w[1]-5)**2
        grad_f = lambda w: np.array([2*(w[0]-3), 2*(w[1]-5)])
        final_w, loss_hist = gradient_descent(f, grad_f, [0., 0.], lr=0.1, n_steps=100)
        test("converges to w1 ≈ 3",    final_w[0], 3.0, tol=1e-3)
        test("converges to w2 ≈ 5",    final_w[1], 5.0, tol=1e-3)
        test("loss_history[0] = 34",   loss_hist[0], 34.0)
        test("loss decreases overall", float(loss_hist[-1]), 0.0, tol=1e-2)
    except NotImplementedError:
        skipped += 1
        print("  [SKIP] Not implemented yet")

    # ── Exercise 4: relu_backward ──────────────────────────────────────────
    print("\nExercise 4: relu_backward")
    try:
        z    = np.array([ 2.0, -1.0,  0.5, -3.0])
        grad = np.array([ 1.0,  2.0,  3.0,  4.0])
        result = relu_backward(grad, z)
        test("active neurons pass gradient",  result[0],  1.0)
        test("inactive neurons blocked",      result[1],  0.0)
        test("active gradient value",         result[2],  3.0)
        test("inactive gradient blocked",     result[3],  0.0)
    except NotImplementedError:
        skipped += 1
        print("  [SKIP] Not implemented yet")

    # ── Exercise 5: linear_backward ───────────────────────────────────────
    print("\nExercise 5: linear_backward")
    try:
        W     = np.array([[1., 2.], [3., 4.]])
        x     = np.array([1., 2.])
        dL_dz = np.array([1., 1.])
        dW, db, dx = linear_backward(dL_dz, x, W)
        test("dL/dW[0] = [1,2]", dW[0], np.array([1., 2.]))
        test("dL/dW[1] = [1,2]", dW[1], np.array([1., 2.]))
        test("dL/db = [1,1]",     db,    np.array([1., 1.]))
        test("dL/dx = [4,6]",     dx,    np.array([4., 6.]))
    except NotImplementedError:
        skipped += 1
        print("  [SKIP] Not implemented yet")

    print(f"\n{'='*55}")
    if skipped > 0:
        print(f"Score: {passed}/{total} passed,  {skipped} exercise(s) not yet implemented.")
        print("Implement the TODO sections, then re-run.")
    elif passed == total and total > 0:
        print(f"Score: {passed}/{total} — All exercises complete! Move on to Module 03.")
    else:
        print(f"Score: {passed}/{total} — {total - passed} failing.")
    print("=" * 55)


if __name__ == '__main__':
    check()
