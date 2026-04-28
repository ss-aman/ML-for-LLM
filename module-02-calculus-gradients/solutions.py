"""
Module 02 — Solutions
======================
Reference solutions. Only look here after genuinely attempting exercises.

Run: python solutions.py
"""

import numpy as np


def numerical_derivative(f, x: float, h: float = 1e-5) -> float:
    return (f(x + h) - f(x - h)) / (2 * h)


def numerical_gradient(f, w: np.ndarray, h: float = 1e-5) -> np.ndarray:
    grad = np.zeros_like(w, dtype=float)
    for i in range(len(w)):
        w_plus       = w.copy(); w_plus[i]  += h
        w_minus      = w.copy(); w_minus[i] -= h
        grad[i]      = (f(w_plus) - f(w_minus)) / (2 * h)
    return grad


def gradient_descent(loss_fn, grad_fn, w_init, lr, n_steps):
    w = np.array(w_init, dtype=float)
    loss_history = [float(loss_fn(w))]
    for _ in range(n_steps):
        g = grad_fn(w)
        w = w - lr * g
        loss_history.append(float(loss_fn(w)))
    return w, loss_history


def relu_backward(grad_output: np.ndarray, z: np.ndarray) -> np.ndarray:
    return grad_output * (z > 0).astype(float)


def linear_backward(grad_output: np.ndarray, x: np.ndarray, W: np.ndarray) -> tuple:
    dW = np.outer(grad_output, x)   # (d_out, d_in)
    db = grad_output                 # (d_out,)
    dx = W.T @ grad_output           # (d_in,)
    return dW, db, dx


# =============================================================================
# CHECK (same tests as exercises.py)
# =============================================================================

def check():
    print("=" * 55)
    print("Module 02 — Solutions Check")
    print("=" * 55)

    passed = 0
    total  = 0

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

    print("\nExercise 1: numerical_derivative")
    f = lambda x: x**2
    test("f(x)=x², f'(3) = 6",        numerical_derivative(f, 3.0),  6.0)
    test("f(x)=x², f'(-2) = -4",       numerical_derivative(f, -2.0), -4.0)
    test("f(x)=x², f'(0) = 0",         numerical_derivative(f, 0.0),  0.0)
    g = lambda x: x**3 - 2*x + 1
    test("f(x)=x³-2x+1, f'(2) = 10",  numerical_derivative(g, 2.0),  10.0)

    print("\nExercise 2: numerical_gradient")
    f = lambda w: (w[0]-3)**2 + (w[1]-5)**2
    grad = numerical_gradient(f, np.array([0., 0.]))
    test("∂f/∂w1 at [0,0] = -6",  grad[0], -6.0)
    test("∂f/∂w2 at [0,0] = -10", grad[1], -10.0)
    grad2 = numerical_gradient(f, np.array([3., 5.]))
    test("gradient at minimum ≈ [0,0]", grad2, np.array([0., 0.]))

    print("\nExercise 3: gradient_descent")
    f      = lambda w: (w[0]-3)**2 + (w[1]-5)**2
    grad_f = lambda w: np.array([2*(w[0]-3), 2*(w[1]-5)])
    final_w, loss_hist = gradient_descent(f, grad_f, [0., 0.], lr=0.1, n_steps=100)
    test("converges to w1 ≈ 3",    final_w[0], 3.0, tol=1e-3)
    test("converges to w2 ≈ 5",    final_w[1], 5.0, tol=1e-3)
    test("loss_history[0] = 34",   loss_hist[0], 34.0)
    test("loss decreases overall", float(loss_hist[-1]), 0.0, tol=1e-2)

    print("\nExercise 4: relu_backward")
    z    = np.array([ 2.0, -1.0,  0.5, -3.0])
    grad = np.array([ 1.0,  2.0,  3.0,  4.0])
    result = relu_backward(grad, z)
    test("active neurons pass gradient",  result[0],  1.0)
    test("inactive neurons blocked",      result[1],  0.0)
    test("active gradient value",         result[2],  3.0)
    test("inactive gradient blocked",     result[3],  0.0)

    print("\nExercise 5: linear_backward")
    W     = np.array([[1., 2.], [3., 4.]])
    x     = np.array([1., 2.])
    dL_dz = np.array([1., 1.])
    dW, db, dx = linear_backward(dL_dz, x, W)
    test("dL/dW[0] = [1,2]", dW[0], np.array([1., 2.]))
    test("dL/dW[1] = [1,2]", dW[1], np.array([1., 2.]))
    test("dL/db = [1,1]",     db,    np.array([1., 1.]))
    test("dL/dx = [4,6]",     dx,    np.array([4., 6.]))

    print(f"\n{'='*55}")
    print(f"Score: {passed}/{total}")
    print("=" * 55)


if __name__ == '__main__':
    check()
