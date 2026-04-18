"""
Module 05: Regularization — Exercises
======================================
Work through these exercises in order.  Each is a function you complete.
Run the file when you're done: python exercise.py

Exercises:
  1. Fit an overfit model, add L2, compare test error
  2. Implement L1 regularization and show it produces sparser weights than L2
  3. (Challenge) Implement k-fold cross-validation from scratch
"""

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def make_poly_features(x, degree):
    """Build polynomial feature matrix: columns [1, x, x^2, ..., x^degree]."""
    return np.column_stack([x ** d for d in range(degree + 1)])


def poly_predict(x, w):
    """Evaluate polynomial with weights w at points x."""
    return make_poly_features(x, len(w) - 1) @ w


# ---------------------------------------------------------------------------
# Exercise 1: Overfit Model → Add L2 → Compare Test Error
# ---------------------------------------------------------------------------

def exercise_1_l2_vs_overfit():
    """
    Fit a degree-10 polynomial to 8 training points from sin(x) + noise.
    Compare:
      a) No regularization (λ=0) — will overfit badly
      b) L2 regularization (λ=0.01) — should generalize better

    The closed-form ridge solution is:
        w = (X'X + λI)^{-1} X'y

    TODO: implement the l2_fit function below.

    Print train and test MSE for each setting, plot both fits against the
    true function.

    Backend analogy: production latency (test MSE) is what matters, not
    your internal benchmark (train MSE). An overfitted model "wins" on
    the benchmark but loses in production.
    """
    rng = np.random.default_rng(7)

    x_train = np.linspace(0, 2 * np.pi, 8)
    y_train = np.sin(x_train) + rng.normal(0, 0.25, 8)

    x_test = np.linspace(0, 2 * np.pi, 300)
    y_test = np.sin(x_test)

    def l2_fit(x, y, degree, lam):
        """
        Fit a polynomial of given degree with L2 regularization strength lam.
        Returns weight vector w of shape (degree+1,).

        Hint: closed-form solution is  w = (X'X + lam*I)^{-1} X'y
              but don't penalise the bias term: set regularizer[0,0] = 0

        TODO: implement this.
        """
        X = make_poly_features(x, degree)
        n_features = X.shape[1]
        # --- YOUR CODE HERE ---
        regularizer = lam * np.eye(n_features)
        regularizer[0, 0] = 0   # don't regularize bias
        w = np.linalg.solve(X.T @ X + regularizer, X.T @ y)
        # --- END YOUR CODE ---
        return w

    lambdas = [0.0, 0.01]
    results = {}

    for lam in lambdas:
        w = l2_fit(x_train, y_train, degree=10, lam=lam)
        train_mse = np.mean((poly_predict(x_train, w) - y_train) ** 2)
        test_mse  = np.mean((poly_predict(x_test,  w) - y_test)  ** 2)
        results[lam] = {"w": w, "train": train_mse, "test": test_mse}
        print(f"  λ={lam:<6}: train MSE={train_mse:.4f}, test MSE={test_mse:.4f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    titles = {0.0: "No Regularization (λ=0)  — OVERFIT",
              0.01: "L2 Regularization (λ=0.01) — Generalizes"}

    for ax, (lam, res) in zip(axes, results.items()):
        y_pred = poly_predict(x_test, res["w"])
        ax.plot(x_test, y_test, "g--", linewidth=2, label="True: sin(x)")
        ax.plot(x_test, y_pred, "r-",  linewidth=2, label=f"Fit (test MSE={res['test']:.3f})")
        ax.scatter(x_train, y_train, c="black", zorder=5, s=40, label="Training pts")
        ax.set_ylim(-3, 3)
        ax.set_title(titles[lam])
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("ex1_l2_vs_overfit.png", dpi=100)
    print("Exercise 1: saved ex1_l2_vs_overfit.png")

    # L2 model should have lower test MSE
    assert results[0.01]["test"] < results[0.0]["test"], \
        "L2 regularization should reduce test MSE"
    print("Exercise 1: PASSED\n")


# ---------------------------------------------------------------------------
# Exercise 2: L1 vs L2 — Sparsity
# ---------------------------------------------------------------------------

def exercise_2_l1_sparsity():
    """
    Compare L1 vs L2 regularization on the same problem.
    Key insight: L1 drives weights to *exactly* zero, L2 only shrinks them.

    The difference: L1 gradient is constant (sign of w) regardless of w's
    magnitude.  Even a tiny weight gets the same push toward zero.
    L2 gradient is 2λw — small weights get a tiny push, and rarely hit zero.

    Since sklearn is not available, we use gradient descent to minimise
    a degree-10 polynomial fit with:
      - L1 penalty:  loss += λ * sum(|w|),   gradient: λ * sign(w)
      - L2 penalty:  loss += λ * sum(w²),    gradient: 2λ * w

    TODO: fill in the gradient updates inside the training loops.

    Backend analogy:
      L1 = aggressive circuit breaker: weak connections get fully disabled
      L2 = timeout + throttle: connections slow down but stay open
    """
    rng = np.random.default_rng(3)

    n = 30
    x = np.linspace(0, 2 * np.pi, n)
    y = np.sin(x) + rng.normal(0, 0.3, n)

    degree = 10
    X = make_poly_features(x, degree)
    lam = 0.5       # strong regularization to clearly show the difference
    lr  = 1e-4
    n_steps = 5000

    # --- L1 training ---
    w_l1 = np.zeros(degree + 1)
    for _ in range(n_steps):
        pred = X @ w_l1
        resid = pred - y
        grad_mse = (2 / n) * X.T @ resid

        # --- YOUR CODE HERE ---
        # L1 gradient: lam * sign(w_l1)
        # Don't regularize the bias (index 0)
        grad_l1 = lam * np.sign(w_l1)
        grad_l1[0] = 0   # no regularization on bias
        w_l1 = w_l1 - lr * (grad_mse + grad_l1)
        # --- END YOUR CODE ---

    # --- L2 training ---
    w_l2 = np.zeros(degree + 1)
    for _ in range(n_steps):
        pred = X @ w_l2
        resid = pred - y
        grad_mse = (2 / n) * X.T @ resid

        # --- YOUR CODE HERE ---
        # L2 gradient: 2 * lam * w_l2
        # Don't regularize the bias (index 0)
        grad_l2 = 2 * lam * w_l2
        grad_l2[0] = 0
        w_l2 = w_l2 - lr * (grad_mse + grad_l2)
        # --- END YOUR CODE ---

    # Count near-zero weights (excluding bias)
    threshold = 0.01
    l1_zeros = np.sum(np.abs(w_l1[1:]) < threshold)
    l2_zeros = np.sum(np.abs(w_l2[1:]) < threshold)

    print(f"  L1 weights (excl. bias): {np.round(w_l1[1:], 3)}")
    print(f"  L2 weights (excl. bias): {np.round(w_l2[1:], 3)}")
    print(f"  Weights near-zero (|w| < {threshold}):")
    print(f"    L1: {l1_zeros}/10 weights ≈ 0")
    print(f"    L2: {l2_zeros}/10 weights ≈ 0")

    # Plot weight magnitudes
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    w_idx = np.arange(1, degree + 1)   # skip bias

    axes[0].bar(w_idx, np.abs(w_l1[1:]), color="steelblue")
    axes[0].axhline(threshold, color="red", linestyle="--", label=f"zero threshold={threshold}")
    axes[0].set_title(f"L1 regularization (λ={lam})\n{l1_zeros}/10 weights ≈ zero (SPARSE)")
    axes[0].set_xlabel("Weight index")
    axes[0].set_ylabel("|weight|")
    axes[0].legend()

    axes[1].bar(w_idx, np.abs(w_l2[1:]), color="orange")
    axes[1].axhline(threshold, color="red", linestyle="--", label=f"zero threshold={threshold}")
    axes[1].set_title(f"L2 regularization (λ={lam})\n{l2_zeros}/10 weights ≈ zero (DENSE, just small)")
    axes[1].set_xlabel("Weight index")
    axes[1].set_ylabel("|weight|")
    axes[1].legend()

    plt.suptitle("L1 produces sparse weights, L2 produces small but non-zero weights")
    plt.tight_layout()
    plt.savefig("ex2_l1_vs_l2_sparsity.png", dpi=100)
    print("Exercise 2: saved ex2_l1_vs_l2_sparsity.png")

    # L1 should produce more near-zero weights than L2
    assert l1_zeros >= l2_zeros, \
        f"L1 should produce >= as many near-zero weights as L2 ({l1_zeros} vs {l2_zeros})"
    print("Exercise 2: PASSED\n")


# ---------------------------------------------------------------------------
# Exercise 3 (Challenge): K-Fold Cross-Validation from Scratch
# ---------------------------------------------------------------------------

def exercise_3_kfold_cross_validation():
    """
    CHALLENGE: Implement k-fold cross-validation from scratch.

    Cross-validation gives a more reliable estimate of generalisation error
    than a single train/test split — especially important with small datasets.

    Algorithm:
      1. Split data into k equal folds (e.g., k=5)
      2. For each fold i:
           - Use fold i as the validation set
           - Train on the remaining k-1 folds
           - Record validation MSE
      3. Average the k validation MSEs — this is the cross-validation error

    Use cross-validation to choose the best polynomial degree from [1,3,5,7,9]
    for fitting y = sin(x) + noise with n=40 training points.

    TODO: implement kfold_cross_val below.

    Backend analogy: instead of testing your new service on one traffic sample,
    you rotate through k different time windows, each time using k-1 windows
    to "train" your config and 1 window to validate.  The average metric across
    all k windows is a more stable estimate than any single window.
    """
    rng = np.random.default_rng(11)

    n = 40
    x = rng.uniform(0, 2 * np.pi, n)
    y = np.sin(x) + rng.normal(0, 0.3, n)

    def kfold_cross_val(x, y, degree, k, lam=1e-3):
        """
        Perform k-fold cross-validation for a polynomial of given degree.
        Returns the mean validation MSE across all k folds.

        Args:
            x, y:   data arrays of length n
            degree: polynomial degree to fit
            k:      number of folds
            lam:    L2 regularization strength

        Returns:
            mean_val_mse (float)
        """
        n = len(x)
        fold_size = n // k
        val_mses = []

        # --- YOUR CODE HERE ---
        # For each fold i in range(k):
        #   1. val indices:   range(i*fold_size, (i+1)*fold_size)
        #   2. train indices: everything else
        #   3. Fit polynomial on train with L2 (use make_poly_features + np.linalg.solve)
        #   4. Compute val MSE; append to val_mses
        for i in range(k):
            val_idx   = np.arange(i * fold_size, (i + 1) * fold_size)
            train_idx = np.concatenate([np.arange(0, i * fold_size),
                                        np.arange((i + 1) * fold_size, n)])

            x_tr, y_tr = x[train_idx], y[train_idx]
            x_va, y_va = x[val_idx],   y[val_idx]

            X_tr = make_poly_features(x_tr, degree)
            reg  = lam * np.eye(degree + 1)
            reg[0, 0] = 0
            w = np.linalg.solve(X_tr.T @ X_tr + reg, X_tr.T @ y_tr)

            val_pred = poly_predict(x_va, w)
            val_mse  = np.mean((val_pred - y_va) ** 2)
            val_mses.append(val_mse)
        # --- END YOUR CODE ---

        return float(np.mean(val_mses))

    degrees = [1, 3, 5, 7, 9]
    k = 5
    cv_errors = []

    print("  K-Fold Cross-Validation (k=5)")
    for deg in degrees:
        cv_mse = kfold_cross_val(x, y, degree=deg, k=k)
        cv_errors.append(cv_mse)
        print(f"    degree={deg}: CV MSE={cv_mse:.4f}")

    best_degree = degrees[int(np.argmin(cv_errors))]
    print(f"  Best degree by CV: {best_degree}")

    # Plot CV error vs degree
    plt.figure(figsize=(7, 4))
    plt.plot(degrees, cv_errors, "bo-", linewidth=2, markersize=8)
    plt.axvline(best_degree, color="green", linestyle="--",
                label=f"Best degree = {best_degree}")
    plt.xlabel("Polynomial Degree")
    plt.ylabel("CV MSE (5-fold)")
    plt.title("Exercise 3: K-Fold Cross-Validation — choosing model complexity")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("ex3_kfold_cv.png", dpi=100)
    print("Exercise 3: saved ex3_kfold_cv.png")

    # Best degree should be in a reasonable range (not degree 1 which underfits,
    # not degree 9 which overfits — sin(x) is well-captured by degree 3-5)
    assert best_degree in [3, 5, 7], \
        f"CV should pick a moderate degree for sin(x); got {best_degree}"
    print("Exercise 3: PASSED\n")


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def main():
    print("Module 05 Exercises\n" + "=" * 40)
    exercise_1_l2_vs_overfit()
    exercise_2_l1_sparsity()
    exercise_3_kfold_cross_validation()
    print("All exercises passed!")


if __name__ == "__main__":
    main()
