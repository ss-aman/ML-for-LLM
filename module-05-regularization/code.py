"""
Module 05: Regularization — Code Demonstrations
================================================
Covers:
  - Overfitting: degree-15 polynomial on 10 points
  - L2 regularization fixes overfitting
  - Dropout implemented from scratch
  - Training vs validation loss curves and early stopping

Run this file directly: python code.py
"""

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_poly_features(x, degree):
    """
    Build a polynomial feature matrix from 1D input x.
    Returns X of shape (len(x), degree+1) with columns [1, x, x^2, ..., x^d].
    Think of this as feature engineering — we're expanding one input into
    many derived features, like extracting hour/day/weekday from a timestamp.
    """
    return np.column_stack([x ** d for d in range(degree + 1)])


def fit_polynomial_ridge(x, y, degree, lam=0.0):
    """
    Fit a polynomial of given degree using closed-form ridge regression.

    Ridge (L2) solution: w = (X'X + λI)^{-1} X'y
    When λ=0 this is plain ordinary least squares.

    Backend analogy: lambda is like a regularization dial.
    Turn it up and you get smoother, more conservative predictions.
    Turn it to zero and you get pure interpolation (memorization).
    """
    X = make_poly_features(x, degree)
    n_features = X.shape[1]
    # λI adds a small penalty to the diagonal — prevents overfitting
    regularizer = lam * np.eye(n_features)
    regularizer[0, 0] = 0   # don't penalise the bias term
    w = np.linalg.solve(X.T @ X + regularizer, X.T @ y)
    return w


def predict_polynomial(x, w):
    """Evaluate polynomial with weights w at points x."""
    degree = len(w) - 1
    X = make_poly_features(x, degree)
    return X @ w


# ---------------------------------------------------------------------------
# 1. Demonstrating Overfitting
# ---------------------------------------------------------------------------

def demo_overfitting():
    """
    Fit a degree-15 polynomial to 10 noisy training points.
    Show it memorizes training data but fails on test data.

    The true function is y = sin(x).  We observe 10 noisy samples.
    A degree-15 polynomial has 16 free parameters for 10 data points —
    more parameters than data, so it can thread through every point exactly,
    including the noise.

    Backend analogy: this is like having an alerting rule for every single
    historical incident.  Each rule fires perfectly on the past incident but
    triggers false positives constantly on new traffic.
    """
    print("=" * 60)
    print("1. Overfitting: degree-15 polynomial on 10 training points")
    print("=" * 60)

    rng = np.random.default_rng(1)

    # 10 training points from sin(x) + noise
    x_train = np.linspace(0, 2 * np.pi, 10)
    y_train = np.sin(x_train) + rng.normal(0, 0.2, 10)

    # 200 test points on the same range
    x_test = np.linspace(0, 2 * np.pi, 200)
    y_test = np.sin(x_test)   # noiseless true function

    # Fit degree-15 polynomial (way too many parameters for 10 points)
    w_overfit = fit_polynomial_ridge(x_train, y_train, degree=15, lam=0.0)
    # Fit degree-3 polynomial (reasonable for sin(x))
    w_good    = fit_polynomial_ridge(x_train, y_train, degree=3,  lam=0.0)

    y_pred_overfit = predict_polynomial(x_test, w_overfit)
    y_pred_good    = predict_polynomial(x_test, w_good)

    # Compute errors
    train_mse_overfit = np.mean((predict_polynomial(x_train, w_overfit) - y_train) ** 2)
    test_mse_overfit  = np.mean((y_pred_overfit - y_test) ** 2)
    train_mse_good    = np.mean((predict_polynomial(x_train, w_good) - y_train) ** 2)
    test_mse_good     = np.mean((y_pred_good - y_test) ** 2)

    print(f"  Degree-15 (overfit): train MSE={train_mse_overfit:.4f}, test MSE={test_mse_overfit:.4f}")
    print(f"  Degree-3  (good):    train MSE={train_mse_good:.4f},  test MSE={test_mse_good:.4f}")
    print(f"  Overfit model test error is {test_mse_overfit/test_mse_good:.1f}x higher than good model\n")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (w, label, color) in zip(
        axes,
        [(w_overfit, "Degree-15 (OVERFIT)", "red"),
         (w_good,    "Degree-3  (good fit)", "blue")]
    ):
        y_pred = predict_polynomial(x_test, w)
        ax.plot(x_test, y_test,  "g--", linewidth=2, label="True: sin(x)")
        ax.plot(x_test, y_pred,  color=color, linewidth=2, label=label)
        ax.scatter(x_train, y_train, c="black", zorder=5, label="Training data")
        ax.set_ylim(-3, 3)
        ax.set_title(label)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle("Overfitting: degree-15 polynomial memorises noise", fontsize=12)
    plt.tight_layout()
    plt.savefig("plot_01_overfitting.png", dpi=100)
    print("  Saved: plot_01_overfitting.png\n")


# ---------------------------------------------------------------------------
# 2. L2 Regularization Fixes Overfitting
# ---------------------------------------------------------------------------

def demo_l2_regularization():
    """
    Same setup as above (degree-15 polynomial, 10 points), but now add L2
    regularization.  Show that increasing λ progressively reduces overfitting.

    Backend analogy: λ is the idle-connection timeout.  Small timeout =
    connections (weights) stay open (large), risk of resource exhaustion
    (overfitting). Large timeout = connections aggressively closed (small
    weights), model can't rely on any single feature, generalizes better.
    """
    print("=" * 60)
    print("2. L2 Regularization on Degree-15 Polynomial")
    print("=" * 60)

    rng = np.random.default_rng(1)

    x_train = np.linspace(0, 2 * np.pi, 10)
    y_train = np.sin(x_train) + rng.normal(0, 0.2, 10)
    x_test  = np.linspace(0, 2 * np.pi, 200)
    y_test  = np.sin(x_test)

    lambdas = [0.0, 1e-4, 1e-2, 1.0]

    fig, axes = plt.subplots(1, len(lambdas), figsize=(16, 4))

    for ax, lam in zip(axes, lambdas):
        w = fit_polynomial_ridge(x_train, y_train, degree=15, lam=lam)
        y_pred = predict_polynomial(x_test, w)
        train_mse = np.mean((predict_polynomial(x_train, w) - y_train) ** 2)
        test_mse  = np.mean((y_pred - y_test) ** 2)

        ax.plot(x_test, y_test,   "g--", linewidth=2, label="True")
        ax.plot(x_test, y_pred,   "r-",  linewidth=2, label=f"λ={lam}")
        ax.scatter(x_train, y_train, c="black", zorder=5, s=20)
        ax.set_ylim(-2, 2)
        ax.set_title(f"λ={lam}\ntrain={train_mse:.3f} test={test_mse:.3f}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        print(f"  λ={lam:<8}: train MSE={train_mse:.4f}, test MSE={test_mse:.4f}")

    print()
    plt.suptitle("L2 Regularization: λ controls the bias-variance tradeoff", fontsize=11)
    plt.tight_layout()
    plt.savefig("plot_02_l2_regularization.png", dpi=100)
    print("  Saved: plot_02_l2_regularization.png\n")


# ---------------------------------------------------------------------------
# 3. Dropout from Scratch
# ---------------------------------------------------------------------------

def dropout(x, p_drop, training=True, rng=None):
    """
    Apply dropout to array x.

    During training:
      - Each element is independently zeroed out with probability p_drop
      - Surviving elements are scaled up by 1/(1-p_drop) to preserve
        expected magnitude (this is "inverted dropout")

    During inference:
      - No dropout: return x unchanged

    Args:
        x:        numpy array (activations from a layer)
        p_drop:   probability of dropping each neuron (e.g. 0.5)
        training: True during training, False during inference

    Backend analogy: chaos engineering.  During "training" (normal ops
    with Chaos Monkey running), random instances get killed.  The system
    must survive without them.  During "inference" (production traffic),
    everything is up — no chaos injection.

    The scaling factor 1/(1-p_drop) ensures the total activation magnitude
    is the same in training and inference — like normalising your chaos test
    throughput so baseline metrics are comparable.
    """
    if rng is None:
        rng = np.random.default_rng()

    if not training:
        return x

    p_keep = 1.0 - p_drop
    # Bernoulli mask: 1 = keep, 0 = drop
    mask = rng.binomial(1, p_keep, size=x.shape).astype(float)
    # Inverted dropout scaling: divide by p_keep to keep expected value same
    return x * mask / p_keep


def demo_dropout():
    """
    Demonstrate dropout on a simple array of activations.
    Show the effect of different dropout rates on a layer's output.
    """
    print("=" * 60)
    print("3. Dropout from Scratch")
    print("=" * 60)

    rng = np.random.default_rng(42)

    # Simulate a layer with 20 neurons, all activated (value=1.0)
    activations = np.ones(20)

    fig, axes = plt.subplots(1, 4, figsize=(14, 3))
    drop_rates = [0.0, 0.2, 0.5, 0.8]

    for ax, p_drop in zip(axes, drop_rates):
        dropped = dropout(activations, p_drop, training=True, rng=rng)
        ax.bar(range(20), dropped, color=["red" if v == 0 else "steelblue" for v in dropped])
        ax.set_title(f"Dropout rate = {p_drop}\nmean={dropped.mean():.2f} (expected 1.0)")
        ax.set_ylim(0, 2.5)
        ax.set_xlabel("Neuron index")
        ax.set_ylabel("Activation")
        ax.axhline(1.0, color="green", linestyle="--", alpha=0.5, label="expected=1.0")
        n_dropped = int((dropped == 0).sum())
        print(f"  p_drop={p_drop}: {n_dropped}/20 neurons dropped, mean activation={dropped.mean():.3f}")

    plt.suptitle("Dropout (training mode): red bars = killed neurons, mean stays ~1.0 due to scaling")
    plt.tight_layout()
    plt.savefig("plot_03_dropout.png", dpi=100)
    print("  Saved: plot_03_dropout.png\n")

    # Verify inference mode: no dropout
    inference_out = dropout(activations, p_drop=0.5, training=False)
    assert np.allclose(inference_out, activations), "Inference mode should return input unchanged"
    print("  Inference mode check: PASSED (no neurons dropped)\n")


# ---------------------------------------------------------------------------
# 4. Training vs Validation Loss — Early Stopping
# ---------------------------------------------------------------------------

def demo_early_stopping():
    """
    Simulate training a model on a small dataset with no regularization.
    Show how training loss keeps falling while validation loss starts rising
    — the classic overfitting signature.

    We simulate this by deliberately having a model with too much capacity
    trained for too many epochs on a small dataset.

    Backend analogy: monitoring two metrics — your benchmark suite (train
    loss) and real-user SLO (val loss).  The benchmark keeps looking better
    while real users are experiencing degradation.  Early stopping = rollback
    to the last checkpoint before the SLO started degrading.
    """
    print("=" * 60)
    print("4. Training vs Validation Loss — Early Stopping Criterion")
    print("=" * 60)

    rng = np.random.default_rng(5)

    # Small dataset: 20 train, 200 val
    n_train = 20
    x_train = np.linspace(0, 2 * np.pi, n_train)
    y_train = np.sin(x_train) + rng.normal(0, 0.3, n_train)

    x_val = np.linspace(0, 2 * np.pi, 200)
    y_val = np.sin(x_val)   # noiseless

    # Fit progressively higher-degree polynomials as a stand-in for "more epochs"
    degrees = list(range(1, 16))
    train_losses = []
    val_losses   = []

    for deg in degrees:
        w = fit_polynomial_ridge(x_train, y_train, degree=deg, lam=0.0)
        train_mse = np.mean((predict_polynomial(x_train, w) - y_train) ** 2)
        val_mse   = np.mean((predict_polynomial(x_val,   w) - y_val)   ** 2)
        train_losses.append(train_mse)
        val_losses.append(val_mse)

    best_epoch = int(np.argmin(val_losses))
    print(f"  Best model: degree={degrees[best_epoch]}, val MSE={val_losses[best_epoch]:.4f}")
    print(f"  Overfit at degree={degrees[-1]}: train={train_losses[-1]:.4f}, val={val_losses[-1]:.4f}")

    plt.figure(figsize=(8, 5))
    plt.plot(degrees, train_losses, "b-o", label="Training loss", markersize=5)
    plt.plot(degrees, val_losses,   "r-o", label="Validation loss", markersize=5)
    plt.axvline(degrees[best_epoch], color="green", linestyle="--",
                label=f"Early stop: degree={degrees[best_epoch]}")
    plt.fill_between(
        degrees,
        train_losses,
        val_losses,
        alpha=0.1,
        color="red",
        label="Generalisation gap (↑ = overfit)"
    )
    plt.xlabel("Model Complexity (polynomial degree  ≈  'training epoch')")
    plt.ylabel("MSE Loss")
    plt.title("Early Stopping: Stop when validation loss starts rising\n"
              "(train loss keeps falling — don't be fooled)")
    plt.legend()
    plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("plot_04_early_stopping.png", dpi=100)
    print("  Saved: plot_04_early_stopping.png\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo_overfitting()
    demo_l2_regularization()
    demo_dropout()
    demo_early_stopping()

    print("All demos complete.  PNG plots saved to the current directory.")
