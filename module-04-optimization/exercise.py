"""
Module 04: Optimization Algorithms — Exercises
===============================================
Work through these exercises in order.  Each is a function you complete.
Run the file when you're done: python exercise.py

Exercises:
  1. Tune the learning rate — observe overshooting vs slow convergence
  2. Gradient descent with momentum from scratch
  3. (Challenge) Implement Adam and verify it beats vanilla SGD on noisy loss
"""

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Exercise 1: Learning Rate Tuning
# ---------------------------------------------------------------------------

def exercise_1_learning_rate_tuning():
    """
    Run gradient descent on f(θ) = θ^2 with four different learning rates:
        lr_values = [0.001, 0.01, 0.1, 1.0]

    For each learning rate:
      - Start at θ = 8.0
      - Run 80 steps of gradient descent
      - Record the loss (θ^2) at each step
      - If θ diverges (|θ| > 1e6) fill the rest with np.nan

    Then plot all four curves on one figure.

    What to observe:
      - lr=0.001: converges but slowly (too cautious)
      - lr=0.01:  steady convergence
      - lr=0.1:   fast convergence
      - lr=1.0:   overshoots and diverges (too aggressive)

    Backend analogy: like setting retry wait times.  Too short → hammers the
    service and makes things worse.  Too long → recovery is painfully slow.

    TODO: Fill in the gradient descent loop below.
    """
    lr_values = [0.001, 0.01, 0.1, 1.0]
    n_steps   = 80
    theta_init = 8.0

    def loss_fn(theta): return theta ** 2
    def grad_fn(theta): return 2.0 * theta   # d/dθ (θ^2) = 2θ

    all_losses = {}

    for lr in lr_values:
        theta = theta_init
        losses = []

        for step in range(n_steps):
            # --- YOUR CODE HERE ---
            # 1. Compute the gradient at the current theta
            # 2. Update theta:  theta = theta - lr * gradient
            # 3. Append loss_fn(theta) to losses
            # 4. If abs(theta) > 1e6, fill remaining steps with np.nan and break
            g = grad_fn(theta)
            theta = theta - lr * g
            losses.append(loss_fn(theta))
            if abs(theta) > 1e6:
                losses.extend([np.nan] * (n_steps - len(losses)))
                break
            # --- END YOUR CODE ---

        all_losses[lr] = losses

    # Plot
    plt.figure(figsize=(10, 5))
    for lr, losses in all_losses.items():
        plt.plot(losses, label=f"lr={lr}", linewidth=2)
    plt.title("Exercise 1: Learning Rate Tuning on f(θ)=θ²")
    plt.xlabel("Step")
    plt.ylabel("Loss (θ²)")
    plt.yscale("symlog", linthresh=1e-6)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("ex1_lr_tuning.png", dpi=100)
    print("Exercise 1: saved ex1_lr_tuning.png")

    # Basic check: lr=0.1 should converge, lr=1.0 should diverge
    assert all_losses[0.1][-1] < 0.01,  "lr=0.1 should converge to near zero"
    assert np.isnan(all_losses[1.0][-1]), "lr=1.0 should diverge (end with nan)"
    print("Exercise 1: PASSED\n")


# ---------------------------------------------------------------------------
# Exercise 2: Gradient Descent with Momentum
# ---------------------------------------------------------------------------

def exercise_2_momentum():
    """
    Implement gradient descent with momentum and compare it to plain GD.

    The update rule:
        v_t = beta * v_{t-1} + grad_t     # accumulate velocity
        θ_t = θ_{t-1} - lr * v_t          # update with velocity

    Use the same 2D loss surface from the notes:
        f(x, y) = (x - 3)^2 + 10*(y + 2)^2

    The '10*' on the y-term makes the landscape elongated (like a narrow
    valley), which causes plain GD to oscillate. Momentum should roll
    straight to the bottom faster.

    Backend analogy: TCP congestion window. TCP builds up "velocity"
    (window size) in directions that are consistently working, rather
    than resetting to 1 on every packet.

    TODO: Fill in the momentum update inside the loop.
    """
    def loss(x, y):
        return (x - 3.0) ** 2 + 10.0 * (y + 2.0) ** 2

    def grad(x, y):
        return np.array([2.0 * (x - 3.0), 20.0 * (y + 2.0)])

    # True minimum: (3, -2)
    start = np.array([-2.0, 3.0])
    lr    = 0.02
    beta  = 0.9      # momentum coefficient — try 0.0, 0.5, 0.9, 0.99
    n_steps = 100

    # --- Plain GD ---
    params_gd  = start.copy()
    losses_gd  = []
    path_gd    = [params_gd.copy()]

    for _ in range(n_steps):
        g = grad(*params_gd)
        params_gd = params_gd - lr * g
        losses_gd.append(loss(*params_gd))
        path_gd.append(params_gd.copy())

    path_gd = np.array(path_gd)

    # --- GD with Momentum ---
    params_m  = start.copy()
    velocity  = np.zeros(2)          # initialise velocity to zero
    losses_m  = []
    path_m    = [params_m.copy()]

    for _ in range(n_steps):
        g = grad(*params_m)

        # --- YOUR CODE HERE ---
        # 1. Update velocity:  velocity = beta * velocity + g
        # 2. Update params:    params_m = params_m - lr * velocity
        velocity = beta * velocity + g
        params_m = params_m - lr * velocity
        # --- END YOUR CODE ---

        losses_m.append(loss(*params_m))
        path_m.append(params_m.copy())

    path_m = np.array(path_m)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    xs = np.linspace(-4, 6, 200)
    ys = np.linspace(-4, 2, 200)
    X, Y = np.meshgrid(xs, ys)
    Z = loss(X, Y)
    for ax in axes:
        ax.contour(X, Y, Z, levels=20, cmap="RdYlGn_r", alpha=0.6)
        ax.plot(3, -2, "r*", markersize=15, label="True min (3,-2)")

    axes[0].plot(path_gd[:, 0], path_gd[:, 1], "b.-", markersize=4, label="Plain GD")
    axes[0].set_title("Plain GD — tends to oscillate in narrow valleys")
    axes[0].legend()

    axes[1].plot(path_m[:, 0], path_m[:, 1], "g.-", markersize=4, label=f"GD+Momentum β={beta}")
    axes[1].set_title("GD with Momentum — smoother path to minimum")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("ex2_momentum.png", dpi=100)
    print("Exercise 2: saved ex2_momentum.png")

    final_loss_gd = losses_gd[-1]
    final_loss_m  = losses_m[-1]
    print(f"  Plain GD final loss:    {final_loss_gd:.6f}")
    print(f"  Momentum  final loss:   {final_loss_m:.6f}")

    # Momentum should converge at least as well as plain GD
    assert final_loss_m <= final_loss_gd * 1.5, \
        "Momentum should converge at least as well as plain GD"
    print("Exercise 2: PASSED\n")


# ---------------------------------------------------------------------------
# Exercise 3 (Challenge): Adam vs Vanilla SGD on Noisy Loss
# ---------------------------------------------------------------------------

def exercise_3_adam_challenge():
    """
    CHALLENGE — Implement Adam from scratch and show it converges faster
    than vanilla SGD on a noisy loss function.

    Problem: minimise f(θ) = sum(θ^2) in 20 dimensions.
    We add large Gaussian noise to the gradient to simulate stochastic
    mini-batch gradients in real training.

    Requirements:
      1. Implement Adam (the full algorithm with m, v, bias correction)
      2. Run both SGD (lr=0.01) and Adam (lr=0.01) for 300 steps
      3. Plot both loss curves on the same figure
      4. Verify Adam reaches a lower loss in fewer steps

    Adam recap:
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g^2
        m_hat = m / (1 - beta1^t)
        v_hat = v / (1 - beta2^t)
        theta = theta - lr * m_hat / (sqrt(v_hat) + eps)

    Default hyperparameters: beta1=0.9, beta2=0.999, eps=1e-8

    Backend analogy: each weight dimension is like a separate microservice
    endpoint with its own traffic volatility.  Adam applies per-parameter
    rate limiting that auto-adapts — busy endpoints get tighter limits,
    quiet ones get looser.
    """
    rng   = np.random.default_rng(99)
    dim   = 20
    noise = 1.0    # large gradient noise to make the problem hard for SGD

    def noisy_grad(theta):
        """True gradient of sum(θ^2) plus noise"""
        return 2.0 * theta + rng.normal(0, noise, size=theta.shape)

    # ---- Vanilla SGD ----
    theta_sgd = rng.normal(0, 2, dim)
    lr_sgd    = 0.01
    sgd_losses = []

    for _ in range(300):
        g = noisy_grad(theta_sgd)
        # Plain SGD: θ = θ - lr * g
        theta_sgd = theta_sgd - lr_sgd * g
        sgd_losses.append(float(np.sum(theta_sgd ** 2)))

    # ---- Adam ----
    theta_adam = rng.normal(0, 2, dim)   # same scale start
    lr_adam    = 0.01
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    m = np.zeros(dim)
    v = np.zeros(dim)
    t = 0
    adam_losses = []

    for _ in range(300):
        g = noisy_grad(theta_adam)

        # --- YOUR CODE HERE ---
        # Implement the Adam update:
        #   1. t += 1
        #   2. Update m (first moment)
        #   3. Update v (second moment)
        #   4. Compute bias-corrected m_hat, v_hat
        #   5. Update theta_adam
        t += 1
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g ** 2
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        theta_adam = theta_adam - lr_adam * m_hat / (np.sqrt(v_hat) + eps)
        # --- END YOUR CODE ---

        adam_losses.append(float(np.sum(theta_adam ** 2)))

    print("Exercise 3: SGD vs Adam on noisy 20-dim problem")
    print(f"  SGD  final loss: {sgd_losses[-1]:.4f}")
    print(f"  Adam final loss: {adam_losses[-1]:.4f}")

    # Plot
    plt.figure(figsize=(8, 4))
    plt.plot(sgd_losses,  label=f"SGD  (lr={lr_sgd})", alpha=0.8)
    plt.plot(adam_losses, label=f"Adam (lr={lr_adam})", alpha=0.8)
    plt.title("Exercise 3: SGD vs Adam on Noisy 20-Dim Quadratic")
    plt.xlabel("Step")
    plt.ylabel("Loss  sum(θ²)")
    plt.yscale("log")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("ex3_adam_vs_sgd.png", dpi=100)
    print("Exercise 3: saved ex3_adam_vs_sgd.png")

    # Adam should significantly outperform SGD on this noisy problem
    assert adam_losses[-1] < sgd_losses[-1], \
        "Adam should converge to a lower loss than vanilla SGD"
    print("Exercise 3: PASSED\n")


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def main():
    print("Module 04 Exercises\n" + "=" * 40)
    exercise_1_learning_rate_tuning()
    exercise_2_momentum()
    exercise_3_adam_challenge()
    print("All exercises passed!")


if __name__ == "__main__":
    main()
